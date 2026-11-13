from copy import deepcopy
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from anthropic import AsyncAnthropic
from anthropic import AsyncStream as AnthropicAsyncStream
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic.lib.streaming._messages import accumulate_event
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI, AsyncStream
from openai import BadRequestError as OpenAIBadRequestError
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.providers import DEFAULT_PROVIDER_BY_ENDPOINT, Provider
from molto.types.anthropic_messages import (
    AnthropicMessagesResponse,
)
from molto.types.exceptions import UnsupportedInput
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsResponse,
)
from molto.types.openai_responses import OpenAIResponsesResponse
from molto.utils.io import dumps_sse, json_loads, load_sse, read_yaml
from molto.utils.test import (
    assert_contains_expected,
    assert_responses_stream_invariants,
    reconstructable_anthropic_event,
    reconstructable_chat_completions_chunks,
    reconstructable_responses_event,
)
from tests.cassette_discovery import (
    MATCHED_CONVERSATIONS,
    NATIVE_CONVERSATIONS,
    Conversation,
    Mode,
    RouteCase,
    parse_cassette,
    public_conversations,
    route_cases,
    select_conversation,
)

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes"
MATCHED_CASSETTE_DIR = CASSETTE_DIR / "matched"
Protocol = Literal["responses", "chat/completions", "messages"]
PROTOCOLS: tuple[Protocol, ...] = ("responses", "chat/completions", "messages")
ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(product(PROTOCOLS, PROTOCOLS))
RESPONSES_MESSAGES_PROTOCOLS: tuple[Protocol, ...] = ("responses", "messages")
RESPONSES_MESSAGES_ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(
    product(RESPONSES_MESSAGES_PROTOCOLS, repeat=2)
)
# Scenarios whose Responses streams carry encrypted reasoning. Messages has no
# equivalents for Responses include or stream options, so these routes compare
# reconstructible requests and events instead of exact wire values.
ENCRYPTED_REASONING_SCENARIOS: tuple[str, ...] = (
    "reasoning_encrypted",
    "input_image_with_reasoning_encrypted",
    "tool_call_with_reasoning_encrypted",
)
TOOL_CALL_SCENARIOS: tuple[str, ...] = (
    "tool_call",
    "tool_call_choice",
    "tool_call_image_result_parallel",
    "tool_call_parallel",
    "tool_call_with_reasoning_encrypted",
)
PROMPT_CACHE_SCENARIOS: tuple[str, ...] = (
    "prompt_cache_implicit",
    "prompt_cache_explicit",
)


def _protocol_name(protocol: Protocol) -> str:
    return {
        "chat/completions": "chat_completions",
        "messages": "messages",
        "responses": "responses",
    }[protocol]


def _provider_from_cassette_path(path: Path) -> Provider:
    return parse_cassette(path).provider


def _matched_cassette_path(
    scenario: str, protocol: Protocol, variant: str, *, provider: Provider | None = None
) -> Path:
    mode, turn = variant.removesuffix(".yaml").rsplit("-", 1)
    conversation = select_conversation(
        MATCHED_CONVERSATIONS, scenario, protocol, cast(Mode, mode), provider=provider
    )
    for cassette in conversation.cassettes:
        if cassette.turn == int(turn):
            return cassette.path
    raise ValueError(f"Missing turn {turn} in {conversation.id}")


def _scenario_routes(scenario: str) -> tuple[tuple[Protocol, Protocol], ...]:
    if scenario in ENCRYPTED_REASONING_SCENARIOS:
        return RESPONSES_MESSAGES_ROUTES
    if scenario == "tool_call_image_result_parallel":
        return (
            *RESPONSES_MESSAGES_ROUTES,
            ("responses", "chat/completions"),
            ("messages", "chat/completions"),
        )
    return ROUTES


def _stream_cases(scenario: str) -> list[Any]:
    return [
        pytest.param(case, id=case.id)
        for case in route_cases(
            MATCHED_CONVERSATIONS, scenario, "stream", _scenario_routes(scenario)
        )
    ]


def _non_stream_cases(scenario: str) -> list[Any]:
    return [
        pytest.param(case, id=case.id)
        for downstream, upstream in _scenario_routes(scenario)
        for case in route_cases(
            MATCHED_CONVERSATIONS,
            scenario,
            "non_stream" if downstream == upstream else "stream",
            [(downstream, upstream)],
        )
    ]


def _default_case(
    scenario: str, downstream: Protocol, upstream: Protocol, mode: Mode
) -> RouteCase:
    return next(
        case
        for case in route_cases(MATCHED_CONVERSATIONS, scenario, mode, [(downstream, upstream)])
        if case.upstream == select_conversation(MATCHED_CONVERSATIONS, scenario, upstream, mode)
    )


# Retain the existing native SDK baselines. Other matched scenarios already
# exercise native routes; faithful provider-deviation recordings join separately.
NATIVE_SDK_CONVERSATIONS = (
    *(
        conversation
        for conversation in MATCHED_CONVERSATIONS
        if conversation.scenario
        in {
            "simple_text",
            "sampling_tests",
            "tool_call",
            "tool_call_image_result_parallel",
            *PROMPT_CACHE_SCENARIOS,
        }
        or conversation.mode == "non_stream"
    ),
    *NATIVE_CONVERSATIONS,
)


@lru_cache(maxsize=5)
def _read_cassette(path: Path) -> dict[str, Any]:
    """Read a cassette YAML file."""
    return read_yaml(path)


def _get_http_client(
    cassette_paths: list[Path],
    upstream: Protocol = "responses",
    *,
    translated: bool = False,
    expected_body: dict[str, Any] | None = None,
    expected_extra_body: dict[str, Any] | None = None,
    requests: list[Request] | None = None,
    response_headers: dict[str, str] | None = None,
    continued: bool = False,
) -> AsyncClient:
    """Replay `cassette_paths`, optionally asserting one exact outgoing request body."""
    cassette_turns = iter(enumerate(cassette_paths))

    def cassette_handler(request: Request) -> Response:
        if requests is not None:
            requests.append(request)
        turn, cassette_path = next(cassette_turns)
        cassette = _read_cassette(cassette_path)
        scenario = cassette_path.parent.name
        request_body = json_loads(request.content)
        try:
            assert request.method == cassette["request"]["method"]
            origin = (
                "https://api.anthropic.com" if upstream == "messages" else "https://api.openai.com"
            )
            assert str(request.url) == f"{origin}{cassette['request']['path']}"
            assert request.url.path == cassette["request"]["path"]
            expected = expected_body
            if expected is None:
                expected = cassette["request"]["body"]
            expected = {**expected, **(expected_extra_body or {})}
            if translated and upstream == "responses":
                expected = _without_reasoning_item_status(expected)
            if continued and turn > 0:
                key = "input" if upstream == "responses" else "messages"
                request_body[key] = _comparable_history(
                    request_body[key], upstream, translated=translated
                )
                expected[key] = _comparable_history(
                    _without_reasoning_item_status(expected).get(key),
                    upstream,
                    translated=translated,
                )
            if (
                translated
                and upstream == "responses"
                and scenario in ENCRYPTED_REASONING_SCENARIOS
            ):
                _assert_reconstructable_encrypted_responses_request(
                    request_body,
                    expected,
                )
            elif translated and upstream == "responses" and scenario in TOOL_CALL_SCENARIOS:
                _assert_reconstructable_tool_responses_request(request_body, expected)
            elif translated and upstream == "chat/completions" and scenario in TOOL_CALL_SCENARIOS:
                _assert_reconstructable_tool_chat_request(request_body, expected)
            else:
                assert request_body == expected
        except AssertionError as e:
            return Response(
                400,
                content=f"Bad request: {repr(e)}".encode(),
                headers={"content-type": "text/plain; charset=utf-8"},
            )
        cassette_response = cassette["response"]
        if "sse" in cassette_response:
            return Response(
                cassette_response["status_code"],
                content="".join(cassette_response["sse"]).encode(),
                headers={
                    "content-type": "text/event-stream; charset=utf-8",
                    **(response_headers or {}),
                },
            )
        return Response(
            cassette_response["status_code"],
            json=cassette_response["body"],
            headers={"content-type": "application/json", **(response_headers or {})},
        )

    return AsyncClient(transport=MockTransport(cassette_handler))


def _without_reasoning_item_status(body: dict[str, Any]) -> dict[str, Any]:
    # Native recordings retain their original fields; translated requests must
    # omit the output-only reasoning status rejected by the Responses API.
    body = deepcopy(body)
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            item.pop("status", None)
    return body


def _assert_reconstructable_encrypted_responses_request(
    actual_body: dict[str, Any],
    expected_body: dict[str, Any],
) -> None:
    actual = deepcopy(actual_body)
    expected = _without_reasoning_item_status(expected_body)
    for body in (actual, expected):
        # These exact options have no Messages equivalent. Inclusion already
        # preserves opaque state, store=false requests no state, and disabling
        # obfuscation affects only SSE padding. Other values must still match.
        for key, value in (
            ("include", ["reasoning.encrypted_content"]),
            ("store", False),
            ("stream_options", {"include_obfuscation": False}),
        ):
            if body.get(key) == value:
                body.pop(key)
        for item in body["input"]:
            item_type = item.get("type", "message")
            if item_type in {"message", "reasoning", "function_call"}:
                # Messages cannot carry Responses output-item IDs or phase.
                # Portable function call_id values remain part of the comparison.
                item.pop("id", None)
                if item_type == "message":
                    item.pop("phase", None)
            if item_type == "reasoning":
                assert "status" not in item, "Reconstructed reasoning input has output-only status"
                # Omitted and empty raw content both mean no raw reasoning.
                item.setdefault("content", [])
            elif item_type == "message":
                item.setdefault("type", "message")
                if item["role"] == "assistant":
                    # Easy assistant text is one completed output_text part, not
                    # a license to join multiple parts or separate messages.
                    item.setdefault("status", "completed")
                    if isinstance(item["content"], str):
                        item["content"] = [
                            {
                                "type": "output_text",
                                "text": item["content"],
                                "annotations": [],
                                "logprobs": [],
                            }
                        ]
    assert actual == expected


@pytest.mark.parametrize(
    "defect",
    [
        "token-limit",
        "call-id",
        "encrypted-content",
        "input-order",
        "cache-key",
        "cache-lifetime",
        "reasoning-status",
        "part-order",
        "part-boundaries",
    ],
)
def test_encrypted_request_comparison_rejects_semantic_changes(defect: str) -> None:
    """Ensures narrow wire equivalences cannot hide changed history or request settings."""
    path = _matched_cassette_path(
        "tool_call_with_reasoning_encrypted", "responses", "stream-01.yaml"
    )
    expected = deepcopy(_read_cassette(path)["request"]["body"])
    expected.update(prompt_cache_key="conversation", prompt_cache_options={"ttl": "30m"})
    expected["input"][1]["status"] = "completed"
    # Two parts with the same joined text must still retain their boundaries.
    expected["input"][2]["content"] = [
        {"type": "output_text", "text": "First", "annotations": [], "logprobs": []},
        {"type": "output_text", "text": "Second", "annotations": [], "logprobs": []},
    ]
    actual = deepcopy(expected)
    for item in actual["input"]:
        item.pop("id", None)
        item.pop("phase", None)
    actual["input"][1].pop("status")
    original_expected = deepcopy(expected)
    original_actual = deepcopy(actual)
    _assert_reconstructable_encrypted_responses_request(actual, expected)
    assert actual == original_actual and expected == original_expected

    if defect == "token-limit":
        actual.pop("max_output_tokens")
    elif defect == "call-id":
        actual["input"][3]["call_id"] = "wrong-call"
    elif defect == "encrypted-content":
        actual["input"][1]["encrypted_content"] = "wrong-opaque-state"
    elif defect == "input-order":
        actual["input"][1:3] = reversed(actual["input"][1:3])
    elif defect == "cache-key":
        actual.pop("prompt_cache_key")
    elif defect == "cache-lifetime":
        actual["prompt_cache_options"]["ttl"] = "5m"
    elif defect == "reasoning-status":
        actual["input"][1]["status"] = "completed"
    elif defect == "part-order":
        actual["input"][2]["content"].reverse()
    elif defect == "part-boundaries":
        actual["input"][2]["content"] = [
            {"type": "output_text", "text": "FirstSecond", "annotations": [], "logprobs": []}
        ]
    with pytest.raises(AssertionError):
        _assert_reconstructable_encrypted_responses_request(actual, expected)


def _assert_reconstructable_tool_responses_request(
    actual_body: dict[str, Any],
    expected_body: dict[str, Any],
) -> None:
    def without_history_item_ids(body: dict[str, Any]) -> dict[str, Any]:
        body = _without_irrelevant_parallel_tool_calls(body)
        for item in body.get("input") or []:
            if isinstance(item, dict):
                item.pop("id", None)
                item.pop("phase", None)
        return body

    # Chat Completions and Messages preserve call IDs but expose no separate
    # Responses output-item IDs or commentary/final phases for assistant history.
    assert without_history_item_ids(actual_body) == without_history_item_ids(expected_body)


def _assert_reconstructable_tool_chat_request(
    actual_body: dict[str, Any],
    expected_body: dict[str, Any],
) -> None:
    assert _without_irrelevant_parallel_tool_calls(
        actual_body
    ) == _without_irrelevant_parallel_tool_calls(expected_body)


def _without_irrelevant_parallel_tool_calls(body: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(body)
    if body.get("tool_choice") == "none":
        # Messages cannot encode parallel-call control when tool use is disabled,
        # and the setting cannot affect a request that forbids every tool call.
        body.pop("parallel_tool_calls", None)
    return body


def _reconstructable_encrypted_responses_event(event: dict[str, Any]) -> dict[str, Any]:
    event.pop("obfuscation", None)

    item = event.get("item")
    if isinstance(item, dict):
        item.pop("status", None)

    response = event.get("response")
    if isinstance(response, dict):
        response.pop("max_output_tokens", None)
        response.pop("reasoning", None)
        for output_item in response.get("output") or []:
            output_item.pop("status", None)
            # output_item.pop("encrypted_content", None)

    return event


def _reconstructable_tool_responses_event(event: dict[str, Any]) -> dict[str, Any]:
    response = event.get("response")
    if not isinstance(response, dict):
        return event

    # These response-envelope values are server defaults or adjustments and are
    # not present in Chat Completions or Messages stream events.
    for field_name in ("max_output_tokens", "reasoning", "service_tier"):
        response.pop(field_name, None)
    for tool in response.get("tools") or []:
        if isinstance(tool, dict):
            tool.pop("output_schema", None)
    return event


def _assert_sse_events_contain(
    actual_sse_events: list[str], expected_sse_events: list[str]
) -> None:
    assert len(actual_sse_events) == len(expected_sse_events)
    for index, (actual_sse, expected_sse) in enumerate(
        zip(actual_sse_events, expected_sse_events, strict=True)
    ):
        assert_contains_expected(
            load_sse(actual_sse),
            load_sse(expected_sse),
            path=f"event[{index}]",
        )


def _sse_records(cassette: dict[str, Any], protocol: Protocol) -> list[str]:
    records = []
    for event in "".join(cassette["response"]["sse"]).split("\n\n"):
        event = event.strip()
        if (
            event
            and event != "data: [DONE]"
            and not (protocol == "messages" and load_sse(event).get("type") == "ping")
        ):
            records.append(f"{event}\n\n")
    return records


def _completed_responses_body(cassette_path: Path) -> dict[str, Any]:
    cassette = _read_cassette(cassette_path)
    response = cassette["response"]
    if "body" in response:
        return response["body"]
    for record in reversed(_sse_records(cassette, "responses")):
        event = load_sse(record)
        if event.get("type") in {"response.completed", "response.incomplete"}:
            return event["response"]
    raise AssertionError("Responses cassette has no terminal response event.")


def _expect_responses_search_usage(events: list[dict[str, Any]], upstream_path: Path) -> None:
    """Use source-reported search counts even when the matched Messages fixture omits them."""
    response = _completed_responses_body(upstream_path)
    search_usage = (response.get("tool_usage") or {}).get("web_search") or {}
    count = search_usage.get("num_requests") if response.get("usage") is not None else None
    for event in events:
        if event["type"] == "message_delta":
            if count is None:
                event["usage"].pop("server_tool_use", None)
            else:
                event["usage"]["server_tool_use"] = {
                    "web_search_requests": count,
                    "web_fetch_requests": 0,
                }


@pytest.mark.parametrize(
    "scenario", ["reasoning_raw", "reasoning_encrypted", "tool_call_with_reasoning_encrypted"]
)
def test_responses_stream_updates_require_the_started_item_id(scenario: str) -> None:
    """Ensures every text, reasoning, summary, and argument update retains its item's identity."""
    path = _matched_cassette_path(scenario, "responses", "stream-00.yaml")
    records = _sse_records(_read_cassette(path), "responses")
    assert_responses_stream_invariants(records)
    events = [load_sse(record) for record in records]
    item_ids = [
        event["item"]["id"] for event in events if event["type"] == "response.output_item.added"
    ]
    # One representative of every update type covers each wire shape without
    # repeating the same assertion for every text token in the recording.
    for event_type in {event["type"] for event in events if "item_id" in event}:
        changed = deepcopy(events)
        update = next(event for event in changed if event["type"] == event_type)
        update["item_id"] = next(item_id for item_id in item_ids if item_id != update["item_id"])
        with pytest.raises(AssertionError, match="Update targets another item"):
            assert_responses_stream_invariants([dumps_sse(event) for event in changed])


@pytest.mark.parametrize(
    "defect",
    [
        "before-start",
        "after-completion",
        "wrong-index",
        "done-id",
        "duplicate-start",
        "missing-completion",
        "terminal-order",
        "terminal-content",
    ],
)
def test_responses_stream_invariants_reject_invalid_item_lifecycles(defect: str) -> None:
    """Ensures balanced item lifecycles and ordered terminal output are checked before ID removal."""
    path = _matched_cassette_path(
        "tool_call_with_reasoning_encrypted", "responses", "stream-00.yaml"
    )
    records = _sse_records(_read_cassette(path), "responses")
    assert_responses_stream_invariants(records)
    events = [load_sse(record) for record in records]
    update_index = next(i for i, event in enumerate(events) if "item_id" in event)
    output_index = events[update_index]["output_index"]
    start_index = next(
        i
        for i, event in enumerate(events)
        if event["type"] == "response.output_item.added" and event["output_index"] == output_index
    )
    done_index = next(
        i
        for i, event in enumerate(events)
        if event["type"] == "response.output_item.done" and event["output_index"] == output_index
    )
    if defect == "before-start":
        events.insert(start_index, events.pop(update_index))
    elif defect == "after-completion":
        events.insert(done_index, events.pop(update_index))
    elif defect == "wrong-index":
        events[update_index]["output_index"] = 0
    elif defect == "done-id":
        events[done_index]["item"]["id"] = "wrong-id"
    elif defect == "duplicate-start":
        events.insert(start_index, deepcopy(events[start_index]))
    elif defect == "missing-completion":
        events.pop(done_index)
    elif defect == "terminal-order":
        events[-1]["response"]["output"].reverse()
    elif defect == "terminal-content":
        events[-1]["response"]["output"][1]["content"][0]["text"] = "wrong-text"
    # Keep sequence numbers valid so a failure demonstrates the item check.
    for index, event in enumerate(events):
        event["sequence_number"] = index
    with pytest.raises(AssertionError):
        assert_responses_stream_invariants([dumps_sse(event) for event in events])


def test_incomplete_responses_stream_requires_completed_items() -> None:
    """Ensures incomplete terminal responses also require a done event for each partial item."""
    path = CASSETTE_DIR / "errors/incomplete_response/openai-responses-stream-00.yaml"
    records = _sse_records(_read_cassette(path), "responses")
    assert_responses_stream_invariants(records)
    events = [load_sse(record) for record in records]
    events = [event for event in events if event["type"] != "response.output_item.done"]
    for index, event in enumerate(events):
        event["sequence_number"] = index
    with pytest.raises(AssertionError, match="Unbalanced output item"):
        assert_responses_stream_invariants([dumps_sse(event) for event in events])


def test_responses_terminal_encrypted_snapshot_policy() -> None:
    """Ensures only native streams may expose a separate terminal encrypted snapshot."""
    path = _matched_cassette_path("reasoning_encrypted", "responses", "stream-00.yaml")
    events = [load_sse(record) for record in _sse_records(_read_cassette(path), "responses")]
    events[-1]["response"]["output"][0]["encrypted_content"] = "separate-terminal-snapshot"
    records = [dumps_sse(event) for event in events]
    assert_responses_stream_invariants(records, native=True)
    with pytest.raises(AssertionError, match="Terminal output differs"):
        assert_responses_stream_invariants(records)


@pytest.mark.parametrize("field", ["id", "status", "query"])
def test_responses_terminal_search_snapshot_policy(field: str) -> None:
    """Ensures only native search sources may change while identity and action stay strict."""
    path = _matched_cassette_path("web_search", "responses", "stream-00.yaml")
    events = [load_sse(record) for record in _sse_records(_read_cassette(path), "responses")]
    output = events[-1]["response"]["output"]
    for event in events:
        if event["type"] == "response.output_item.done" and event["item"]["type"] == "reasoning":
            # Isolate the search-source difference from the independent
            # terminal reasoning snapshot in this recording.
            output[event["output_index"]] = deepcopy(event["item"])
    records = [dumps_sse(event) for event in events]
    assert_responses_stream_invariants(records, native=True)
    with pytest.raises(AssertionError, match="Terminal output differs"):
        assert_responses_stream_invariants(records)

    search = next(item for item in output if item["type"] == "web_search_call")
    if field == "query":
        search["action"]["query"] = "changed query"
    else:
        search[field] = "changed"
    with pytest.raises(AssertionError, match="Terminal output differs"):
        assert_responses_stream_invariants([dumps_sse(event) for event in events], native=True)


async def test_extra_inputs():
    """Ensures translated requests reject unknown input properties by default."""
    cassette_path = _matched_cassette_path("simple_text", "responses", "stream-00.yaml")

    async with _get_http_client([cassette_path]) as http_client:
        client = Molto(http_client=http_client)

        # --- Extra inputs --- #
        cassette = _read_cassette(cassette_path)
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            await client.responses(
                endpoint="chat/completions",
                body=dict(
                    messages=[
                        {"role": "user", "content": "Write one short sentence about the ocean."}
                    ],
                    **cassette["request"]["body"],
                ),
            )


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_transport_options_and_response_headers(
    downstream: Protocol,
    upstream: Protocol,
    stream: bool,
):
    translated = downstream != upstream
    mode = "stream" if stream or translated else "non_stream"
    variant = f"{mode}-00.yaml"
    downstream_path = _matched_cassette_path("simple_text", downstream, variant)
    upstream_path = _matched_cassette_path("simple_text", upstream, variant)
    requests: list[Request] = []
    extra_body = {"client_metadata": {"session_id": "session-123"}}
    extra_headers = {
        "Accept": "client-facing/type",
        "Accept-Encoding": "client-facing-encoding",
        "Authorization": "Bearer client-facing-key",
        "Connection": "close, x-connection-specific",
        "Content-Length": "1",
        "Host": "client-facing.example",
        "OpenAI-Organization": "client-facing-organization",
        "Proxy-Authorization": "Bearer client-facing-proxy-key",
        "Transfer-Encoding": "chunked",
        "X-Api-Key": "client-facing-api-key",
        "X-Codex-Window-Id": "window-123",
        "X-Connection-Specific": "remove-me",
        "X-OAI-Attestation": "client-facing-attestation",
    }

    async with _get_http_client(
        [upstream_path],
        upstream,
        translated=translated,
        expected_extra_body=extra_body,
        requests=requests,
        response_headers={
            "x-codex-turn-state": "turn-state-123",
            "x-request-id": "request-123",
        },
    ) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        request_body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
        if translated and not stream:
            request_body["stream"] = False
        result = await coordinator(
            body=request_body,
            endpoint=upstream,
            provider=_provider_from_cassette_path(upstream_path),
            api_key="test-api-key",
            extra_body=extra_body,
            extra_headers=extra_headers,
        )

        if stream:
            assert result.response_headers["x-codex-turn-state"] == "turn-state-123"
            assert result.response_headers["x-request-id"] == "request-123"
            assert [event async for event in result]
        else:
            assert isinstance(result, BaseModel)

    assert len(requests) == 1
    upstream_request = requests[0]
    assert upstream_request.headers["x-codex-window-id"] == "window-123"
    assert upstream_request.headers["accept"] != "client-facing/type"
    assert upstream_request.headers["accept-encoding"] != "client-facing-encoding"
    assert upstream_request.headers.get("connection") != "close, x-connection-specific"
    assert "openai-organization" not in upstream_request.headers
    assert "proxy-authorization" not in upstream_request.headers
    assert "transfer-encoding" not in upstream_request.headers
    assert "x-connection-specific" not in upstream_request.headers
    assert "x-oai-attestation" not in upstream_request.headers
    assert upstream_request.headers["host"] != "client-facing.example"
    assert upstream_request.headers["content-length"] != "1"
    if upstream == "messages":
        assert upstream_request.headers["x-api-key"] == "test-api-key"
        assert "authorization" not in upstream_request.headers
    else:
        assert upstream_request.headers["authorization"] == "Bearer test-api-key"
        assert "x-api-key" not in upstream_request.headers


@pytest.mark.parametrize("upstream", ["chat/completions", "messages"])
async def test_responses_translation_ignores_disabled_store(upstream: Protocol):
    variant = "stream-00.yaml"
    downstream_path = _matched_cassette_path("simple_text", "responses", variant)
    upstream_path = _matched_cassette_path("simple_text", upstream, variant)
    request_body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
    request_body["store"] = False

    async with _get_http_client([upstream_path], upstream, translated=True) as http_client:
        client = Molto(http_client=http_client)
        stream = await client.responses(
            body=request_body,
            endpoint=upstream,
            provider=_provider_from_cassette_path(upstream_path),
            api_key="test-api-key",
        )

        assert [event async for event in stream]


@pytest.mark.parametrize(
    "conversation",
    [c for c in NATIVE_SDK_CONVERSATIONS if c.mode == "stream"],
    ids=lambda c: f"{c.paths[0].parent.parent.name}-{c.id}",
)
async def test_native_stream_matches_provider_sdk(conversation: Conversation):
    """Ensures every discovered native turn preserves parsed SDK events and provider selection."""
    for cassette_path in conversation.paths:
        await _assert_native_stream_matches_provider_sdk(cassette_path, conversation.endpoint)


async def _assert_native_stream_matches_provider_sdk(cassette_path: Path, protocol: Protocol):
    cassette = _read_cassette(cassette_path)
    request_body = deepcopy(cassette["request"]["body"])

    async with _get_http_client([cassette_path], protocol) as http_client:
        if protocol == "messages":
            sdk_client = AsyncAnthropic(
                api_key="test-api-key",
                base_url="https://api.anthropic.com",
                http_client=http_client,
            )
            sdk_stream = await sdk_client.messages.create(**request_body)
            assert isinstance(sdk_stream, AnthropicAsyncStream)
        else:
            sdk_client = AsyncOpenAI(
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                http_client=http_client,
            )
            if protocol == "responses":
                sdk_stream = await sdk_client.responses.create(**request_body)
            else:
                sdk_stream = await sdk_client.chat.completions.create(**request_body)
            assert isinstance(sdk_stream, AsyncStream)
        async with sdk_stream:
            sdk_events = [event async for event in sdk_stream]

    async with _get_http_client([cassette_path], protocol) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[protocol]
        molto_stream = await coordinator(
            body=request_body,
            endpoint=protocol,
            provider=_provider_from_cassette_path(cassette_path),
            api_key="test-api-key",
        )
        molto_events = [event async for event in molto_stream]

    assert [type(event) for event in molto_events] == [type(event) for event in sdk_events]
    assert [event.model_dump(mode="json", exclude_unset=True) for event in molto_events] == [
        event.model_dump(mode="json", exclude_unset=True) for event in sdk_events
    ]


@pytest.mark.parametrize(
    "conversation",
    [c for c in NATIVE_SDK_CONVERSATIONS if c.mode == "non_stream"],
    ids=lambda c: f"{c.paths[0].parent.parent.name}-{c.id}",
)
async def test_native_non_streaming_responses_match_provider_sdk(conversation: Conversation):
    """Ensures discovered complete native responses retain SDK parity and provider selection."""
    protocol = conversation.endpoint
    cassette_paths = conversation.paths
    request_bodies = [
        deepcopy(_read_cassette(cassette_path)["request"]["body"])
        for cassette_path in cassette_paths
    ]
    sdk_responses = []

    async with _get_http_client(cassette_paths, protocol) as http_client:
        if protocol == "messages":
            sdk_client = AsyncAnthropic(
                api_key="test-api-key",
                base_url="https://api.anthropic.com",
                http_client=http_client,
            )
            for request_body in request_bodies:
                sdk_response = await sdk_client.messages.create(**request_body)
                assert not isinstance(sdk_response, AnthropicAsyncStream)
                sdk_responses.append(sdk_response)
        else:
            sdk_client = AsyncOpenAI(
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                http_client=http_client,
            )
            for request_body in request_bodies:
                if protocol == "responses":
                    sdk_response = await sdk_client.responses.create(**request_body)
                else:
                    sdk_response = await sdk_client.chat.completions.create(**request_body)
                assert not isinstance(sdk_response, AsyncStream)
                sdk_responses.append(sdk_response)

    async with _get_http_client(cassette_paths, protocol) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[protocol]
        molto_responses = [
            await coordinator(
                body=request_body,
                endpoint=protocol,
                provider=conversation.provider,
                api_key="test-api-key",
            )
            for request_body in request_bodies
        ]

    response_type = {
        "responses": OpenAIResponsesResponse,
        "chat/completions": OpenAIChatCompletionsResponse,
        "messages": AnthropicMessagesResponse,
    }[protocol]
    for molto_response, sdk_response in zip(molto_responses, sdk_responses, strict=True):
        assert type(molto_response) is response_type
        assert type(molto_response) is type(sdk_response)
        assert molto_response.model_dump(
            mode="json", exclude_unset=True
        ) == sdk_response.model_dump(mode="json", exclude_unset=True)


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
async def test_discovered_native_runners_forward_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: Mode
) -> None:
    """Ensures both native runners pass a discovered non-default provider to the coordinator."""
    path = tmp_path / "native" / "simple_text" / f"vllm-chat_completions-{mode}-00.yaml"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        _matched_cassette_path("simple_text", "chat/completions", f"{mode}-00.yaml").read_bytes()
    )
    (conversation,) = public_conversations(tmp_path / "native")
    providers = []
    original = Molto.chat_completions

    async def capture_provider(self, **kwargs):
        providers.append(kwargs.get("provider"))
        return await original(self, **kwargs)

    monkeypatch.setattr(Molto, "chat_completions", capture_provider)
    if mode == "stream":
        await test_native_stream_matches_provider_sdk(conversation)
    else:
        await test_native_non_streaming_responses_match_provider_sdk(conversation)
    assert providers == ["vllm"]


async def _native_result(
    scenario: str, protocol: Literal["responses", "chat/completions"], mode: Mode
) -> tuple[dict[str, Any], Any]:
    """Replay a single-turn native OpenAI recording through Molto and return its request and result.

    The replay transport rejects any outgoing body that differs from the recording, so
    each caller also proves the native options were forwarded exactly.
    """
    conversation = select_conversation(NATIVE_CONVERSATIONS, scenario, protocol, mode)
    (cassette_path,) = conversation.paths
    body = deepcopy(_read_cassette(cassette_path)["request"]["body"])
    async with _get_http_client([cassette_path], protocol) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
        }[protocol]
        result = await coordinator(
            body=body,
            endpoint=protocol,
            provider=conversation.provider,
            api_key="test-api-key",
        )
        if mode == "non_stream":
            return body, result
        events = [event async for event in result]
    return body, _final_stream_result(events, protocol)


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
@pytest.mark.parametrize(
    "scenario", ["response_format_json_object", "response_format_json_schema"]
)
async def test_native_chat_response_format_returns_requested_json(
    scenario: str, mode: Mode
) -> None:
    """Ensures native Chat response formats return a JSON object.

    A JSON-schema format must also return exactly the schema's required fields.
    """
    body, completion = await _native_result(scenario, "chat/completions", mode)
    output = json_loads(completion.choices[0].message.content)
    assert isinstance(output, dict)
    schema = body["response_format"].get("json_schema", {}).get("schema")
    if schema is not None:
        assert set(output) == set(schema["required"])


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
async def test_native_chat_model_settings_return_logprobs(mode: Mode) -> None:
    """Ensures explicit native Chat settings return token logprobs with the requested alternatives."""
    body, completion = await _native_result("model_settings", "chat/completions", mode)
    logprobs = completion.choices[0].logprobs
    assert logprobs is not None and logprobs.content
    assert (
        "".join(token.token for token in logprobs.content) == completion.choices[0].message.content
    )
    assert all(len(token.top_logprobs) == body["top_logprobs"] for token in logprobs.content)


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
async def test_native_chat_required_tool_choice_returns_a_tool_call(mode: Mode) -> None:
    """Ensures a native Chat required tool choice returns a call to a requested tool."""
    body, completion = await _native_result("tool_choice_required", "chat/completions", mode)
    assert body["tool_choice"] == "required"
    (choice,) = completion.choices
    assert choice.finish_reason == "tool_calls"
    (call,) = choice.message.tool_calls
    assert call.function.name in {tool["function"]["name"] for tool in body["tools"]}
    assert isinstance(json_loads(call.function.arguments), dict)


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
async def test_native_responses_text_format_returns_requested_json(mode: Mode) -> None:
    """Ensures a native Responses JSON-schema format is echoed and returns its required fields."""
    body, response = await _native_result("text_format_json_schema", "responses", mode)
    requested = body["text"]["format"]
    assert response.text.format.model_dump(by_alias=True, exclude_unset=True) == requested
    assert response.text.verbosity == "medium"
    assert set(json_loads(response.output_text)) == set(requested["schema"]["required"])


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
async def test_native_responses_model_settings_are_echoed_with_logprobs_and_sources(
    mode: Mode,
) -> None:
    """Ensures explicit native Responses settings are echoed and requested includes are returned."""
    body, response = await _native_result("model_settings", "responses", mode)
    for setting in (
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "parallel_tool_calls",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "service_tier",
        "store",
        "temperature",
        "tool_choice",
        "top_logprobs",
        "top_p",
        "truncation",
    ):
        assert getattr(response, setting) == body[setting], setting
    (message,) = [item for item in response.output if item.type == "message"]
    (text,) = message.content
    assert text.logprobs
    assert all(len(token.top_logprobs) == body["top_logprobs"] for token in text.logprobs)
    (search,) = [item for item in response.output if item.type == "web_search_call"]
    assert search.status == "completed"
    assert search.action.sources
    assert {source.type for source in search.action.sources} == {"url"}


def _recorded_responses_output(cassette: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a recorded Responses output from its JSON body or terminal SSE event."""
    if "body" in cassette["response"]:
        return cassette["response"]["body"]["output"]
    return load_sse(_sse_records(cassette, "responses")[-1])["response"]["output"]


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
def test_native_responses_api_search_source_history_is_recorded(mode: Mode) -> None:
    """Ensures the `api` source scenario records that source and its replay as history.

    OpenAI answers a weather search with `{"type": "api", "name": "oai-weather"}`, which
    the pinned SDK does not declare. The second turn sends that output back as history,
    and OpenAI accepted it. The generic native runners replay both turns through Molto
    with exact request-body and SDK parity checks.
    """
    conversation = select_conversation(
        NATIVE_CONVERSATIONS, "web_search_api_source", "responses", mode
    )
    first, second = (_read_cassette(path) for path in conversation.paths)
    for cassette in (first, second):
        (search,) = [
            item
            for item in _recorded_responses_output(cassette)
            if item["type"] == "web_search_call"
        ]
        assert search["status"] == "completed"
        assert search["action"]["sources"] == [{"type": "api", "name": "oai-weather"}]
    assert second["request"]["body"]["input"] == [
        *first["request"]["body"]["input"],
        *_recorded_responses_output(first),
        {"role": "user", "content": "And tomorrow?"},
    ]


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
async def test_native_responses_web_search_options_are_echoed(mode: Mode) -> None:
    """Ensures native web-search options are echoed and unrequested sources stay omitted."""
    body, response = await _native_result("web_search_tool_options", "responses", mode)
    (requested,) = body["tools"]
    (tool,) = response.tools
    assert {
        key: value for key, value in tool.model_dump().items() if key in requested
    } == requested
    searches = [item for item in response.output if item.type == "web_search_call"]
    assert searches
    for search in searches:
        assert search.status == "completed"
        assert search.action.type == "search" and search.action.queries
        assert search.action.sources is None


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
async def test_native_responses_conversation_id_is_forwarded(mode: Mode) -> None:
    """Ensures a provider-managed conversation ID is forwarded and returned with its history."""
    body, response = await _native_result("conversation_id", "responses", mode)
    setup = _read_cassette(
        CASSETTE_DIR / "native/conversation_id/setup" / f"openai-conversations-{mode}.yaml"
    )
    # Each mode created its own conversation; never pair one mode's ID with the other.
    assert body["conversation"] == setup["response"]["body"]["id"]
    assert response.conversation.id == body["conversation"]
    assert response.output_text


@pytest.mark.parametrize("mode", ["stream", "non_stream"])
async def test_native_responses_required_tool_choice_returns_a_function_call(mode: Mode) -> None:
    """Ensures a native Responses required tool choice returns a call to a requested tool."""
    body, response = await _native_result("tool_choice_required", "responses", mode)
    assert response.tool_choice == "required"
    (call,) = response.output
    assert call.type == "function_call"
    assert call.name in {tool["name"] for tool in body["tools"]}
    assert isinstance(json_loads(call.arguments), dict)


def _assert_matched_response(
    response: BaseModel,
    downstream: Protocol,
    expected: dict[str, Any] | OpenAIResponsesResponse,
) -> None:
    # Parse the reference with the SDK, independently of Molto's translators and
    # collectors. These scenarios contain ordered text and encrypted reasoning.
    expected_response = (
        expected
        if isinstance(expected, OpenAIResponsesResponse)
        else OpenAIResponsesResponse.model_validate(expected)
    )
    expected_usage = expected_response.usage
    assert expected_usage is not None
    expected_input_details = expected_usage.input_tokens_details
    expected_output_details = expected_usage.output_tokens_details
    cache_write_tokens = expected_input_details.cache_write_tokens or 0
    cached_tokens = expected_input_details.cached_tokens
    reasoning_tokens = expected_output_details.reasoning_tokens
    response_type = {
        "responses": OpenAIResponsesResponse,
        "chat/completions": OpenAIChatCompletionsResponse,
        "messages": AnthropicMessagesResponse,
    }[downstream]
    assert type(response) is response_type
    assert response.id == expected_response.id
    assert response.model == expected_response.model

    if downstream == "responses":
        assert isinstance(response, OpenAIResponsesResponse)
        assert [item.type for item in response.output] == [
            item.type for item in expected_response.output
        ]
        for item, reference in zip(response.output, expected_response.output, strict=True):
            if item.type == "message":
                assert item.role == reference.role == "assistant"
                assert item.status == reference.status
                assert all(part.type == "output_text" for part in item.content)
                assert all(part.type == "output_text" for part in reference.content)
                assert [part.text for part in item.content] == [
                    part.text for part in reference.content
                ]
            elif item.type == "reasoning":
                assert (item.content or []) == (reference.content or [])
                assert item.summary == reference.summary
                assert item.encrypted_content == reference.encrypted_content
            else:
                raise AssertionError(f"Unsupported matched response item: {item.type}")
        assert response.status == expected_response.status
        assert response.incomplete_details == expected_response.incomplete_details
        assert response.usage is not None
        assert response.usage.input_tokens == expected_usage.input_tokens
        assert response.usage.output_tokens == expected_usage.output_tokens
        assert response.usage.total_tokens == expected_usage.total_tokens
        assert response.usage.input_tokens_details.cached_tokens == cached_tokens
        assert (response.usage.input_tokens_details.cache_write_tokens or 0) == cache_write_tokens
        assert response.usage.output_tokens_details.reasoning_tokens == reasoning_tokens
    elif downstream == "chat/completions":
        assert isinstance(response, OpenAIChatCompletionsResponse)
        # Chat has only one scalar text field. Require the fixture's single text
        # item/part instead of silently flattening unsupported output boundaries.
        assert len(expected_response.output) == 1
        reference = expected_response.output[0]
        assert reference.type == "message"
        assert len(reference.content) == 1
        assert reference.content[0].type == "output_text"
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.index == 0
        assert choice.message.role == reference.role == "assistant"
        assert choice.message.content == reference.content[0].text
        assert not choice.message.tool_calls
        assert choice.message.refusal is None
        assert not getattr(choice.message, "reasoning_content", None)
        if expected_response.status == "incomplete":
            assert expected_response.incomplete_details.reason == "max_output_tokens"
            assert choice.finish_reason == "length"
        else:
            assert expected_response.status == "completed"
            assert choice.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.prompt_tokens == expected_usage.input_tokens
        assert response.usage.completion_tokens == expected_usage.output_tokens
        assert response.usage.total_tokens == expected_usage.total_tokens
        assert response.usage.prompt_tokens_details.cached_tokens == cached_tokens
        assert (response.usage.prompt_tokens_details.cache_write_tokens or 0) == cache_write_tokens
        assert response.usage.completion_tokens_details.reasoning_tokens == reasoning_tokens
    else:
        assert isinstance(response, AnthropicMessagesResponse)
        expected_content = []
        for item in expected_response.output:
            if item.type == "message":
                assert item.role == "assistant"
                for part in item.content:
                    assert part.type == "output_text"
                    expected_content.append({"type": "text", "text": part.text})
            elif item.type == "reasoning":
                assert not item.content
                # The retained encrypted fixtures have at most one summary part;
                # Messages cannot expose separate summary-part boundaries.
                assert len(item.summary) <= 1
                expected_content.append(
                    {
                        "type": "thinking",
                        "thinking": item.summary[0].text if item.summary else "",
                        "signature": item.encrypted_content,
                    }
                )
            else:
                raise AssertionError(f"Unsupported matched response item: {item.type}")
        assert [block.model_dump(exclude_none=True) for block in response.content] == (
            expected_content
        )
        assert response.role == "assistant"
        assert expected_response.status == "completed"
        assert response.stop_reason == "end_turn"
        assert response.stop_sequence is None
        assert response.usage.input_tokens == (
            expected_usage.input_tokens - cache_write_tokens - cached_tokens
        )
        assert response.usage.cache_creation_input_tokens == cache_write_tokens
        assert response.usage.cache_read_input_tokens == cached_tokens
        assert response.usage.output_tokens == expected_usage.output_tokens
        assert response.usage.output_tokens_details.thinking_tokens == reasoning_tokens


@pytest.mark.parametrize("downstream", RESPONSES_MESSAGES_PROTOCOLS)
@pytest.mark.parametrize(
    "defect", ["order", "extra_reasoning", "missing_reasoning", "extra_text_part", "encrypted"]
)
def test_matched_response_rejects_changed_output(downstream: Protocol, defect: str) -> None:
    """Ensure output comparisons preserve every reasoning item and text-part boundary."""
    expected = _completed_responses_body(
        _matched_cassette_path("reasoning_encrypted", "responses", "non_stream-00.yaml")
    )
    response_type = {
        "responses": OpenAIResponsesResponse,
        "messages": AnthropicMessagesResponse,
    }[downstream]
    response = response_type.model_validate(
        _read_cassette(
            _matched_cassette_path("reasoning_encrypted", downstream, "non_stream-00.yaml")
        )["response"]["body"]
    )
    _assert_matched_response(response, downstream, expected)
    response = response.model_copy(deep=True)
    items = response.output if downstream == "responses" else response.content
    if defect == "order":
        items.reverse()
    elif defect == "extra_reasoning":
        items.append(items[0].model_copy(deep=True))
    elif defect == "missing_reasoning":
        items.pop(0)
    elif defect == "extra_text_part":
        parts = items[-1].content if downstream == "responses" else items
        parts.append(parts[-1].model_copy(update={"text": ""}))
    elif downstream == "responses":
        items[0].encrypted_content = "changed-opaque-state"
    else:
        items[0].signature = "changed-opaque-state"

    with pytest.raises(AssertionError):
        _assert_matched_response(response, downstream, expected)


@pytest.mark.parametrize("downstream", PROTOCOLS)
@pytest.mark.parametrize("detail", ["cache_read", "cache_write", "reasoning"])
def test_matched_response_rejects_changed_usage_details(downstream: Protocol, detail: str) -> None:
    """Ensure non-streaming comparisons check semantic cache and reasoning usage."""
    expected = _completed_responses_body(
        _matched_cassette_path("simple_text", "responses", "non_stream-00.yaml")
    )
    response_type = {
        "responses": OpenAIResponsesResponse,
        "chat/completions": OpenAIChatCompletionsResponse,
        "messages": AnthropicMessagesResponse,
    }[downstream]
    response = response_type.model_validate(
        _read_cassette(_matched_cassette_path("simple_text", downstream, "non_stream-00.yaml"))[
            "response"
        ]["body"]
    )
    _assert_matched_response(response, downstream, expected)
    response = response.model_copy(deep=True)
    usage = response.usage
    if downstream == "messages":
        if detail == "reasoning":
            usage.output_tokens_details.thinking_tokens += 1
        elif detail == "cache_read":
            usage.cache_read_input_tokens += 1
        else:
            usage.cache_creation_input_tokens += 1
    else:
        input_details = (
            usage.input_tokens_details
            if downstream == "responses"
            else usage.prompt_tokens_details
        )
        output_details = (
            usage.output_tokens_details
            if downstream == "responses"
            else usage.completion_tokens_details
        )
        if detail == "reasoning":
            output_details.reasoning_tokens += 1
        elif detail == "cache_read":
            input_details.cached_tokens += 1
        else:
            input_details.cache_write_tokens += 1

    with pytest.raises(AssertionError):
        _assert_matched_response(response, downstream, expected)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_native_incomplete_responses_preserve_text_and_usage(stream: bool) -> None:
    """Ensure truncated native Responses retain SDK text, usage, and terminal metadata."""
    mode = "stream" if stream else "non_stream"
    cassette_path = (
        CASSETTE_DIR / "errors/incomplete_response" / f"openai-responses-{mode}-00.yaml"
    )
    body = deepcopy(_read_cassette(cassette_path)["request"]["body"])
    async with _get_http_client([cassette_path]) as http_client:
        sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        expected = await sdk.responses.create(**body)
        if stream:
            expected_events = [event async for event in expected]
            assert expected_events[-1].type == "response.incomplete"
            expected = expected_events[-1].response

    async with _get_http_client([cassette_path]) as http_client:
        client = Molto(http_client=http_client)
        actual = await client.responses(body=body, endpoint="responses", api_key="test-api-key")
        if stream:
            actual_events = [event async for event in actual]
            assert [event.model_dump() for event in actual_events] == [
                event.model_dump() for event in expected_events
            ]
            assert_responses_stream_invariants([dumps_sse(event) for event in actual_events])
            actual = actual_events[-1].response
            assert (
                "".join(
                    event.delta
                    for event in actual_events
                    if event.type == "response.output_text.delta"
                )
                == expected.output_text
            )

    assert actual.model_dump() == expected.model_dump()
    _assert_matched_response(actual, "responses", expected)
    assert actual.output_text == expected.output_text != ""
    assert actual.status == "incomplete"
    assert actual.incomplete_details.reason == "max_output_tokens"
    assert actual.usage.input_tokens == expected.usage.input_tokens > 0
    assert actual.usage.output_tokens == expected.usage.output_tokens == body["max_output_tokens"]
    assert actual.usage.total_tokens == actual.usage.input_tokens + actual.usage.output_tokens


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_incomplete_responses_translate_to_chat_with_text_and_usage(stream: bool) -> None:
    """Ensure translated Chat retains partial text, SDK usage, and length termination."""
    # Both downstream modes consume the same upstream stream before projection.
    cassette_path = CASSETTE_DIR / "errors/incomplete_response" / "openai-responses-stream-00.yaml"
    recorded_body = deepcopy(_read_cassette(cassette_path)["request"]["body"])
    async with _get_http_client([cassette_path]) as http_client:
        sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        expected_events = [event async for event in await sdk.responses.create(**recorded_body)]
    assert expected_events[-1].type == "response.incomplete"
    expected = expected_events[-1].response
    assert expected.output_text and expected.usage.output_tokens > 0

    messages = [
        {"role": "system", "content": recorded_body["instructions"]},
        *recorded_body["input"],
    ]
    body = {
        "model": recorded_body["model"],
        "messages": messages,
        "max_completion_tokens": recorded_body["max_output_tokens"],
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    # Chat expresses instructions as messages; Responses uses developer messages
    # with typed content on translated requests. Assert the complete wire payload.
    expected_body = {
        "model": recorded_body["model"],
        "input": [
            {
                "role": "developer" if message["role"] == "system" else message["role"],
                "content": [{"type": "input_text", "text": message["content"]}],
            }
            for message in messages
        ],
        "max_output_tokens": recorded_body["max_output_tokens"],
        "stream": True,
    }
    async with _get_http_client([cassette_path], expected_body=expected_body) as http_client:
        client = Molto(http_client=http_client)
        actual = await client.chat_completions(
            body=body, endpoint="responses", api_key="test-api-key"
        )
        if stream:
            chunks = [chunk async for chunk in actual]
            text = "".join(
                choice.delta.content or "" for chunk in chunks for choice in chunk.choices
            )
            finish_reasons = [
                choice.finish_reason
                for chunk in chunks
                for choice in chunk.choices
                if choice.finish_reason is not None
            ]
            usages = [chunk.usage for chunk in chunks if chunk.usage is not None]
            assert len(usages) == 1
            usage = usages[0]
        else:
            _assert_matched_response(actual, "chat/completions", expected)
            assert len(actual.choices) == 1
            text = actual.choices[0].message.content
            finish_reasons = [actual.choices[0].finish_reason]
            usage = actual.usage

    assert text == expected.output_text
    assert finish_reasons == ["length"]
    assert usage.prompt_tokens == expected.usage.input_tokens
    assert usage.completion_tokens == expected.usage.output_tokens == body["max_completion_tokens"]
    assert usage.total_tokens == expected.usage.total_tokens
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert (
        usage.prompt_tokens_details.cached_tokens
        == expected.usage.input_tokens_details.cached_tokens
    )
    assert (
        usage.completion_tokens_details.reasoning_tokens
        == expected.usage.output_tokens_details.reasoning_tokens
    )


def _response_history(response: BaseModel, protocol: Protocol) -> list[dict[str, Any]]:
    """Convert SDK output to the explicit input shapes used by these conversations."""
    if protocol == "responses":
        assert isinstance(response, OpenAIResponsesResponse)
        history = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
        for item in history:
            assert item["type"] in {"message", "reasoning", "function_call"}
            if item["type"] == "reasoning":
                # Reasoning status is output-only, unlike message/call status.
                item.pop("status", None)
        return history
    if protocol == "chat/completions":
        assert isinstance(response, OpenAIChatCompletionsResponse)
        assert len(response.choices) == 1 and response.choices[0].index == 0
        message = response.choices[0].message.model_dump(mode="json", exclude_none=True)
        # The SDK's streaming parser adds these local parsing properties.
        message.pop("parsed", None)
        if message.get("annotations") == []:
            message.pop("annotations")
        for call in message.get("tool_calls", []):
            call.pop("index", None)
            call["function"].pop("parsed_arguments", None)
        if message.get("content") == "" and message.get("tool_calls"):
            message.pop("content")
        return [message]
    assert isinstance(response, AnthropicMessagesResponse)
    content = []
    for block in response.content:
        item = block.model_dump(exclude_none=True)
        assert item["type"] in {"text", "thinking", "tool_use"}
        if item["type"] == "tool_use":
            item.pop("__json_buf", None)
            # The SDK supplies the default direct caller even when absent on
            # the wire; omitting that default preserves the same tool use.
            if item.get("caller") == {"type": "direct"}:
                item.pop("caller")
        elif item["type"] == "thinking" and item.get("signature") == "":
            # Unsigned raw reasoning has no opaque state to send back.
            item.pop("signature")
        content.append(item)
    return [{"role": response.role, "content": content}]


def _final_stream_result(events: list[BaseModel], protocol: Protocol) -> BaseModel:
    """Accumulate returned SDK events with SDK helpers, independently of Molto's collectors."""
    if protocol == "responses":
        assert events[-1].type in {"response.completed", "response.incomplete"}
        return events[-1].response
    if protocol == "chat/completions":
        state = ChatCompletionStreamState()
        for event in events:
            state.handle_chunk(event.model_copy(deep=True))
        return state.get_final_completion()
    snapshot = None
    for event in events:
        snapshot = accumulate_event(event=event.model_copy(deep=True), current_snapshot=snapshot)
    assert events[-1].type == "message_stop" and snapshot is not None
    return snapshot


def _stream_history(events: list[BaseModel], protocol: Protocol) -> list[dict[str, Any]]:
    """Collect returned SDK events independently of Molto's adapters and collectors."""
    return _response_history(_final_stream_result(events, protocol), protocol)


def _comparable_history(
    history: list[dict[str, Any]], protocol: Protocol, *, translated: bool
) -> list[dict[str, Any]]:
    """Allow only named native input equivalences; keep parts, calls and state ordered."""
    history = deepcopy(history)
    if protocol != "responses":
        return history
    for item in history:
        item_type = item.get("type", "message")
        if item_type not in {"message", "reasoning", "function_call"}:
            continue
        # Responses output-item IDs are optional history metadata. The other
        # protocols cannot reproduce IDs or phase; portable call_id stays exact.
        item.pop("id", None)
        if item_type == "reasoning":
            assert "status" not in item, "Reasoning history must omit output-only status"
            item.setdefault("content", [])
        elif item_type == "message":
            item.setdefault("type", "message")
            # Native fixtures also omit the optional final-answer phase.
            # Keep commentary exact on native routes, where it is recorded.
            if translated or item.get("phase") == "final_answer":
                item.pop("phase", None)
            if item["role"] == "assistant":
                item.setdefault("status", "completed")
                if isinstance(item["content"], str):
                    item["content"] = [{"type": "output_text", "text": item["content"]}]
                for part in item["content"]:
                    if part["type"] == "output_text":
                        part.setdefault("annotations", [])
                        part.setdefault("logprobs", [])
    return history


def _continued_request(
    saved_body: dict[str, Any],
    protocol: Protocol,
    history: list[dict[str, Any]] | None,
    *,
    translated: bool,
) -> dict[str, Any]:
    """Append only saved new inputs to the cumulative history returned by Molto."""
    body = _without_reasoning_item_status(saved_body)
    if history is None:
        return body
    key = "input" if protocol == "responses" else "messages"
    expected_history = body[key][: len(history)]
    assert _comparable_history(history, protocol, translated=translated) == _comparable_history(
        expected_history, protocol, translated=translated
    ), "Returned history differs from the next saved request"
    new_inputs = body[key][len(history) :]
    assert new_inputs
    # Never restore an earlier assistant turn from a fixture, even when it has
    # the same length as the actual history.
    assert all(
        item.get("role") != "assistant" and item.get("type") not in {"reasoning", "function_call"}
        for item in new_inputs
    )
    body[key] = deepcopy(history) + new_inputs
    return body


async def _assert_matched_non_stream(
    scenario: str,
    downstream: Protocol,
    upstream: Protocol,
    *,
    case: RouteCase | None = None,
) -> None:
    translated = downstream != upstream
    mode: Literal["stream", "non_stream"] = "stream" if translated else "non_stream"
    case = case or _default_case(scenario, downstream, upstream, mode)
    async with _get_http_client(
        case.upstream.paths,
        upstream,
        translated=translated,
        continued=True,
    ) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        history = None
        for downstream_path, reference_path in zip(
            case.downstream.paths, case.reference.paths, strict=True
        ):
            request_body = _continued_request(
                _read_cassette(downstream_path)["request"]["body"],
                downstream,
                history,
                translated=translated,
            )
            if translated:
                request_body["stream"] = False
            response = await coordinator(
                body=request_body,
                endpoint=upstream,
                provider=case.upstream.provider,
                api_key="test-api-key",
            )
            _assert_matched_response(
                response,
                downstream,
                _completed_responses_body(reference_path),
            )
            key = "input" if downstream == "responses" else "messages"
            history = request_body[key] + _response_history(response, downstream)


async def test_responses_native_stream_normalizes_omitted_cache_write_tokens():
    """Ensures the matched omission parses in the SDK and receives only the documented adjustment."""
    cassette_path = _matched_cassette_path("reasoning_raw", "responses", "stream-00.yaml")
    cassette = _read_cassette(cassette_path)
    request_body = deepcopy(cassette["request"]["body"])

    # This matched fixture exercises compatibility, but has no faithful provider
    # original establishing the omission's provenance.
    async with _get_http_client([cassette_path], "responses") as http_client:
        sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        async with await sdk.responses.create(**request_body) as sdk_stream:
            sdk_events = [event async for event in sdk_stream]
    sdk_completed = next(event for event in sdk_events if event.type == "response.completed")
    assert sdk_completed.response.usage.input_tokens_details.cache_write_tokens is None
    expected = [event.model_dump(mode="json", exclude_unset=True) for event in sdk_events]
    expected[-1]["response"]["usage"]["input_tokens_details"]["cache_write_tokens"] = 0

    async with _get_http_client([cassette_path], "responses") as http_client:
        client = Molto(http_client=http_client)
        stream = await client.responses(
            body=request_body,
            endpoint="responses",
            provider="vllm",
            api_key="test-api-key",
        )
        events = [event async for event in stream]

    assert [event.model_dump(mode="json", exclude_unset=True) for event in events] == expected
    completed = next(event for event in events if event.type == "response.completed")
    assert completed.response.usage is not None
    assert completed.response.usage.input_tokens_details.cache_write_tokens == 0


async def _assert_stream_events(
    cassette_path_pairs: list[tuple[Path, Path]],
    downstream: Protocol,
    upstream: Protocol,
    *,
    requests: list[Request] | None = None,
) -> None:
    first_downstream_path, first_upstream_path = cassette_path_pairs[0]
    scenario = first_downstream_path.parent.name
    continued = (
        first_downstream_path.is_relative_to(MATCHED_CASSETTE_DIR)
        and scenario not in PROMPT_CACHE_SCENARIOS
    )
    provider = (
        _provider_from_cassette_path(first_upstream_path)
        if first_upstream_path.is_relative_to(MATCHED_CASSETTE_DIR)
        else DEFAULT_PROVIDER_BY_ENDPOINT[upstream]
    )
    async with _get_http_client(
        [upstream_path for _, upstream_path in cassette_path_pairs],
        upstream,
        translated=downstream != upstream,
        continued=continued,
        requests=requests,
    ) as http_client:
        client = Molto(http_client=http_client)
        history = None
        for downstream_path, upstream_path in cassette_path_pairs:
            cassette = _read_cassette(downstream_path)
            request_body = (
                _continued_request(
                    cassette["request"]["body"],
                    downstream,
                    history,
                    translated=downstream != upstream,
                )
                if continued
                else deepcopy(cassette["request"]["body"])
            )
            coordinator = {
                "responses": client.responses,
                "chat/completions": client.chat_completions,
                "messages": client.messages,
            }[downstream]
            response = await coordinator(
                body=request_body,
                endpoint=upstream,
                provider=provider,
                api_key="test-api-key",
            )
            events = [event async for event in response]
            expected_sse_events = _sse_records(cassette, downstream)
            actual_sse_events = []
            for event in events:
                assert isinstance(event, BaseModel)
                actual_sse_events.append(dumps_sse(event))
            if downstream == "responses":
                assert len(events) == len(expected_sse_events)
                assert_responses_stream_invariants(
                    actual_sse_events, native=upstream == "responses"
                )
                if scenario in PROMPT_CACHE_SCENARIOS:
                    usage = events[-1].response.usage
                    assert usage.input_tokens_details.cached_tokens > 0
                    assert usage.input_tokens_details.cache_write_tokens == 0
                if upstream == "responses":
                    _assert_sse_events_contain(actual_sse_events, expected_sse_events)
                elif upstream == "chat/completions":
                    # Chat does not expose Responses output-item IDs or the exact terminal
                    # timestamp or phase. Every other translated field must still match exactly.
                    actual_events = [
                        reconstructable_responses_event(
                            sse, body=deepcopy(cassette["request"]["body"])
                        )
                        for sse in actual_sse_events
                    ]
                    expected_events = [
                        reconstructable_responses_event(
                            sse, body=deepcopy(cassette["request"]["body"])
                        )
                        for sse in expected_sse_events
                    ]
                    if scenario in TOOL_CALL_SCENARIOS:
                        for event in actual_events + expected_events:
                            _reconstructable_tool_responses_event(event)
                    assert actual_events == expected_events
                elif upstream == "messages":
                    actual_events = [
                        reconstructable_responses_event(
                            sse, body=deepcopy(cassette["request"]["body"])
                        )
                        for sse in actual_sse_events
                    ]
                    expected_events = [
                        reconstructable_responses_event(
                            sse, body=deepcopy(cassette["request"]["body"])
                        )
                        for sse in expected_sse_events
                    ]
                    for event in actual_events + expected_events:
                        response_data = event.get("response")
                        if isinstance(response_data, dict):
                            response_data.pop("created_at", None)
                        if scenario in ENCRYPTED_REASONING_SCENARIOS:
                            _reconstructable_encrypted_responses_event(event)
                        if scenario in TOOL_CALL_SCENARIOS:
                            _reconstructable_tool_responses_event(event)
                    assert actual_events == expected_events
            elif downstream == "chat/completions":
                if upstream == "chat/completions":
                    assert len(events) == len(expected_sse_events)
                    _assert_sse_events_contain(actual_sse_events, expected_sse_events)
                elif upstream in {"responses", "messages"}:
                    actual_events = [
                        chunk
                        for sse in actual_sse_events
                        for chunk in reconstructable_chat_completions_chunks(
                            sse, body=deepcopy(cassette["request"]["body"])
                        )
                    ]
                    expected_events = [
                        chunk
                        for sse in expected_sse_events
                        for chunk in reconstructable_chat_completions_chunks(
                            sse, body=deepcopy(cassette["request"]["body"])
                        )
                    ]
                    if upstream == "messages":
                        for chunk in actual_events + expected_events:
                            chunk.pop("created", None)
                    assert actual_events == expected_events
            elif downstream == "messages":
                if upstream == "messages":
                    assert len(events) == len(expected_sse_events)
                    _assert_sse_events_contain(actual_sse_events, expected_sse_events)
                else:
                    actual_events = [
                        reconstructable_anthropic_event(sse) for sse in actual_sse_events
                    ]
                    expected_events = [
                        reconstructable_anthropic_event(sse) for sse in expected_sse_events
                    ]
                    if upstream == "responses":
                        _expect_responses_search_usage(expected_events, upstream_path)
                    assert actual_events == expected_events
            else:
                raise ValueError(f"Unsupported downstream protocol: {downstream}")
            if continued:
                key = "input" if downstream == "responses" else "messages"
                history = request_body[key] + _stream_history(events, downstream)


async def _assert_matched_stream_events(
    scenario: str,
    downstream: Protocol,
    upstream: Protocol,
    *,
    requests: list[Request] | None = None,
    case: RouteCase | None = None,
) -> None:
    case = case or _default_case(scenario, downstream, upstream, "stream")
    cassette_path_pairs = list(zip(case.downstream.paths, case.upstream.paths, strict=True))
    await _assert_stream_events(cassette_path_pairs, downstream, upstream, requests=requests)


@pytest.mark.parametrize(
    ("downstream", "defect", "stream"),
    [(protocol, "call-id", True) for protocol in PROTOCOLS]
    + [
        (protocol, "opaque-snapshot", stream)
        for protocol in RESPONSES_MESSAGES_PROTOCOLS
        for stream in (False, True)
    ],
)
async def test_continuation_rejects_changed_returned_history(
    monkeypatch: pytest.MonkeyPatch, downstream: Protocol, defect: str, stream: bool
) -> None:
    """Ensure public replay continuation rejects corrupted returned calls and opaque state."""
    collect_history = _response_history

    def changed_history(response: BaseModel, protocol: Protocol) -> list[dict[str, Any]]:
        history = collect_history(response, protocol)
        if defect == "call-id":
            if protocol == "responses":
                next(item for item in history if item["type"] == "function_call")["call_id"] = (
                    "changed-call"
                )
            elif protocol == "chat/completions":
                history[0]["tool_calls"][0]["id"] = "changed-call"
            else:
                next(block for block in history[0]["content"] if block["type"] == "tool_use")[
                    "id"
                ] = "changed-call"
        elif protocol == "responses":
            next(item for item in history if item["type"] == "reasoning")["encrypted_content"] = (
                "changed-snapshot"
            )
        else:
            next(block for block in history[0]["content"] if block["type"] == "thinking")[
                "signature"
            ] = "changed-snapshot"
        return history

    monkeypatch.setattr(f"{__name__}._response_history", changed_history)
    scenario = "tool_call" if defect == "call-id" else "reasoning_encrypted"
    upstream = "messages" if downstream == "responses" else "responses"
    runner = _assert_matched_stream_events if stream else _assert_matched_non_stream
    with pytest.raises(AssertionError, match="Returned history differs"):
        await runner(scenario, downstream, upstream)


async def test_continuation_keeps_actual_earlier_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the third public call retains the first returned output, including generated IDs."""
    collect_history = _response_history
    continue_request = _continued_request
    returned_history = []
    sent_bodies = []

    def remember_history(response: BaseModel, protocol: Protocol) -> list[dict[str, Any]]:
        history = collect_history(response, protocol)
        returned_history.append(deepcopy(history))
        return history

    def remember_request(*args, **kwargs) -> dict[str, Any]:
        body = continue_request(*args, **kwargs)
        sent_bodies.append(deepcopy(body))
        return body

    monkeypatch.setattr(f"{__name__}._response_history", remember_history)
    monkeypatch.setattr(f"{__name__}._continued_request", remember_request)
    await _assert_matched_stream_events("tool_call", "responses", "chat/completions")
    assert len(sent_bodies) == len(returned_history) == 3
    first_history = sent_bodies[0]["input"] + returned_history[0]
    assert sent_bodies[1]["input"][: len(first_history)] == first_history
    second_history = sent_bodies[1]["input"] + returned_history[1]
    assert sent_bodies[2]["input"][: len(second_history)] == second_history
    saved_body = _read_cassette(
        _matched_cassette_path("tool_call", "responses", "stream-01.yaml")
    )["request"]["body"]
    assert returned_history[0][0]["id"] != saved_body["input"][1]["id"]


@pytest.mark.parametrize("case", _non_stream_cases("simple_text"))
async def test_simple_text_non_stream(case: RouteCase):
    """Ensure complete text output becomes actual history for the next public request."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_non_stream("simple_text", downstream, upstream, case=case)


@pytest.mark.parametrize("case", _stream_cases("simple_text"))
async def test_simple_text_stream(case: RouteCase):
    """Ensure streamed text output becomes actual history across every protocol pair."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events("simple_text", downstream, upstream, case=case)


@pytest.mark.parametrize(
    "case", [case for scenario in PROMPT_CACHE_SCENARIOS for case in _stream_cases(scenario)]
)
async def test_prompt_cache_stream(case: RouteCase):
    """Ensures discovered cache conversations retain their scenario-specific request and usage assertions."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    scenario = case.upstream.scenario
    await _assert_matched_stream_events(scenario, downstream, upstream, case=case)


@pytest.mark.parametrize(
    ("downstream", "upstream", "representation"),
    [(downstream, upstream, "raw") for downstream, upstream in ROUTES]
    + [
        (downstream, upstream, representation)
        for downstream, upstream in RESPONSES_MESSAGES_ROUTES
        for representation in ("summary", "signature")
    ],
)
@pytest.mark.parametrize("followed_by_user", [False, True])
async def test_reasoning_only_history(
    downstream: Protocol,
    upstream: Protocol,
    representation: str,
    followed_by_user: bool,
) -> None:
    """Ensures representable reasoning-only history reaches every upstream protocol."""
    reasoning = "Prior thought" if representation != "signature" else ""
    signed = representation != "raw"
    responses_item = {
        "type": "reasoning",
        "status": "completed",
        "content": [] if signed else [{"type": "reasoning_text", "text": reasoning}],
        "summary": [{"type": "summary_text", "text": reasoning}] if reasoning and signed else [],
        **({"encrypted_content": "opaque-state"} if signed else {}),
    }
    thinking_block = {
        "type": "thinking",
        "thinking": reasoning,
        **({"signature": "opaque-state"} if signed else {}),
    }
    additions: dict[Protocol, dict[str, Any]] = {
        "responses": {"input": [responses_item], "reasoning": {"effort": "none"}},
        "chat/completions": {
            "messages": [{"role": "assistant", "reasoning_content": reasoning}],
            "reasoning_effort": "none",
        },
        "messages": {
            "messages": [{"role": "assistant", "content": [thinking_block]}],
            "thinking": {"type": "disabled"},
        },
    }
    if followed_by_user:
        for protocol, fields in additions.items():
            key = "input" if protocol == "responses" else "messages"
            part_type = "input_text" if protocol == "responses" else "text"
            fields[key].append(
                {"role": "user", "content": [{"type": part_type, "text": "Continue"}]}
            )
    await _assert_request_parity(
        downstream,
        upstream,
        additions[downstream],
        additions[upstream],
        provider="openai" if signed else "vllm",
    )


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("marked", [False, True])
async def test_assistant_history_message_boundaries(
    downstream: Protocol, upstream: Protocol, marked: bool
) -> None:
    """Ensures multipart assistant history retains parts, markers, and adjacent messages."""

    def history(protocol: Protocol) -> dict[str, Any]:
        messages = []
        for texts in (("First part.", "Second part."), ("Separate message.",)):
            content = []
            for index, text in enumerate(texts):
                part: dict[str, Any] = {"type": "text", "text": text}
                if protocol == "responses":
                    part.update(type="output_text", annotations=[], logprobs=[])
                if marked and index == 0:
                    if protocol == "messages":
                        part["cache_control"] = {"type": "ephemeral"}
                    else:
                        part["prompt_cache_breakpoint"] = {"mode": "explicit"}
                content.append(part)
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if protocol == "responses":
                message.update(type="message", status="completed")
            elif protocol == "chat/completions" and len(content) == 1 and not marked:
                message["content"] = texts[0]
            messages.append(message)
        additions = {"input" if protocol == "responses" else "messages": messages}
        if marked and protocol != "messages":
            additions["prompt_cache_options"] = {"mode": "explicit"}
        return additions

    additions = history(downstream)
    expected = history(upstream)
    if marked and downstream != upstream and downstream != "messages":
        if upstream == "messages":
            for message in expected["messages"]:
                message["content"][0]["cache_control"]["ttl"] = "1h"
        else:
            expected["prompt_cache_options"]["ttl"] = "30m"
    await _assert_request_parity(downstream, upstream, additions, expected)


@pytest.mark.parametrize("downstream", ["chat/completions", "messages"])
@pytest.mark.parametrize("upstream", PROTOCOLS)
async def test_separate_reasoning_and_text_history_messages(
    downstream: Protocol, upstream: Protocol
) -> None:
    """Ensures an explicit reasoning-only assistant message never absorbs the next message."""
    history: dict[Protocol, dict[str, Any]] = {
        "chat/completions": {
            "messages": [
                {"role": "assistant", "reasoning_content": "Prior thought"},
                {"role": "assistant", "content": "Prior answer"},
            ],
            "reasoning_effort": "none",
        },
        "messages": {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "Prior thought"}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "Prior answer"}]},
            ],
            "thinking": {"type": "disabled"},
        },
        "responses": {
            "input": [
                {
                    "type": "reasoning",
                    "status": "completed",
                    "content": [{"type": "reasoning_text", "text": "Prior thought"}],
                    "summary": [],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Prior answer",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
            ],
            "reasoning": {"effort": "none"},
        },
    }
    await _assert_request_parity(
        downstream, upstream, history[downstream], history[upstream], provider="vllm"
    )


@pytest.mark.parametrize("with_text_and_tools", [False, True])
@pytest.mark.parametrize(
    ("reasoning_parts", "expected_reasoning"),
    [(["First", "second"], "First\nsecond"), (["First", "", "second"], "First\n\nsecond")],
)
async def test_chat_groups_consecutive_assistant_parts(
    with_text_and_tools: bool, reasoning_parts: list[str], expected_reasoning: str
) -> None:
    """Ensures reasoning joins with newlines while parts stay within their source message."""
    parts: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": text} for text in reasoning_parts
    ]
    expected_message: dict[str, Any] = {
        "role": "assistant",
        "reasoning_content": expected_reasoning,
    }
    if with_text_and_tools:
        text_parts = [{"type": "text", "text": text} for text in ("Answer ", "continued")]
        calls = [
            {"type": "tool_use", "id": call_id, "name": "lookup", "input": {}}
            for call_id in ("c1", "c2")
        ]
        parts.extend([*text_parts, *calls])
        expected_message["content"] = text_parts
        expected_message["tool_calls"] = [
            {
                "type": "function",
                "id": call["id"],
                "function": {"name": "lookup", "arguments": "{}"},
            }
            for call in calls
        ]
    await _assert_request_parity(
        "messages",
        "chat/completions",
        {
            "messages": [
                {"role": "assistant", "content": parts},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "Separate"}]},
            ],
            "thinking": {"type": "disabled"},
        },
        {
            "messages": [
                expected_message,
                {"role": "assistant", "reasoning_content": "Separate"},
            ],
            "reasoning_effort": "none",
        },
        provider="vllm",
    )


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (
            [{"type": "text", "text": "Answer"}, {"type": "thinking", "thinking": "Later"}],
            "cannot preserve this assistant reasoning order",
        ),
        (
            [
                {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
                {"type": "thinking", "thinking": "After tool"},
            ],
            "cannot preserve this assistant reasoning order",
        ),
        (
            [
                {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
                {"type": "text", "text": "After tool"},
            ],
            "cannot preserve assistant text after tool calls",
        ),
    ],
)
async def test_chat_rejects_unrepresentable_assistant_part_order(content, match) -> None:
    """Ensures Chat rejects assistant order it cannot preserve within one message."""
    body = deepcopy(_read_cassette(_simple_text_cassette("messages"))["request"]["body"])
    body["messages"] = [{"role": "assistant", "content": content}]
    async with _get_http_client([]) as http_client:
        with pytest.raises(UnsupportedInput, match=match):
            await Molto(http_client=http_client).messages(
                body=body,
                endpoint="chat/completions",
                provider="vllm",
                api_key="test-api-key",
            )


@pytest.mark.parametrize("upstream", ["responses", "messages"])
async def test_interleaved_assistant_history(upstream: Protocol) -> None:
    """Ensures mixed assistant parts retain text runs and order around reasoning and calls."""
    first_texts = [{"type": "text", "text": text} for text in ("First", "Second")]
    last_texts = [{"type": "text", "text": text} for text in ("Third", "Fourth")]
    additions = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    *first_texts,
                    {"type": "thinking", "thinking": "A thought"},
                    {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
                    *last_texts,
                ],
            }
        ],
        "thinking": {"type": "disabled"},
    }
    if upstream == "messages":
        expected = additions
    else:
        text_messages = [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": part["text"],
                        "annotations": [],
                        "logprobs": [],
                    }
                    for part in parts
                ],
            }
            for parts in (first_texts, last_texts)
        ]
        expected = {
            "input": [
                text_messages[0],
                {
                    "type": "reasoning",
                    "status": "completed",
                    "content": [{"type": "reasoning_text", "text": "A thought"}],
                    "summary": [],
                },
                {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{}",
                    "call_id": "c1",
                    "status": "completed",
                },
                text_messages[1],
            ],
            "reasoning": {"effort": "none"},
        }
    await _assert_request_parity("messages", upstream, additions, expected, provider="vllm")


@pytest.mark.parametrize("case", _stream_cases("reasoning_raw"))
async def test_reasoning_raw_stream(case: RouteCase):
    """Ensure returned raw reasoning and text survive actual conversation continuation."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events("reasoning_raw", downstream, upstream, case=case)


@pytest.mark.parametrize("case", _stream_cases("reasoning_encrypted"))
async def test_reasoning_encrypted_stream(case: RouteCase):
    """Ensure streamed summaries and exact opaque snapshots survive the next request."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events("reasoning_encrypted", downstream, upstream, case=case)


@pytest.mark.parametrize("case", _non_stream_cases("reasoning_encrypted"))
async def test_reasoning_encrypted_non_stream(case: RouteCase):
    """Ensure complete encrypted reasoning output supplies the next request's history."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_non_stream("reasoning_encrypted", downstream, upstream, case=case)


@pytest.mark.parametrize("case", _stream_cases("input_image_base64"))
async def test_input_image_base64_stream(case: RouteCase):
    """Ensure base64 image conversations retain actual assistant history and image input."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events("input_image_base64", downstream, upstream, case=case)


@pytest.mark.parametrize("case", _stream_cases("input_image_remote_url"))
async def test_input_image_remote_url_stream(case: RouteCase):
    """Ensure every route retains the remote URL after text and in continued history."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    requests: list[Request] = []
    await _assert_matched_stream_events(
        "input_image_remote_url", downstream, upstream, requests=requests, case=case
    )
    url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
    expected_content = {
        "responses": [
            {"type": "input_text", "text": "What logo is this?"},
            {"type": "input_image", "image_url": url},
        ],
        "chat/completions": [
            {"type": "text", "text": "What logo is this?"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
        "messages": [
            {"type": "text", "text": "What logo is this?"},
            {"type": "image", "source": {"type": "url", "url": url}},
        ],
    }[upstream]
    assert len(requests) == 2
    for request in requests:
        body = json_loads(request.content)
        history = body["input" if upstream == "responses" else "messages"]
        assert history[0] == {"role": "user", "content": expected_content}


@pytest.mark.parametrize("case", _stream_cases("input_image_with_reasoning_encrypted"))
async def test_input_image_with_reasoning_encrypted_stream(case: RouteCase):
    """Ensure image input and actual encrypted reasoning history survive together."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events(
        "input_image_with_reasoning_encrypted", downstream, upstream, case=case
    )


@pytest.mark.parametrize("case", _stream_cases("tool_call"))
async def test_tool_call_stream(case: RouteCase):
    """Ensure three public turns reuse returned tool calls and cumulatively linked results."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events("tool_call", downstream, upstream, case=case)


@pytest.mark.parametrize("case", _stream_cases("tool_call_choice"))
async def test_tool_call_choice_stream(case: RouteCase):
    """Ensure tool selection changes retain the actual preceding tool call in history."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events("tool_call_choice", downstream, upstream, case=case)


@pytest.mark.parametrize("case", _stream_cases("tool_call_image_result_parallel"))
async def test_tool_call_image_result_parallel_stream(case: RouteCase):
    """Ensure linked parallel results precede relocated images in actual conversations."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    requests: list[Request] = []
    await _assert_matched_stream_events(
        "tool_call_image_result_parallel", downstream, upstream, requests=requests, case=case
    )
    if upstream == "chat/completions":
        assert len(requests) == 2
        messages = json_loads(requests[1].content)["messages"]
        assert [message["role"] for message in messages] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "user",
            "user",
            "user",
        ]
        call_ids = [call["id"] for call in messages[1]["tool_calls"]]
        assert len(call_ids) == len(set(call_ids)) == 2
        assert [message["tool_call_id"] for message in messages[2:4]] == call_ids
        expected_input = _read_cassette(case.reference.paths[1])["request"]["body"]["input"]
        image_urls = [
            part["image_url"]
            for item in expected_input
            if item.get("type") == "function_call_output"
            for part in item["output"]
            if part["type"] == "input_image"
        ]
        assert messages[4:6] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Image returned by tool call {call_id!r}:"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
            for call_id, url in zip(call_ids, image_urls, strict=True)
        ]


@pytest.mark.parametrize("case", _stream_cases("tool_call_parallel"))
async def test_tool_call_parallel_stream(case: RouteCase):
    """Ensure both returned parallel calls survive into the request containing their results."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events("tool_call_parallel", downstream, upstream, case=case)


@pytest.mark.parametrize("case", _stream_cases("tool_call_with_reasoning_encrypted"))
async def test_tool_call_with_reasoning_encrypted_stream(case: RouteCase):
    """Ensure three-turn tool history retains actual call IDs and opaque reasoning snapshots."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    await _assert_matched_stream_events(
        "tool_call_with_reasoning_encrypted", downstream, upstream, case=case
    )


@pytest.mark.parametrize(
    ("case", "turn"),
    [
        pytest.param(case, turn, id=f"{case.id}-{turn:02d}")
        for case in route_cases(
            MATCHED_CONVERSATIONS, "error-invalid_request_error", "stream", ROUTES
        )
        for turn in range(len(case.upstream.cassettes))
    ],
)
async def test_error_matched(case: RouteCase, turn: int):
    """Ensures discovered unknown error fixtures preserve the original upstream SDK exception."""
    downstream, upstream = case.downstream.endpoint, case.upstream.endpoint
    downstream_path = case.downstream.paths[turn]
    upstream_path = case.upstream.paths[turn]
    downstream_cassette = _read_cassette(downstream_path)
    upstream_cassette = _read_cassette(upstream_path)
    expected_response = upstream_cassette["response"]
    error_type = AnthropicBadRequestError if upstream == "messages" else OpenAIBadRequestError

    async with _get_http_client(
        [upstream_path],
        upstream,
        translated=downstream != upstream,
    ) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        with pytest.raises(error_type) as exc_info:
            await coordinator(
                body=deepcopy(downstream_cassette["request"]["body"]),
                endpoint=upstream,
                provider=_provider_from_cassette_path(upstream_path),
                api_key="test-api-key",
            )

    assert exc_info.value.status_code == expected_response["status_code"]
    assert expected_response["body"]["error"]["message"] in exc_info.value.message
    assert exc_info.value.response.json() == expected_response["body"]


@pytest.mark.parametrize("upstream", ["chat/completions", "messages"])
async def test_translated_top_level_instructions(upstream: Protocol) -> None:
    """Ensures nonempty top-level instructions precede existing translated instructions."""
    additions = {
        "instructions": "Follow the leading instruction.",
        "input": [
            {"role": "system", "content": "Keep the second instruction."},
            {"role": "user", "content": "Hello"},
        ],
    }
    instructions = ["Follow the leading instruction.", "Keep the second instruction."]
    user = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    expected = (
        {
            "messages": [
                *[
                    {"role": "developer", "content": [{"type": "text", "text": text}]}
                    for text in instructions
                ],
                user,
            ]
        }
        if upstream == "chat/completions"
        else {"system": "\n\n".join(instructions), "messages": [user]}
    )
    await _assert_request_parity("responses", upstream, additions, expected)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("parallel", [False, True])
async def test_required_tool_choice_request(
    downstream: Protocol, upstream: Protocol, parallel: bool
) -> None:
    """Ensures required/any tool selection preserves explicit parallel-call control."""
    function = {
        "name": "lookup",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    }
    additions = {
        "responses": {
            "tools": [{"type": "function", **function}],
            "tool_choice": "required",
            "parallel_tool_calls": parallel,
        },
        "chat/completions": {
            "tools": [{"type": "function", "function": function}],
            "tool_choice": "required",
            "parallel_tool_calls": parallel,
        },
        "messages": {
            "tools": [{"name": "lookup", "input_schema": function["parameters"]}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": not parallel},
        },
    }
    await _assert_request_parity(downstream, upstream, additions[downstream], additions[upstream])


@pytest.mark.parametrize("downstream", RESPONSES_MESSAGES_PROTOCOLS)
@pytest.mark.parametrize(
    "image_names",
    [(("a",), ()), ((), ("b",)), (("b", "a"), ("d", "c"))],
    ids=["first-result-image", "last-result-image", "multiple-images-in-both-results"],
)
async def test_chat_tool_result_image_order(
    downstream: Protocol, image_names: tuple[tuple[str, ...], tuple[str, ...]]
) -> None:
    """Ensures linked parallel results precede all relocated images in source order."""
    calls = [
        {"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"}
        for call_id in ("call-a", "call-b")
    ]
    results = []
    chat_results = []
    deferred_images = []
    for call, names in zip(calls, image_names, strict=True):
        call_id = call["call_id"]
        urls = [f"https://example.com/{name}.png" for name in names]
        text = f"Result for {call_id}"
        results.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": [
                    {"type": "input_text", "text": text},
                    *[{"type": "input_image", "image_url": url} for url in urls],
                ],
            }
        )
        chat_content = [{"type": "text", "text": text}]
        if urls:
            chat_content.append(
                {
                    "type": "text",
                    "text": (
                        f"The image result of tool call {call_id!r} will be "
                        "attached as user prompt after all tool responses."
                    ),
                }
            )
            deferred_images.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image returned by tool call {call_id!r}:"},
                        *[{"type": "image_url", "image_url": {"url": url}} for url in urls],
                    ],
                }
            )
        chat_results.append({"role": "tool", "tool_call_id": call_id, "content": chat_content})
    if downstream == "responses":
        additions = {"input": [*calls, *results]}
    else:
        additions = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": call["call_id"], "name": "lookup", "input": {}}
                        for call in calls
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result["call_id"],
                            "content": [
                                {"type": "text", "text": part["text"]}
                                if part["type"] == "input_text"
                                else {
                                    "type": "image",
                                    "source": {"type": "url", "url": part["image_url"]},
                                }
                                for part in result["output"]
                            ],
                        }
                        for result in results
                    ],
                },
            ]
        }
    expected = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": call["call_id"],
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                    for call in calls
                ],
            },
            *chat_results,
            *deferred_images,
        ]
    }
    await _assert_request_parity(downstream, "chat/completions", additions, expected)


# --- Prompt caching --- #

_OPENAI_PROMPT_CACHE: dict[str, Any] = {
    "prompt_cache_key": "codex-turn-1",
    "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
    "prompt_cache_retention": "24h",
}
_MESSAGES_PROMPT_CACHE: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}


def _coordinator(client: Molto, protocol: Protocol):
    return {
        "responses": client.responses,
        "chat/completions": client.chat_completions,
        "messages": client.messages,
    }[protocol]


def _simple_text_cassette(protocol: Protocol) -> Path:
    return _matched_cassette_path("simple_text", protocol, "stream-00.yaml")


async def _assert_request_parity(
    downstream: Protocol,
    upstream: Protocol,
    additions: dict[str, Any],
    expected_additions: dict[str, Any],
    *,
    provider: Provider | None = None,
    stream: bool = True,
) -> None:
    """Assert the outgoing request and parity with independently parsed SDK output."""
    mode = "stream" if stream or downstream != upstream else "non_stream"
    downstream_path = _matched_cassette_path("simple_text", downstream, f"{mode}-00.yaml")
    upstream_path = _matched_cassette_path("simple_text", upstream, f"{mode}-00.yaml")
    body = (
        deepcopy(_read_cassette(downstream_path)["request"]["body"])
        | additions
        | {"stream": stream}
    )
    expected_body = deepcopy(_read_cassette(upstream_path)["request"]["body"]) | expected_additions
    if downstream == upstream:
        expected_body["stream"] = stream

    async with _get_http_client(
        [upstream_path],
        upstream,
        translated=downstream != upstream,
        expected_body=expected_body,
    ) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body,
            endpoint=upstream,
            provider=provider or _provider_from_cassette_path(upstream_path),
            api_key="test-api-key",
        )
        if stream:
            events = [event async for event in result]

    if not stream:
        _assert_matched_response(
            result,
            downstream,
            _completed_responses_body(
                _matched_cassette_path("simple_text", "responses", f"{mode}-00.yaml")
            ),
        )
        return

    # Reuse the recorded response without rewriting fixtures: these tests exercise
    # request projection and stream fidelity, not the provider's acceptance of the modified input.
    async with _get_http_client([downstream_path], downstream, expected_body=body) as http_client:
        if downstream == "messages":
            sdk_stream = await AsyncAnthropic(
                api_key="test-api-key", http_client=http_client
            ).messages.create(**body)
        else:
            sdk_client = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
            if downstream == "responses":
                sdk_stream = await sdk_client.responses.create(**body)
            else:
                sdk_stream = await sdk_client.chat.completions.create(**body)
        async with sdk_stream:
            sdk_events = [event async for event in sdk_stream]

    if downstream == upstream:
        assert [type(event) for event in events] == [type(event) for event in sdk_events]
        assert [event.model_dump(mode="json", exclude_unset=True) for event in events] == [
            event.model_dump(mode="json", exclude_unset=True) for event in sdk_events
        ]
        return

    actual_sse = [dumps_sse(event) for event in events]
    expected_sse = [dumps_sse(event) for event in sdk_events]
    # The recorded response predates these request overrides, so compare its original
    # reconstructible fields rather than treating its request metadata as a fresh result.
    recorded_body = _read_cassette(downstream_path)["request"]["body"]
    if downstream == "responses":
        assert_responses_stream_invariants(actual_sse)
        actual = [reconstructable_responses_event(sse, body=recorded_body) for sse in actual_sse]
        expected = [
            reconstructable_responses_event(sse, body=recorded_body) for sse in expected_sse
        ]
        if upstream == "messages":
            for event in actual + expected:
                if isinstance(event.get("response"), dict):
                    event["response"].pop("created_at", None)
    elif downstream == "chat/completions":
        actual = [
            chunk
            for sse in actual_sse
            for chunk in reconstructable_chat_completions_chunks(sse, body=body)
        ]
        expected = [
            chunk
            for sse in expected_sse
            for chunk in reconstructable_chat_completions_chunks(sse, body=body)
        ]
        if upstream == "messages":
            for chunk in actual + expected:
                chunk.pop("created", None)
    else:
        actual = [reconstructable_anthropic_event(sse) for sse in actual_sse]
        expected = [reconstructable_anthropic_event(sse) for sse in expected_sse]
        if upstream == "responses":
            _expect_responses_search_usage(expected, upstream_path)
    assert actual == expected


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize(
    "controls",
    [
        {"temperature": 0.45},
        {"top_p": 0.55},
        {"temperature": 0.0},
        {"top_p": 0.0},
        {},
        {"temperature": None, "top_p": None},
    ],
    ids=["temperature", "top_p", "zero_temperature", "zero_top_p", "omitted", "explicit_none"],
)
async def test_sampling_controls_preserve_public_request_and_sdk_output(
    downstream: Protocol, upstream: Protocol, stream: bool, controls: dict[str, float | None]
) -> None:
    """Ensures sampling values and omissions survive every route and downstream response mode."""
    await _assert_request_parity(
        downstream,
        upstream,
        controls,
        controls
        if downstream == upstream
        else {key: value for key, value in controls.items() if value is not None},
        stream=stream,
    )


async def _assert_prompt_cache_rejected(
    downstream: Protocol,
    upstream: Protocol,
    additions: dict[str, Any],
    match: str,
) -> None:
    """Assert an unportable cache value fails before any upstream request is made."""
    downstream_path = _simple_text_cassette(downstream)
    upstream_path = _simple_text_cassette(upstream)
    body = deepcopy(_read_cassette(downstream_path)["request"]["body"]) | additions

    async with _get_http_client([]) as http_client:
        with pytest.raises(UnsupportedInput, match=match):
            await _coordinator(Molto(http_client=http_client), downstream)(
                body=body,
                endpoint=upstream,
                provider=_provider_from_cassette_path(upstream_path),
                api_key="test-api-key",
            )


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_prompt_cache_translates_across_protocols(
    downstream: Protocol,
    upstream: Protocol,
):
    """Ensures cache settings reach every upstream route while preserving SDK event parity."""
    if downstream == "messages":
        additions = deepcopy(_MESSAGES_PROMPT_CACHE)
        expected_additions = (
            additions if upstream == "messages" else {"prompt_cache_options": {"mode": "implicit"}}
        )
        await _assert_request_parity(downstream, upstream, additions, expected_additions)
        return

    additions = deepcopy(_OPENAI_PROMPT_CACHE)
    if upstream == "messages":
        additions.pop("prompt_cache_retention")
        expected_additions = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    else:
        expected_additions = deepcopy(additions)
    await _assert_request_parity(downstream, upstream, additions, expected_additions)


@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
@pytest.mark.parametrize("key", [None, "codex-turn-1"])
@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (None, {}),
        ({}, {}),
        ({"mode": "implicit"}, {"cache_control": {"type": "ephemeral"}}),
        ({"ttl": "30m"}, {"cache_control": {"type": "ephemeral", "ttl": "1h"}}),
        ({"mode": "explicit"}, {}),
        ({"mode": "explicit", "ttl": "30m"}, {}),
    ],
)
async def test_messages_prompt_cache_options(downstream, key, options, expected):
    """Ensures Messages cache mode and lifetime are independent of the dropped caller key."""
    additions = {}
    if key is not None:
        additions["prompt_cache_key"] = key
    if options is not None:
        additions["prompt_cache_options"] = options
    await _assert_request_parity(
        downstream,
        "messages",
        additions,
        expected,
    )


@pytest.mark.parametrize(
    ("downstream", "additions", "upstream", "match"),
    [
        ("responses", {"prompt_cache_retention": "24h"}, "messages", "prompt cache retention"),
        (
            "chat/completions",
            {"prompt_cache_key": "turn", "prompt_cache_retention": "in_memory"},
            "messages",
            "prompt cache retention",
        ),
    ],
)
async def test_prompt_cache_rejects_unportable_values(
    downstream: Protocol,
    additions: dict[str, Any],
    upstream: Protocol,
    match: str,
):
    """Asserts that Messages routes reject retention guarantees they cannot preserve."""
    await _assert_prompt_cache_rejected(downstream, upstream, additions, match)


def _breakpoint_input(protocol: Protocol, position: str, marker: dict[str, Any]) -> dict[str, Any]:
    text_type = "input_text" if protocol == "responses" else "text"
    field = "cache_control" if protocol == "messages" else "prompt_cache_breakpoint"
    part = {"type": text_type, "text": "Cached prefix.", field: marker}
    suffix = {"type": text_type, "text": "Uncached suffix."}
    if position in {"image", "tool_image"}:
        part = (
            {"type": "input_image", "image_url": "https://example.com/image.png", field: marker}
            if protocol == "responses"
            else {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
                field: marker,
            }
            if protocol == "chat/completions"
            else {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/image.png"},
                field: marker,
            }
        )
    user = {"role": "user", "content": [suffix]}
    messages = [user]
    if position in {"user", "image"}:
        messages = [{"role": "user", "content": [part, suffix]}]
    elif position == "system":
        if protocol == "messages":
            return {"system": [part, suffix], "messages": [user]}
        messages = [
            {"role": "developer", "content": [part]},
            {"role": "developer", "content": [suffix]},
            user,
        ]
    elif position == "assistant":
        messages = [{"role": "assistant", "content": [part]}, user]
        if protocol == "responses":
            messages[0].update(type="message", status="completed")
            part.update(type="output_text", annotations=[], logprobs=[])
    elif position in {"tool", "tool_image"}:
        if protocol == "responses":
            messages = [
                {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{}",
                    "call_id": "c1",
                    "status": "completed",
                },
                {"type": "function_call_output", "call_id": "c1", "output": [part, suffix]},
                user,
            ]
        elif protocol == "chat/completions":
            messages = [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": [part, suffix]},
                user,
            ]
        else:
            messages = [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "c1", "name": "lookup", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "c1", "content": [part, suffix]}
                    ],
                },
                user,
            ]
    return {"input" if protocol == "responses" else "messages": messages}


@pytest.mark.parametrize(
    ("position", "downstream", "upstream"),
    [
        (position, downstream, upstream)
        for position in ("user", "image", "system", "assistant", "tool", "tool_image")
        for downstream, upstream in ROUTES
        if position != "tool_image" or "chat/completions" not in (downstream, upstream)
    ],
)
@pytest.mark.parametrize("mode", ["implicit", "explicit"])
async def test_cache_breakpoint_translation(downstream, upstream, position, mode):
    """Ensures translation preserves marker positions, cache modes, and SDK event parity."""
    openai_marker = {"mode": "explicit"}
    messages_marker = {"type": "ephemeral"}
    additions = _breakpoint_input(
        downstream, position, messages_marker if downstream == "messages" else openai_marker
    )
    if downstream == "messages":
        if mode == "implicit":
            additions["cache_control"] = {"type": "ephemeral"}
    else:
        additions["prompt_cache_options"] = {"mode": mode, "ttl": "30m"}
    if downstream == upstream:
        expected = deepcopy(additions)
    else:
        marker = {"type": "ephemeral", "ttl": "1h"} if upstream == "messages" else openai_marker
        expected = _breakpoint_input(upstream, position, marker)
        if upstream == "messages":
            if mode == "implicit":
                expected["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        else:
            expected["prompt_cache_options"] = {"mode": mode}
            if downstream != "messages":
                expected["prompt_cache_options"]["ttl"] = "30m"
    await _assert_request_parity(downstream, upstream, additions, expected)


@pytest.mark.parametrize("upstream", PROTOCOLS)
async def test_responses_assistant_cache_marker_with_history_id(upstream):
    """Ensures marked assistant history remains valid and Responses routes retain its ID."""
    additions = _breakpoint_input("responses", "assistant", {"mode": "explicit"})
    additions["input"][0]["id"] = "msg_123"
    additions["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
    if upstream == "responses":
        expected = deepcopy(additions)
    else:
        marker = (
            {"type": "ephemeral", "ttl": "1h"} if upstream == "messages" else {"mode": "explicit"}
        )
        expected = _breakpoint_input(upstream, "assistant", marker)
        if upstream == "chat/completions":
            expected["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
    await _assert_request_parity("responses", upstream, additions, expected)


@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
@pytest.mark.parametrize(("count", "mode"), [(5, "explicit"), (5, "implicit"), (4, "implicit")])
async def test_messages_cache_breakpoint_count_is_left_to_upstream(downstream, count, mode):
    """Ensures all markers reach Messages so the upstream owns changing count limits."""
    text_type = "input_text" if downstream == "responses" else "text"
    content = [
        {
            "type": text_type,
            "text": str(i),
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
        for i in range(count)
    ]
    content.append({"type": text_type, "text": "Uncached tail"})
    additions = {
        "input" if downstream == "responses" else "messages": [
            {"role": "user", "content": content}
        ],
        "prompt_cache_options": {"mode": mode, "ttl": "30m"},
    }
    expected_content = [
        {
            "type": "text",
            "text": str(i),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
        for i in range(count)
    ]
    expected_content.append({"type": "text", "text": "Uncached tail"})
    expected = {"messages": [{"role": "user", "content": expected_content}]}
    if mode == "implicit":
        expected["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    await _assert_request_parity(downstream, "messages", additions, expected)


@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
async def test_breakpoint_without_options_inherits_openai_defaults(downstream):
    """Ensures marker-only requests carry OpenAI cache defaults to Messages."""
    additions = _breakpoint_input(downstream, "user", {"mode": "explicit"})
    expected = _breakpoint_input("messages", "user", {"type": "ephemeral", "ttl": "1h"})
    expected["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    await _assert_request_parity(downstream, "messages", additions, expected)


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("ttl", ["5m", "1h"])
async def test_messages_request_cache_lifetime(upstream, ttl):
    """Ensures native lifetimes survive and OpenAI routes use the documented approximation."""
    additions = {"cache_control": {"type": "ephemeral", "ttl": ttl}}
    expected = (
        additions
        if upstream == "messages"
        else {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}
    )
    await _assert_request_parity("messages", upstream, additions, expected)


@pytest.mark.parametrize(
    ("upstream", "position"),
    [
        (upstream, position)
        for upstream in PROTOCOLS
        for position in ("user", "image", "system", "assistant", "tool", "tool_image")
        if upstream != "chat/completions" or position != "tool_image"
    ],
)
@pytest.mark.parametrize("ttl", ["5m", "1h"])
@pytest.mark.parametrize("mode", ["implicit", "explicit"])
async def test_messages_block_cache_lifetime(upstream, position, ttl, mode):
    """Ensures lifetime mapping preserves marker positions and automatic-cache mode."""
    additions = _breakpoint_input("messages", position, {"type": "ephemeral", "ttl": ttl})
    if mode == "implicit":
        additions["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    if upstream == "messages":
        expected = additions
    else:
        expected = _breakpoint_input(upstream, position, {"mode": "explicit"})
        expected["prompt_cache_options"] = {"mode": mode, "ttl": "30m"}
    await _assert_request_parity("messages", upstream, additions, expected)


@pytest.mark.parametrize("upstream", ["responses", "chat/completions"])
@pytest.mark.parametrize("automatic", [False, True])
async def test_messages_mixed_cache_lifetimes_use_one_openai_lifetime(upstream, automatic):
    """Ensures mixed lifetimes share one OpenAI TTL without losing markers or cache mode."""
    additions = _breakpoint_input("messages", "user", {"type": "ephemeral", "ttl": "1h"})
    additions["messages"][0]["content"][1]["cache_control"] = {
        "type": "ephemeral",
        "ttl": "5m",
    }
    if automatic:
        additions["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
    expected = _breakpoint_input(upstream, "user", {"mode": "explicit"})
    expected["input" if upstream == "responses" else "messages"][0]["content"][1][
        "prompt_cache_breakpoint"
    ] = {"mode": "explicit"}
    expected["prompt_cache_options"] = {
        "mode": "implicit" if automatic else "explicit",
        "ttl": "30m",
    }
    await _assert_request_parity("messages", upstream, additions, expected)
