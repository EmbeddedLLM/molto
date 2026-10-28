from copy import deepcopy
from functools import lru_cache
from itertools import pairwise, product
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from anthropic import AsyncAnthropic
from anthropic import AsyncStream as AnthropicAsyncStream
from anthropic import BadRequestError as AnthropicBadRequestError
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI, AsyncStream
from openai import BadRequestError as OpenAIBadRequestError
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import DEFAULT_PROVIDER_BY_ENDPOINT, PROVIDER_CAPABILITIES, Provider
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesResponse,
)
from molto.types.exceptions import UnsupportedInput
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsResponse,
)
from molto.types.openai_responses import OpenAIResponsesCreateRequest, OpenAIResponsesResponse
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from molto.utils.io import dumps_sse, json_loads, load_sse, read_yaml
from molto.utils.test import (
    assert_contains_expected,
    reconstructable_anthropic_event,
    reconstructable_chat_completions_chunks,
    reconstructable_responses_event,
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
PROMPT_CACHE_CASES: tuple[tuple[str, str], ...] = tuple(
    product(PROMPT_CACHE_SCENARIOS, ("stream-00.yaml", "stream-01.yaml"))
)


def _protocol_name(protocol: Protocol) -> str:
    return {
        "chat/completions": "chat_completions",
        "messages": "messages",
        "responses": "responses",
    }[protocol]


def _provider_from_cassette_path(path: Path) -> Provider:
    provider, separator, _ = path.name.partition("-")
    if not separator or provider not in PROVIDER_CAPABILITIES:
        raise ValueError(f"Cannot determine provider from matched cassette: {path}")
    return cast(Provider, provider)


def _recorded_cassette_path(protocol: Protocol, cassette_name: str) -> Path:
    return CASSETTE_DIR / "openai" / _protocol_name(protocol) / "test_interface" / cassette_name


def _matched_cassette_paths(
    scenario: str,
    protocol: Protocol,
    mode: Literal["stream", "non_stream"] = "stream",
) -> dict[str, Path]:
    protocol_name = _protocol_name(protocol)
    filename_marker = f"-{protocol_name}-"
    cassette_paths: dict[str, Path] = {}
    for path in sorted((MATCHED_CASSETTE_DIR / scenario).glob(f"*{filename_marker}{mode}-*.yaml")):
        variant = path.name.partition(filename_marker)[2]
        if variant in cassette_paths:
            raise ValueError(
                f"Expected one matched {protocol!r} cassette for {scenario!r}/{variant!r}"
            )
        cassette_paths[variant] = path
    return cassette_paths


def _matched_cassette_path(scenario: str, protocol: Protocol, variant: str) -> Path:
    protocol_name = _protocol_name(protocol)
    paths = sorted((MATCHED_CASSETTE_DIR / scenario).glob(f"*-{protocol_name}-{variant}"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one matched {protocol!r} cassette for {scenario!r}/{variant!r}, "
            f"found {len(paths)}."
        )
    return paths[0]


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
) -> AsyncClient:
    """Replay `cassette_paths`, optionally asserting one exact outgoing request body."""
    cassette_paths = cassette_paths.copy()

    def cassette_handler(request: Request) -> Response:
        if requests is not None:
            requests.append(request)
        cassette_path = cassette_paths.pop(0)
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
    adapter = OpenAIResponsesAdapter(PROVIDER_CAPABILITIES["openai"])

    def normalize(body: dict[str, Any]) -> dict[str, Any]:
        request = OpenAIResponsesCreateRequest.model_validate(body)
        normalized = adapter.normalize_request(request).model_dump(
            exclude={"extra_fields", "stream_options"}
        )
        for item in normalized.get("input") or []:
            if isinstance(item, dict):
                item.pop("id", None)
        return normalized

    # Messages has no equivalents for Responses include/stream options or item IDs.
    assert normalize(actual_body) == normalize(expected_body)


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


def _responses_output_text(response: dict[str, Any]) -> str:
    return "".join(
        part["text"]
        for item in response["output"]
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )


def _assert_responses_stream_invariants(sse_events: list[str]) -> None:
    events = [load_sse(sse) for sse in sse_events]
    sequence_numbers = [event["sequence_number"] for event in events]
    assert all(current == previous + 1 for previous, current in pairwise(sequence_numbers)), (
        f"Non-contiguous sequence numbers: {sequence_numbers}"
    )

    output_item_ids = [
        event["item"]["id"]
        for event in events
        if event.get("type") == "response.output_item.added"
    ]
    assert len(output_item_ids) == len(set(output_item_ids)), (
        f"Duplicate output item IDs: {output_item_ids}"
    )


async def test_extra_inputs():
    cassette_path = _matched_cassette_path("simple_text", "responses", "stream-00.yaml")

    async with _get_http_client([cassette_path]) as http_client:
        client = Molto(http_client=http_client)

        # --- Extra inputs --- #
        cassette = _read_cassette(cassette_path)
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            await client.responses(
                body=dict(
                    messages=[
                        {"role": "user", "content": "Write one short sentence about the ocean."}
                    ],
                    **cassette["request"]["body"],
                )
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


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    ("scenario", "variant"),
    [
        ("simple_text", "stream-00.yaml"),
        ("tool_call", "stream-00.yaml"),
        ("tool_call", "stream-01.yaml"),
        ("tool_call", "stream-02.yaml"),
        *PROMPT_CACHE_CASES,
    ],
)
async def test_native_stream_matches_provider_sdk(
    protocol: Protocol,
    scenario: str,
    variant: str,
):
    cassette_path = _matched_cassette_path(scenario, protocol, variant)
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
            body=request_body, endpoint=protocol, api_key="test-api-key"
        )
        molto_events = [event async for event in molto_stream]

    assert [type(event) for event in molto_events] == [type(event) for event in sdk_events]
    assert [event.model_dump(mode="json", exclude_unset=True) for event in molto_events] == [
        event.model_dump(mode="json", exclude_unset=True) for event in sdk_events
    ]


@pytest.mark.parametrize(
    ("scenario", "protocol"),
    [
        ("simple_text", "responses"),
        ("simple_text", "chat/completions"),
        ("simple_text", "messages"),
        ("reasoning_encrypted", "responses"),
        ("reasoning_encrypted", "messages"),
    ],
)
async def test_native_non_streaming_responses_match_provider_sdk(
    scenario: str,
    protocol: Protocol,
):
    cassette_paths = [
        _matched_cassette_path(scenario, protocol, variant)
        for variant in ("non_stream-00.yaml", "non_stream-01.yaml")
    ]
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


def _assert_matched_response(
    response: BaseModel,
    downstream: Protocol,
    expected: dict[str, Any],
) -> None:
    expected_text = _responses_output_text(expected)
    expected_usage = expected["usage"]
    response_type = {
        "responses": OpenAIResponsesResponse,
        "chat/completions": OpenAIChatCompletionsResponse,
        "messages": AnthropicMessagesResponse,
    }[downstream]
    assert type(response) is response_type
    assert response.id == expected["id"]
    assert response.model == expected["model"]

    if downstream == "responses":
        assert isinstance(response, OpenAIResponsesResponse)
        assert response.output_text == expected_text
        assert response.status == expected["status"]
        assert response.usage is not None
        assert response.usage.input_tokens == expected_usage["input_tokens"]
        assert response.usage.output_tokens == expected_usage["output_tokens"]
        assert response.usage.total_tokens == expected_usage["total_tokens"]
    elif downstream == "chat/completions":
        assert isinstance(response, OpenAIChatCompletionsResponse)
        assert len(response.choices) == 1
        assert response.choices[0].message.content == expected_text
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.prompt_tokens == expected_usage["input_tokens"]
        assert response.usage.completion_tokens == expected_usage["output_tokens"]
        assert response.usage.total_tokens == expected_usage["total_tokens"]
    else:
        assert isinstance(response, AnthropicMessagesResponse)
        assert "".join(block.text for block in response.content if block.type == "text") == (
            expected_text
        )
        assert response.stop_reason == "end_turn"
        input_details = expected_usage["input_tokens_details"]
        cache_write_tokens = input_details.get("cache_write_tokens", 0)
        cached_tokens = input_details.get("cached_tokens", 0)
        assert response.usage.input_tokens == (
            expected_usage["input_tokens"] - cache_write_tokens - cached_tokens
        )
        assert response.usage.cache_creation_input_tokens == cache_write_tokens
        assert response.usage.cache_read_input_tokens == cached_tokens
        assert response.usage.output_tokens == expected_usage["output_tokens"]

    expected_reasoning = next(
        (item for item in expected["output"] if item["type"] == "reasoning"),
        None,
    )
    if expected_reasoning is None:
        return

    expected_summary = "".join(part["text"] for part in expected_reasoning["summary"])
    expected_signature = expected_reasoning["encrypted_content"]
    if downstream == "responses":
        assert isinstance(response, OpenAIResponsesResponse)
        reasoning = next(item for item in response.output if item.type == "reasoning")
        assert "".join(part.text for part in reasoning.summary) == expected_summary
        assert reasoning.encrypted_content == expected_signature
    elif downstream == "messages":
        assert isinstance(response, AnthropicMessagesResponse)
        thinking = next(block for block in response.content if block.type == "thinking")
        assert thinking.thinking == expected_summary
        assert thinking.signature == expected_signature
    else:
        raise AssertionError("Chat Completions cannot preserve encrypted reasoning.")


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


async def _assert_matched_non_stream(
    scenario: str,
    downstream: Protocol,
    upstream: Protocol,
) -> None:
    translated = downstream != upstream
    mode: Literal["stream", "non_stream"] = "stream" if translated else "non_stream"
    downstream_paths = _matched_cassette_paths(scenario, downstream, mode)
    upstream_paths = _matched_cassette_paths(scenario, upstream, mode)
    responses_paths = _matched_cassette_paths(scenario, "responses", mode)
    assert downstream_paths, f"No matched {downstream!r} {mode} cassettes in {scenario!r}"
    assert downstream_paths.keys() == upstream_paths.keys() == responses_paths.keys(), (
        f"Mismatched cassette variants for {scenario!r}: "
        f"{downstream!r} has {sorted(downstream_paths)}, "
        f"{upstream!r} has {sorted(upstream_paths)}, "
        f"'responses' has {sorted(responses_paths)}"
    )
    variants = sorted(downstream_paths)

    async with _get_http_client(
        [upstream_paths[variant] for variant in variants],
        upstream,
        translated=translated,
    ) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        for variant in variants:
            request_body = deepcopy(_read_cassette(downstream_paths[variant])["request"]["body"])
            if translated:
                request_body["stream"] = False
            response = await coordinator(
                body=request_body,
                endpoint=upstream,
                provider=_provider_from_cassette_path(upstream_paths[variant]),
                api_key="test-api-key",
            )
            _assert_matched_response(
                response,
                downstream,
                _completed_responses_body(responses_paths[variant]),
            )


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize(
    ("scenario", "variant"),
    [
        ("simple_text", "stream-00.yaml"),
        ("reasoning_raw", "stream-00.yaml"),
        ("reasoning_raw", "stream-01.yaml"),
        ("tool_call", "stream-00.yaml"),
        ("tool_call", "stream-01.yaml"),
        ("tool_call", "stream-02.yaml"),
        *PROMPT_CACHE_CASES,
    ],
)
def test_matched_request_body_matches_across_protocols(
    downstream: Protocol,
    upstream: Protocol,
    scenario: str,
    variant: str,
):
    downstream_path = _matched_cassette_path(scenario, downstream, variant)
    upstream_path = _matched_cassette_path(scenario, upstream, variant)
    downstream_body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
    request_type = {
        "responses": OpenAIResponsesCreateRequest,
        "chat/completions": OpenAIChatCompletionsCreateRequest,
        "messages": AnthropicMessagesCreateRequest,
    }[downstream]
    request = request_type.model_validate(downstream_body)

    if downstream == upstream:
        actual_body = request.model_dump()
    else:
        capabilities = PROVIDER_CAPABILITIES[_provider_from_cassette_path(upstream_path)]
        adapter = {
            "responses": OpenAIResponsesAdapter(capabilities),
            "chat/completions": OpenAIChatCompletionsAdapter(capabilities),
            "messages": AnthropicMessagesAdapter(capabilities),
        }[downstream]
        normalized_request = adapter.normalize_request(request)
        project_request = {
            "responses": OpenAIResponsesClient._request_body,
            "chat/completions": OpenAIChatCompletionsClient._request_body,
            "messages": AnthropicMessagesClient._request_body,
        }[upstream]
        actual_body = project_request(normalized_request)

    expected_body = _read_cassette(upstream_path)["request"]["body"]
    if upstream == "responses" and downstream != upstream:
        expected_body = _without_reasoning_item_status(expected_body)
    if scenario == "tool_call" and upstream == "responses" and downstream != upstream:
        _assert_reconstructable_tool_responses_request(actual_body, expected_body)
    else:
        assert actual_body == expected_body


async def test_responses_native_stream_normalizes_omitted_cache_write_tokens():
    cassette_path = _matched_cassette_path("reasoning_raw", "responses", "stream-00.yaml")
    cassette = _read_cassette(cassette_path)
    request_body = deepcopy(cassette["request"]["body"])

    async with _get_http_client([cassette_path], "responses") as http_client:
        client = Molto(http_client=http_client)
        stream = await client.responses(
            body=request_body,
            endpoint="responses",
            provider="vllm",
            api_key="test-api-key",
        )
        events = [event async for event in stream]

    completed = next(event for event in events if event.type == "response.completed")
    assert completed.response.usage is not None
    assert completed.response.usage.input_tokens_details.cache_write_tokens == 0


async def _assert_stream_events(
    cassette_path_pairs: list[tuple[Path, Path]],
    downstream: Protocol,
    upstream: Protocol,
) -> None:
    first_downstream_path, first_upstream_path = cassette_path_pairs[0]
    scenario = first_downstream_path.parent.name
    provider = (
        _provider_from_cassette_path(first_upstream_path)
        if first_upstream_path.is_relative_to(MATCHED_CASSETTE_DIR)
        else DEFAULT_PROVIDER_BY_ENDPOINT[upstream]
    )
    async with _get_http_client(
        [upstream_path for _, upstream_path in cassette_path_pairs],
        upstream,
        translated=downstream != upstream,
    ) as http_client:
        client = Molto(http_client=http_client)
        for downstream_path, upstream_path in cassette_path_pairs:
            cassette = _read_cassette(downstream_path)
            coordinator = {
                "responses": client.responses,
                "chat/completions": client.chat_completions,
                "messages": client.messages,
            }[downstream]
            response = await coordinator(
                body=deepcopy(cassette["request"]["body"]),
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
                _assert_responses_stream_invariants(actual_sse_events)
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


async def _assert_responses_events(
    cassette_names: list[str],
    upstream: Protocol = "responses",
) -> None:
    cassette_path_pairs = [
        (
            _recorded_cassette_path("responses", cassette_name),
            _recorded_cassette_path(upstream, cassette_name),
        )
        for cassette_name in cassette_names
    ]
    await _assert_stream_events(
        cassette_path_pairs,
        downstream="responses",
        upstream=upstream,
    )


async def _assert_matched_stream_events(
    scenario: str,
    downstream: Protocol,
    upstream: Protocol,
) -> None:
    downstream_paths = _matched_cassette_paths(scenario, downstream)
    upstream_paths = _matched_cassette_paths(scenario, upstream)
    assert downstream_paths, f"No matched {downstream!r} cassettes found in {scenario!r}"
    assert downstream_paths.keys() == upstream_paths.keys(), (
        f"Mismatched cassette variants for {scenario!r}: "
        f"{downstream!r} has {sorted(downstream_paths)}, "
        f"{upstream!r} has {sorted(upstream_paths)}"
    )
    cassette_path_pairs = [
        (downstream_paths[variant], upstream_paths[variant])
        for variant in sorted(downstream_paths)
    ]
    await _assert_stream_events(cassette_path_pairs, downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_simple_text_non_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_non_stream("simple_text", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_simple_text_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("simple_text", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("scenario", PROMPT_CACHE_SCENARIOS)
async def test_prompt_cache_stream(scenario: str, downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events(scenario, downstream, upstream)


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


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_reasoning_raw_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("reasoning_raw", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), RESPONSES_MESSAGES_ROUTES)
async def test_reasoning_encrypted_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("reasoning_encrypted", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), RESPONSES_MESSAGES_ROUTES)
async def test_reasoning_encrypted_non_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_non_stream("reasoning_encrypted", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_input_image_base64_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("input_image_base64", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), RESPONSES_MESSAGES_ROUTES)
async def test_input_image_remote_url_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("input_image_remote_url", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), RESPONSES_MESSAGES_ROUTES)
async def test_input_image_with_reasoning_encrypted_stream(
    downstream: Protocol, upstream: Protocol
):
    await _assert_matched_stream_events(
        "input_image_with_reasoning_encrypted", downstream, upstream
    )


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_tool_call_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("tool_call", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_tool_call_choice_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("tool_call_choice", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), RESPONSES_MESSAGES_ROUTES)
async def test_tool_call_image_result_parallel_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("tool_call_image_result_parallel", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_tool_call_parallel_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("tool_call_parallel", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), RESPONSES_MESSAGES_ROUTES)
async def test_tool_call_with_reasoning_encrypted_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("tool_call_with_reasoning_encrypted", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_error_matched(downstream: Protocol, upstream: Protocol):
    scenario = "error-invalid_request_error"
    variant = "stream-00.yaml"
    downstream_path = _matched_cassette_path(scenario, downstream, variant)
    upstream_path = _matched_cassette_path(scenario, upstream, variant)
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


async def test_instructions_stream():
    cassette_names = [
        "instructions-stream-00.yaml",
        "instructions-stream-01.yaml",
    ]
    await _assert_responses_events(cassette_names)


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
) -> None:
    """Assert the outgoing request and parity with parsed downstream SDK events."""
    downstream_path = _simple_text_cassette(downstream)
    upstream_path = _simple_text_cassette(upstream)
    body = deepcopy(_read_cassette(downstream_path)["request"]["body"]) | additions
    expected_body = deepcopy(_read_cassette(upstream_path)["request"]["body"]) | expected_additions

    async with _get_http_client(
        [upstream_path],
        upstream,
        translated=downstream != upstream,
        expected_body=expected_body,
    ) as http_client:
        stream = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body,
            endpoint=upstream,
            provider=provider or _provider_from_cassette_path(upstream_path),
            api_key="test-api-key",
        )
        events = [event async for event in stream]

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
        _assert_responses_stream_invariants(actual_sse)
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
