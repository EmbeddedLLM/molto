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
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.exceptions import UnsupportedInput
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
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


def _matched_cassette_paths(scenario: str, protocol: Protocol) -> dict[str, Path]:
    protocol_name = _protocol_name(protocol)
    filename_marker = f"-{protocol_name}-"
    cassette_paths: dict[str, Path] = {}
    for path in sorted((MATCHED_CASSETTE_DIR / scenario).glob(f"*{filename_marker}stream-*.yaml")):
        variant = path.name.partition(filename_marker)[2]
        if variant in cassette_paths:
            raise ValueError(
                f"Expected one matched {protocol!r} cassette for {scenario!r}/{variant!r}"
            )
        cassette_paths[variant] = path
    return cassette_paths


def _matched_cassette_path(scenario: str, protocol: Protocol, variant: str) -> Path:
    cassette_paths = _matched_cassette_paths(scenario, protocol)
    try:
        return cassette_paths[variant]
    except KeyError as error:
        raise ValueError(
            f"No matched {protocol!r} cassette for {scenario!r}/{variant!r}"
        ) from error


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
async def test_transport_options_and_response_headers(
    downstream: Protocol,
    upstream: Protocol,
):
    variant = "stream-00.yaml"
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
        translated=downstream != upstream,
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
        stream = await coordinator(
            body=deepcopy(_read_cassette(downstream_path)["request"]["body"]),
            endpoint=upstream,
            provider=_provider_from_cassette_path(upstream_path),
            api_key="test-api-key",
            extra_body=extra_body,
            extra_headers=extra_headers,
        )

        assert stream.response_headers["x-codex-turn-state"] == "turn-state-123"
        assert stream.response_headers["x-request-id"] == "request-123"
        assert [event async for event in stream]

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
        for downstream_path, _ in cassette_path_pairs:
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
async def test_simple_text_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("simple_text", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_reasoning_raw_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("reasoning_raw", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), RESPONSES_MESSAGES_ROUTES)
async def test_reasoning_encrypted_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("reasoning_encrypted", downstream, upstream)


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


async def _assert_prompt_cache_request(
    downstream: Protocol,
    upstream: Protocol,
    additions: dict[str, Any],
    expected_additions: dict[str, Any],
) -> None:
    """Assert the outgoing cache fields and parity with parsed downstream SDK events."""
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
            provider=_provider_from_cassette_path(upstream_path),
            api_key="test-api-key",
        )
        events = [event async for event in stream]

    # Reuse the recorded response without rewriting fixtures: these tests exercise
    # request projection and stream fidelity, not the provider's cache-hit decisions.
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
    # reconstructible fields rather than treating its cache metadata as a fresh result.
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
    if downstream == "messages":
        additions = deepcopy(_MESSAGES_PROMPT_CACHE)
        expected_additions = (
            additions if upstream == "messages" else {"prompt_cache_options": {"mode": "implicit"}}
        )
        await _assert_prompt_cache_request(downstream, upstream, additions, expected_additions)
        return

    additions = deepcopy(_OPENAI_PROMPT_CACHE)
    if upstream == "messages":
        additions.pop("prompt_cache_retention")
        expected_additions = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    else:
        expected_additions = deepcopy(additions)
    await _assert_prompt_cache_request(downstream, upstream, additions, expected_additions)


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
    additions = {}
    if key is not None:
        additions["prompt_cache_key"] = key
    if options is not None:
        additions["prompt_cache_options"] = options
    await _assert_prompt_cache_request(
        downstream,
        "messages",
        additions,
        expected,
    )


@pytest.mark.parametrize(
    ("downstream", "additions", "upstream", "match"),
    [
        (
            "messages",
            {"cache_control": {"type": "ephemeral", "ttl": "1h"}},
            "responses",
            r"prompt cache ttl '1h'",
        ),
        (
            "messages",
            {"cache_control": {"type": "ephemeral", "ttl": "5m"}},
            "chat/completions",
            r"prompt cache ttl '5m'",
        ),
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
            {"role": "developer", "content": "Uncached suffix."},
            user,
        ]
    elif position == "assistant":
        messages = [{"role": "assistant", "content": [part]}, user]
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
    await _assert_prompt_cache_request(downstream, upstream, additions, expected)


@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
async def test_breakpoint_without_options_inherits_openai_defaults(downstream):
    additions = _breakpoint_input(downstream, "user", {"mode": "explicit"})
    expected = _breakpoint_input("messages", "user", {"type": "ephemeral", "ttl": "1h"})
    expected["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    await _assert_prompt_cache_request(downstream, "messages", additions, expected)


@pytest.mark.parametrize("upstream", ["responses", "chat/completions"])
@pytest.mark.parametrize("ttl", ["5m", "1h"])
async def test_messages_block_ttl_rejects_unrepresentable_openai_lifetime(upstream, ttl):
    additions = _breakpoint_input("messages", "user", {"type": "ephemeral", "ttl": ttl})
    await _assert_prompt_cache_rejected("messages", upstream, additions, "prompt cache ttl")
