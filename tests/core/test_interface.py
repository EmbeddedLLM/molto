from copy import deepcopy
from functools import lru_cache
from itertools import pairwise, product
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from anthropic import AsyncAnthropic
from anthropic import AsyncStream as AnthropicAsyncStream
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI, AsyncStream
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import DEFAULT_PROVIDER_BY_ENDPOINT, PROVIDER_CAPABILITIES, Provider
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
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
Protocol = Literal["responses", "chat/completions", "messages"]
PROTOCOLS: tuple[Protocol, ...] = ("responses", "chat/completions", "messages")
ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(product(PROTOCOLS, PROTOCOLS))
ENCRYPTED_REASONING_PROTOCOLS: tuple[Protocol, ...] = ("responses", "messages")
ENCRYPTED_REASONING_ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(
    product(ENCRYPTED_REASONING_PROTOCOLS, repeat=2)
)
# Scenarios whose Responses streams carry encrypted reasoning. Messages has no
# equivalents for Responses include/stream options or item IDs, so these routes
# compare reconstructible requests and events instead of exact wire values.
ENCRYPTED_REASONING_SCENARIOS: tuple[str, ...] = (
    "reasoning_encrypted",
    "input_image_with_reasoning_encrypted",
    "tool_call_with_reasoning_encrypted",
)
TOOL_CALL_SCENARIOS: tuple[str, ...] = (
    "tool_call",
    "tool_call_choice",
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


def _cassette_path(
    protocol: Protocol,
    cassette_name: str,
    *,
    matched: bool = False,
) -> Path:
    relative_path = Path(cassette_name)
    if len(relative_path.parts) > 1:
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Cassette path must be relative to the cassette directory: {cassette_name}"
            )
        return CASSETTE_DIR / relative_path

    protocol_name = _protocol_name(protocol)
    if not matched:
        return CASSETTE_DIR / "openai" / protocol_name / "test_interface" / cassette_name

    scenario, separator, variant = cassette_name.partition("-")
    if not separator:
        raise ValueError(f"Matched cassette name must include a scenario: {cassette_name}")
    matches = list((CASSETTE_DIR / "matched" / scenario).glob(f"*-{protocol_name}-{variant}"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one matched {protocol!r} cassette for {cassette_name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


@lru_cache(maxsize=5)
def _read_cassette(path: Path) -> dict[str, Any]:
    """Read a cassette YAML file."""
    return read_yaml(path)


def _get_http_client(
    cassette_names: list[str],
    upstream: Protocol = "responses",
    *,
    matched: bool = False,
    translated: bool = False,
) -> AsyncClient:
    cassette_names = cassette_names.copy()

    def cassette_handler(request: Request) -> Response:
        cassette_name = cassette_names.pop(0)
        cassette = _read_cassette(_cassette_path(upstream, cassette_name, matched=matched))
        request_body = json_loads(request.content)
        try:
            assert request.method == cassette["request"]["method"]
            origin = (
                "https://api.anthropic.com" if upstream == "messages" else "https://api.openai.com"
            )
            assert str(request.url) == f"{origin}{cassette['request']['path']}"
            assert request.url.path == cassette["request"]["path"]
            expected_body = cassette["request"]["body"]
            if (
                matched
                and translated
                and upstream == "responses"
                and Path(cassette_name).name.partition("-")[0] in ENCRYPTED_REASONING_SCENARIOS
            ):
                _assert_reconstructable_encrypted_responses_request(
                    request_body,
                    expected_body,
                )
            elif (
                matched
                and translated
                and upstream == "responses"
                and Path(cassette_name).name.partition("-")[0] in TOOL_CALL_SCENARIOS
            ):
                _assert_reconstructable_tool_responses_request(request_body, expected_body)
            elif (
                matched
                and translated
                and upstream == "chat/completions"
                and Path(cassette_name).name.partition("-")[0] in TOOL_CALL_SCENARIOS
            ):
                _assert_reconstructable_tool_chat_request(request_body, expected_body)
            else:
                assert request_body == expected_body
        except AssertionError as e:
            return Response(
                400,
                content=f"Bad request: {repr(e)}".encode(),
                headers={"content-type": "text/plain; charset=utf-8"},
            )
        return Response(
            cassette["response"]["status_code"],
            content="".join(cassette["response"]["sse"]).encode(),
            headers={"content-type": "text/event-stream; charset=utf-8"},
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
    cassette_names = [
        "simple_text-stream-00.yaml",
    ]

    async with _get_http_client(cassette_names, matched=True) as http_client:
        client = Molto(http_client=http_client)

        # --- Extra inputs --- #
        cassette = _read_cassette(_cassette_path("responses", cassette_names[0], matched=True))
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            await client.responses(
                body=dict(
                    messages=[
                        {"role": "user", "content": "Write one short sentence about the ocean."}
                    ],
                    **cassette["request"]["body"],
                )
            )


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "cassette_name",
    [
        "simple_text-stream-00.yaml",
        "tool_call-stream-00.yaml",
        "tool_call-stream-01.yaml",
        "tool_call-stream-02.yaml",
    ],
)
async def test_native_stream_matches_provider_sdk(
    protocol: Protocol,
    cassette_name: str,
):
    cassette = _read_cassette(_cassette_path(protocol, cassette_name, matched=True))
    request_body = deepcopy(cassette["request"]["body"])

    async with _get_http_client([cassette_name], protocol, matched=True) as http_client:
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

    async with _get_http_client([cassette_name], protocol, matched=True) as http_client:
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
    "cassette_name",
    [
        "simple_text-stream-00.yaml",
        "reasoning_raw-stream-00.yaml",
        "reasoning_raw-stream-01.yaml",
        "tool_call-stream-00.yaml",
        "tool_call-stream-01.yaml",
        "tool_call-stream-02.yaml",
    ],
)
def test_matched_request_body_matches_across_protocols(
    downstream: Protocol,
    upstream: Protocol,
    cassette_name: str,
):
    downstream_body = deepcopy(
        _read_cassette(_cassette_path(downstream, cassette_name, matched=True))["request"]["body"]
    )
    request_type = {
        "responses": OpenAIResponsesCreateRequest,
        "chat/completions": OpenAIChatCompletionsCreateRequest,
        "messages": AnthropicMessagesCreateRequest,
    }[downstream]
    request = request_type.model_validate(downstream_body)

    if downstream == upstream:
        actual_body = request.model_dump()
    else:
        upstream_path = _cassette_path(upstream, cassette_name, matched=True)
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

    expected_body = _read_cassette(_cassette_path(upstream, cassette_name, matched=True))[
        "request"
    ]["body"]
    if (
        cassette_name.startswith("tool_call-")
        and upstream == "responses"
        and downstream != upstream
    ):
        _assert_reconstructable_tool_responses_request(actual_body, expected_body)
    else:
        assert actual_body == expected_body


async def test_responses_native_stream_normalizes_omitted_cache_write_tokens():
    cassette_name = "reasoning_raw-stream-00.yaml"
    cassette = _read_cassette(_cassette_path("responses", cassette_name, matched=True))
    request_body = deepcopy(cassette["request"]["body"])

    async with _get_http_client([cassette_name], "responses", matched=True) as http_client:
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
    cassette_names: list[str],
    downstream: Protocol,
    upstream: Protocol,
    *,
    matched: bool = False,
) -> None:
    num_turns = len(cassette_names)
    scenario = Path(cassette_names[0]).name.partition("-")[0]
    provider = (
        _provider_from_cassette_path(_cassette_path(upstream, cassette_names[0], matched=True))
        if matched
        else DEFAULT_PROVIDER_BY_ENDPOINT[upstream]
    )
    async with _get_http_client(
        cassette_names,
        upstream,
        matched=matched,
        translated=downstream != upstream,
    ) as http_client:
        client = Molto(http_client=http_client)
        for _ in range(num_turns):
            cassette = _read_cassette(
                _cassette_path(downstream, cassette_names.pop(0), matched=matched)
            )
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
    *,
    matched: bool = False,
) -> None:
    await _assert_stream_events(
        cassette_names,
        downstream="responses",
        upstream=upstream,
        matched=matched,
    )


async def _assert_matched_stream_events(
    cassette_dir: str,
    downstream: Protocol,
    upstream: Protocol,
) -> None:
    protocol_name = _protocol_name(downstream)
    filename_marker = f"-{protocol_name}-"
    cassette_names = [
        f"{cassette_dir}-{path.name.partition(filename_marker)[2]}"
        for path in sorted(
            (CASSETTE_DIR / "matched" / cassette_dir).glob(f"*{filename_marker}stream-*.yaml")
        )
    ]
    assert cassette_names, f"No matched cassettes found in {cassette_dir!r}"
    await _assert_stream_events(cassette_names, downstream, upstream, matched=True)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_simple_text_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("simple_text", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_reasoning_raw_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("reasoning_raw", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ENCRYPTED_REASONING_ROUTES)
async def test_reasoning_encrypted_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("reasoning_encrypted", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_input_image_base64_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("input_image_base64", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ENCRYPTED_REASONING_ROUTES)
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


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_tool_call_parallel_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("tool_call_parallel", downstream, upstream)


@pytest.mark.parametrize(("downstream", "upstream"), ENCRYPTED_REASONING_ROUTES)
async def test_tool_call_with_reasoning_encrypted_stream(downstream: Protocol, upstream: Protocol):
    await _assert_matched_stream_events("tool_call_with_reasoning_encrypted", downstream, upstream)


async def test_instructions_stream():
    cassette_names = [
        "instructions-stream-00.yaml",
        "instructions-stream-01.yaml",
    ]
    await _assert_responses_events(cassette_names)
