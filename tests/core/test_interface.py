from copy import deepcopy
from functools import lru_cache
from itertools import pairwise, product
from pathlib import Path
from typing import Any, Literal

import pytest
from anthropic import AsyncAnthropic
from anthropic import AsyncStream as AnthropicAsyncStream
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI, AsyncStream
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
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


def _cassette_path(
    protocol: Protocol,
    cassette_name: str,
    *,
    matched: bool = False,
) -> Path:
    protocol_dir = {
        "chat/completions": "chat_completions",
        "messages": "messages",
        "responses": "responses",
    }[protocol]
    if not matched:
        return CASSETTE_DIR / "openai" / protocol_dir / "test_interface" / cassette_name

    scenario, separator, variant = cassette_name.partition("-")
    if not separator:
        raise ValueError(f"Matched cassette name must include a scenario: {cassette_name}")
    prefix = "anthropic" if protocol == "messages" else "openai"
    return CASSETTE_DIR / "matched" / scenario / f"{prefix}_{protocol_dir}-{variant}"


@lru_cache(maxsize=5)
def _read_cassette(path: Path) -> dict[str, Any]:
    """Read a cassette YAML file."""
    return read_yaml(path)


def _get_http_client(
    cassette_names: list[str],
    upstream: Protocol = "responses",
    *,
    matched: bool = False,
) -> AsyncClient:
    cassette_names = cassette_names.copy()

    def cassette_handler(request: Request) -> Response:
        cassette = _read_cassette(_cassette_path(upstream, cassette_names.pop(0), matched=matched))
        request_body = json_loads(request.content)
        try:
            assert request.method == cassette["request"]["method"]
            origin = (
                "https://api.anthropic.com" if upstream == "messages" else "https://api.openai.com"
            )
            assert str(request.url) == f"{origin}{cassette['request']['path']}"
            assert request.url.path == cassette["request"]["path"]
            assert request_body == cassette["request"]["body"]
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
async def test_native_stream_matches_provider_sdk(protocol: Protocol):
    cassette_name = "simple_text-stream-00.yaml"
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
def test_simple_text_request_body_matches_across_protocols(
    downstream: Protocol,
    upstream: Protocol,
):
    cassette_name = "simple_text-stream-00.yaml"
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
        adapter = {
            "responses": OpenAIResponsesAdapter(),
            "chat/completions": OpenAIChatCompletionsAdapter(),
            "messages": AnthropicMessagesAdapter(),
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
    async with _get_http_client(cassette_names, upstream, matched=matched) as http_client:
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
                provider="anthropic" if upstream == "messages" else "openai",
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
                    # timestamp. Every other translated field must still match exactly.
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


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_simple_text_stream(downstream: Protocol, upstream: Protocol):
    cassette_names = [
        "simple_text-stream-00.yaml",
        "simple_text-stream-01.yaml",
    ]
    await _assert_stream_events(cassette_names, downstream, upstream, matched=True)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_reasoning_stream_raw_content(downstream: Protocol, upstream: Protocol):
    cassette_names = [
        "reasoning_raw-stream-00.yaml",
        "reasoning_raw-stream-01.yaml",
    ]
    await _assert_stream_events(cassette_names, downstream, upstream, matched=True)


async def test_reasoning_stream_encrypted_content():
    cassette_names = [
        "reasoning-stream-00.yaml",
        "reasoning-stream-01.yaml",
    ]
    await _assert_responses_events(cassette_names)


async def test_instructions_stream():
    cassette_names = [
        "instructions-stream-00.yaml",
        "instructions-stream-01.yaml",
    ]
    await _assert_responses_events(cassette_names)
