"""Cassette-backed parity for provider SDK output and native request validation."""

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import pytest
from anthropic import AsyncAnthropic
from anthropic import AsyncStream as AnthropicAsyncStream
from httpx import Request
from openai import AsyncOpenAI, AsyncStream
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.providers import DEFAULT_PROVIDER_BY_ENDPOINT, Provider
from molto.types.anthropic_messages import AnthropicMessagesResponse
from molto.types.openai_chat_completions import OpenAIChatCompletionsResponse
from molto.types.openai_responses import OpenAIResponsesResponse
from molto.utils.io import dumps_sse, json_loads
from molto.utils.test import (
    assert_chat_completions_chunks_match,
    reconstructable_anthropic_event,
    reconstructable_chat_completions_chunks,
    reconstructable_responses_event,
)
from tests.core import cassette_helpers
from tests.core.cassette_helpers import (
    MATCHED_CASSETTE_DIR,
    PROMPT_CACHE_CASES,
    PROTOCOLS,
    ROUTES,
    Protocol,
    _assert_matched_response,
    _assert_responses_stream_invariants,
    _get_http_client,
    _matched_cassette_path,
    _protocol_name,
    _provider_from_cassette_path,
    _read_cassette,
    _recorded_cassette_path,
    _sdk_stream_events,
    _split_sdk_request_body,
    _sse_records,
)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_transport_options_and_response_headers(
    downstream: Protocol,
    upstream: Protocol,
    stream: bool,
):
    """Ensures transport options preserve allowed headers and bodies while protecting credentials."""
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


@pytest.mark.parametrize(
    ("protocol", "provider", "scenario", "variant"),
    [
        *[
            (protocol, DEFAULT_PROVIDER_BY_ENDPOINT[protocol], scenario, variant)
            for protocol, (scenario, variant) in product(
                PROTOCOLS,
                [
                    ("simple_text", "stream-00.yaml"),
                    ("tool_call", "stream-00.yaml"),
                    ("tool_call", "stream-01.yaml"),
                    ("tool_call", "stream-02.yaml"),
                    *PROMPT_CACHE_CASES,
                ],
            )
        ],
        ("chat/completions", "togetherai", "tool_call_choice", "stream-00.yaml"),
    ],
)
async def test_native_stream_matches_provider_sdk(
    protocol: Protocol,
    provider: Provider,
    scenario: str,
    variant: str,
):
    """Ensures native provider streams preserve exact SDK events and request bodies."""
    cassette_path = (
        MATCHED_CASSETTE_DIR / scenario / f"{provider}-{_protocol_name(protocol)}-{variant}"
    )
    cassette = _read_cassette(cassette_path)
    request_body, extra_body = _split_sdk_request_body(
        deepcopy(cassette["request"]["body"]), protocol, provider
    )
    base_url = "https://api.together.ai/v1" if provider == "togetherai" else None

    sdk_events = await _sdk_stream_events(
        cassette_path, protocol, request_body, extra_body=extra_body, base_url=base_url
    )

    async with _get_http_client([cassette_path], protocol, base_url=base_url) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[protocol]
        molto_stream = await coordinator(
            body=request_body,
            endpoint=protocol,
            provider=provider,
            api_key="test-api-key",
            extra_body=extra_body,
        )
        molto_events = [event async for event in molto_stream]

    assert [type(event) for event in molto_events] == [type(event) for event in sdk_events]
    assert [event.model_dump(mode="json", exclude_unset=True) for event in molto_events] == [
        event.model_dump(mode="json", exclude_unset=True) for event in sdk_events
    ]


@pytest.mark.parametrize(
    ("provider", "scenario"),
    [
        *product(
            ["openai", "togetherai"], ["tool_call", "tool_call_parallel", "tool_call_choice"]
        ),
        ("vllm", "reasoning_raw"),
        ("togetherai", "reasoning_raw"),
    ],
)
async def test_native_chat_output_matches_next_turn_history(provider: Provider, scenario: str):
    """Ensures SDK-collected output matches next-turn YAML history and survives the next request."""
    paths = sorted(
        (MATCHED_CASSETTE_DIR / scenario).glob(f"{provider}-chat_completions-stream-*.yaml")
    )
    assert len(paths) >= 2
    history = deepcopy(_read_cassette(paths[0])["request"]["body"]["messages"])

    for turn, path in enumerate(paths):
        recorded_body = deepcopy(_read_cassette(path)["request"]["body"])
        expected_body = deepcopy(recorded_body)
        for actual, expected in zip(history, expected_body["messages"], strict=True):
            # Together emits empty content for tool-only output; the YAML history omits it.
            # Preserve that output on the wire and permit only this empty representation.
            if (
                actual.get("tool_calls")
                and actual.get("content") == ""
                and "content" not in expected
            ):
                expected["content"] = ""
        assert history == expected_body["messages"]
        body, extra_body = _split_sdk_request_body(recorded_body, "chat/completions", provider)
        body["messages"] = deepcopy(history)
        sdk_events = await _sdk_stream_events(
            path, "chat/completions", body, extra_body=extra_body
        )

        requests: list[Request] = []
        async with _get_http_client(
            [path], "chat/completions", expected_body=expected_body, requests=requests
        ) as http_client:
            async with await Molto(http_client=http_client).chat_completions(
                body=body, provider=provider, api_key="test-api-key", extra_body=extra_body
            ) as stream:
                events = [event async for event in stream]

        assert len(requests) == 1
        assert json_loads(requests[0].content) == expected_body
        assert [type(event) for event in events] == [type(event) for event in sdk_events]
        assert [event.model_dump(mode="json", exclude_unset=True) for event in events] == [
            event.model_dump(mode="json", exclude_unset=True) for event in sdk_events
        ]
        messages = []
        for chunks in (sdk_events, events):
            state = ChatCompletionStreamState()
            for chunk in chunks:
                state.handle_chunk(chunk)
            messages.append(state.get_final_completion().choices[0].message)
        assert messages[0].model_dump() == messages[1].model_dump()

        if turn + 1 < len(paths):
            assistant = messages[1].model_dump(
                exclude_none=True, exclude={"tool_calls": {"__all__": {"index"}}}
            )
            next_history = _read_cassette(paths[turn + 1])["request"]["body"]["messages"]
            continuation = deepcopy(next_history[len(history) + 1 :])
            assert continuation and all(item["role"] in {"tool", "user"} for item in continuation)
            calls = assistant.get("tool_calls", [])
            assert [item["tool_call_id"] for item in continuation if item["role"] == "tool"] == [
                call["id"] for call in calls
            ]
            history.extend([assistant, *continuation])


@pytest.mark.parametrize("provider", ["openai", "togetherai"])
@pytest.mark.parametrize("scenario", ["tool_call", "tool_call_parallel"])
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_native_chat_rejects_stream_indices_before_transport(
    provider: Provider, scenario: str, stream: bool
):
    """Ensures unconverted SDK tool indices fail before HTTP instead of silently dropping calls."""
    first_path, next_path = [
        _matched_cassette_path(scenario, "chat/completions", f"stream-{turn:02d}.yaml", provider)
        for turn in (0, 1)
    ]
    first_body, first_extra = _split_sdk_request_body(
        deepcopy(_read_cassette(first_path)["request"]["body"]), "chat/completions", provider
    )
    state = ChatCompletionStreamState()
    for event in await _sdk_stream_events(
        first_path, "chat/completions", first_body, extra_body=first_extra
    ):
        state.handle_chunk(event)
    assistant = state.get_final_completion().choices[0].message.model_dump(exclude_none=True)
    assert all("index" in call for call in assistant["tool_calls"])
    body, extra_body = _split_sdk_request_body(
        deepcopy(_read_cassette(next_path)["request"]["body"]), "chat/completions", provider
    )
    body["messages"][len(first_body["messages"])] = assistant
    body["stream"] = stream
    if not stream:
        body.pop("stream_options", None)
    requests: list[Request] = []
    async with _get_http_client([next_path], "chat/completions", requests=requests) as http_client:
        with pytest.raises(ValidationError, match="index"):
            await Molto(http_client=http_client).chat_completions(
                body=body, provider=provider, api_key="test-api-key", extra_body=extra_body
            )
    assert requests == []


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
    """Ensures native non-streaming responses preserve parsed SDK types and fields."""
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


async def test_responses_native_stream_normalizes_omitted_cache_write_tokens():
    """Ensures omitted vLLM cache-write tokens normalize to zero in native Responses streams."""
    cassette_path = _matched_cassette_path(
        "reasoning_raw", "responses", "stream-00.yaml", provider="vllm"
    )
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


async def test_instructions_stream():
    """Ensures recorded instructions and continuation streams preserve exact native SDK events."""
    for cassette_name in ("instructions-stream-00.yaml", "instructions-stream-01.yaml"):
        path = _recorded_cassette_path("responses", cassette_name)
        cassette = _read_cassette(path)
        body = deepcopy(cassette["request"]["body"])
        async with _get_http_client([path]) as http_client:
            sdk_stream = await AsyncOpenAI(
                api_key="test-api-key", http_client=http_client
            ).responses.create(**body)
            async with sdk_stream:
                sdk_events = [event async for event in sdk_stream]
        async with _get_http_client([path]) as http_client:
            stream = await Molto(http_client=http_client).responses(
                body=body, api_key="test-api-key"
            )
            events = [event async for event in stream]
        assert len(events) == len(_sse_records(cassette, "responses"))
        _assert_responses_stream_invariants([dumps_sse(event) for event in events])
        assert [type(event) for event in events] == [type(event) for event in sdk_events]
        assert [event.model_dump(mode="json", exclude_unset=True) for event in events] == [
            event.model_dump(mode="json", exclude_unset=True) for event in sdk_events
        ]


async def test_extra_inputs():
    """Ensures public Responses request validation rejects fields from another protocol."""
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


async def _assert_together_route(
    downstream: Protocol,
    stream: bool,
    base_url: str | None,
    scenario: str = "simple_text",
    turn: int = 0,
    upstream: Protocol = "chat/completions",
) -> None:
    """Compare Together routes with cassette-backed SDK output."""
    translated = downstream != upstream
    mode = "stream" if stream or translated else "non_stream"
    variant = f"{mode}-{turn:02d}.yaml"
    fixture_provider = "vllm" if scenario == "reasoning_raw" else None
    upstream_path = _matched_cassette_path(scenario, upstream, variant, provider=fixture_provider)
    downstream_path = _matched_cassette_path(
        scenario, downstream, variant, provider=fixture_provider
    )
    reference_protocol = "responses" if translated and not stream else downstream
    reference_path = _matched_cassette_path(
        scenario, reference_protocol, variant, provider=fixture_provider
    )
    reference_body = deepcopy(_read_cassette(reference_path)["request"]["body"])

    async with _get_http_client([reference_path], reference_protocol) as http_client:
        if reference_protocol == "messages":
            sdk_client = AsyncAnthropic(
                api_key="test-api-key",
                base_url="https://api.anthropic.com",
                http_client=http_client,
            )
            sdk_result = await sdk_client.messages.create(**reference_body)
        else:
            sdk_client = AsyncOpenAI(
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                http_client=http_client,
            )
            create = (
                sdk_client.responses.create
                if reference_protocol == "responses"
                else sdk_client.chat.completions.create
            )
            sdk_result = await create(**reference_body)
        if mode == "stream":
            async with sdk_result:
                sdk_events = [event async for event in sdk_result]

    requests: list[Request] = []
    async with _get_http_client(
        [upstream_path],
        upstream,
        requests=requests,
        base_url=base_url
        or {
            "responses": "https://api.openai.com/v1",
            "chat/completions": "https://api.together.ai/v1",
            "messages": "https://api.anthropic.com",
        }[upstream],
    ) as http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
        if translated:
            body["stream"] = stream
        result = await coordinator(
            body=body,
            provider="togetherai",
            api_key="test-api-key",
            base_url=base_url,
            **({"endpoint": upstream} if translated else {}),
        )
        if stream:
            events = [event async for event in result]

    assert len(requests) == 1
    if upstream == "messages":
        assert requests[0].headers["x-api-key"] == "test-api-key"
    else:
        assert requests[0].headers["authorization"] == "Bearer test-api-key"
    if not translated:
        actual = events if stream else [result]
        expected = sdk_events if stream else [sdk_result]
        assert [type(event) for event in actual] == [type(event) for event in expected]
        assert [event.model_dump(mode="json", exclude_unset=True) for event in actual] == [
            event.model_dump(mode="json", exclude_unset=True) for event in expected
        ]
    elif not stream:
        _assert_matched_response(result, downstream, sdk_events[-1].response.model_dump())
    elif downstream == "responses":
        _assert_responses_stream_invariants([dumps_sse(event) for event in events])
        actual_events = [
            reconstructable_responses_event(dumps_sse(event), body=deepcopy(body))
            for event in events
        ]
        expected_events = [
            reconstructable_responses_event(dumps_sse(event), body=deepcopy(body))
            for event in sdk_events
        ]
        if upstream == "messages":
            for event in actual_events + expected_events:
                if isinstance(event.get("response"), dict):
                    event["response"].pop("created_at", None)
        assert actual_events == expected_events
    elif downstream == "chat/completions":
        actual_chunks = [
            chunk
            for event in events
            for chunk in reconstructable_chat_completions_chunks(dumps_sse(event), body=body)
        ]
        expected_chunks = [
            chunk
            for event in sdk_events
            for chunk in reconstructable_chat_completions_chunks(dumps_sse(event), body=body)
        ]
        if upstream == "messages":
            for chunk in actual_chunks + expected_chunks:
                chunk.pop("created", None)
        assert_chat_completions_chunks_match(actual_chunks, expected_chunks)
    else:
        assert [reconstructable_anthropic_event(dumps_sse(event)) for event in events] == [
            reconstructable_anthropic_event(dumps_sse(event)) for event in sdk_events
        ]


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("custom_base_url", [False, True])
async def test_together_preserves_endpoint_and_matches_sdk(
    downstream: Protocol,
    upstream: Protocol,
    stream: bool,
    custom_base_url: bool,
) -> None:
    """Ensures Together retains every selected endpoint, SDK output, and HTTP destination."""
    base_url = None
    if custom_base_url:
        base_url = (
            "https://together.example" if upstream == "messages" else "https://together.example/v1"
        )
    await _assert_together_route(downstream, stream, base_url, upstream=upstream)


@pytest.mark.parametrize("downstream", PROTOCOLS)
@pytest.mark.parametrize("turn", [0, 1])
async def test_together_chat_reasoning_and_history_match_sdk(
    monkeypatch: pytest.MonkeyPatch,
    downstream: Protocol,
    turn: int,
) -> None:
    """Ensures Together reasoning deltas and continuation history preserve SDK semantics."""
    read_cassette = _read_cassette

    def with_together_reasoning(path: Path) -> dict[str, Any]:
        cassette = deepcopy(read_cassette(path))
        if "-chat_completions-" in path.name:
            for message in cassette["request"]["body"]["messages"]:
                if "reasoning_content" in message:
                    message["reasoning"] = message.pop("reasoning_content")
            # Extend existing replay data in memory; this is not a Together recording.
            cassette["response"]["sse"] = [
                record.replace('"reasoning_content":', '"reasoning":')
                for record in cassette["response"]["sse"]
            ]
        return cassette

    monkeypatch.setattr(cassette_helpers, "_read_cassette", with_together_reasoning)
    monkeypatch.setattr(__name__ + "._read_cassette", with_together_reasoning)
    await _assert_together_route(downstream, True, None, "reasoning_raw", turn)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_native_reasoning_array_matches_provider_sdk(stream: bool) -> None:
    """Ensures native reasoning arrays retain exact requests and parsed SDK replay output."""
    mode = "stream" if stream else "non_stream"
    path = _matched_cassette_path(
        "simple_text", "chat/completions", f"{mode}-00.yaml", provider="togetherai"
    )
    body, extra_body = _split_sdk_request_body(
        deepcopy(_read_cassette(path)["request"]["body"]), "chat/completions", "togetherai"
    )
    # Synthetic history exercises serialization against recorded output, not live acceptance.
    body["messages"][:0] = [
        {"role": "user", "content": "What is two plus two?"},
        {
            "role": "assistant",
            "reasoning": [" Two plus two is four.\n", "", "Two times two is also four. "],
            "content": "Four.",
        },
    ]
    requests: list[Request] = []
    results = []
    for use_molto in (False, True):
        async with _get_http_client(
            [path],
            "chat/completions",
            expected_body=body,
            expected_extra_body=extra_body,
            requests=requests,
        ) as http_client:
            if use_molto:
                result = await Molto(http_client=http_client).chat_completions(
                    body=body,
                    provider="togetherai",
                    api_key="test-api-key",
                    extra_body=extra_body,
                )
            else:
                result = await AsyncOpenAI(
                    api_key="test-api-key",
                    base_url="https://api.together.ai/v1",
                    http_client=http_client,
                ).chat.completions.create(**body, extra_body=extra_body)
            if stream:
                async with result:
                    results.append([event async for event in result])
            else:
                results.append([result])

    assert len(requests) == 2
    assert json_loads(requests[0].content) == json_loads(requests[1].content)
    sdk_results, molto_results = results
    assert [type(result) for result in molto_results] == [type(result) for result in sdk_results]
    assert [result.model_dump(mode="json", exclude_unset=True) for result in molto_results] == [
        result.model_dump(mode="json", exclude_unset=True) for result in sdk_results
    ]


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_messages_multiple_reasoning_blocks_match_provider_sdk(stream: bool) -> None:
    """Ensures multiple Messages thinking blocks reach Together and preserve SDK replay output."""
    path = _matched_cassette_path(
        "simple_text", "chat/completions", "stream-00.yaml", provider="togetherai"
    )
    blocks = ["Two plus two is four.", "Two times two is also four."]
    body = {
        "model": "gpt-4.1-nano",
        "max_tokens": 1024,
        "stream": stream,
        "messages": [
            {"role": "user", "content": "What is two plus two?"},
            {
                "role": "assistant",
                "content": [
                    *[{"type": "thinking", "thinking": block} for block in blocks],
                    {"type": "text", "text": "Four."},
                ],
            },
            {"role": "user", "content": "Write one short sentence about the ocean."},
        ],
    }
    # Only request history is synthesized; the response remains the recorded replay fixture.
    expected_body = {
        "model": body["model"],
        "max_completion_tokens": 1024,
        "stream": True,
        "stream_options": {"include_usage": True, "include_obfuscation": False},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is two plus two?"}],
            },
            {
                "role": "assistant",
                "reasoning": "\n".join(blocks),
                "content": "Four.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Write one short sentence about the ocean."}],
            },
        ],
    }
    sdk_events = await _sdk_stream_events(path, "chat/completions", expected_body)
    requests: list[Request] = []
    async with _get_http_client(
        [path], "chat/completions", expected_body=expected_body, requests=requests
    ) as http_client:
        result = await Molto(http_client=http_client).messages(
            body=body,
            endpoint="chat/completions",
            provider="togetherai",
            api_key="test-api-key",
        )
        if stream:
            events = [event async for event in result]
            message = events[0].message
            text = "".join(
                event.delta.text
                for event in events
                if event.type == "content_block_delta" and event.delta.type == "text_delta"
            )
            terminal = next(event for event in events if event.type == "message_delta")
            input_tokens = terminal.usage.input_tokens
            output_tokens = terminal.usage.output_tokens
            assert terminal.delta.stop_reason == "end_turn"
            assert events[-1].type == "message_stop"
        else:
            message = result
            text = "".join(block.text for block in message.content if block.type == "text")
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            assert message.stop_reason == "end_turn"

    assert len(requests) == 1
    assert type(message) is AnthropicMessagesResponse
    assert message.id == sdk_events[0].id
    assert message.model == sdk_events[0].model
    assert text == "".join(
        choice.delta.content or "" for event in sdk_events for choice in event.choices
    )
    sdk_usage = next(event.usage for event in sdk_events if event.usage is not None)
    assert input_tokens == sdk_usage.prompt_tokens
    assert output_tokens == sdk_usage.completion_tokens


@pytest.mark.parametrize("upstream", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_translated_reasoning_array_matches_provider_sdk(
    upstream: Protocol, stream: bool
) -> None:
    """Ensures Chat reasoning arrays preserve ordered upstream blocks and SDK replay semantics."""
    path = _matched_cassette_path("reasoning_raw", upstream, "stream-01.yaml")
    reference_path = _matched_cassette_path("reasoning_raw", "chat/completions", "stream-01.yaml")
    body = deepcopy(_read_cassette(reference_path)["request"]["body"])
    blocks = ["Two plus two is four.", "Two times two is also four."]
    assistant = body["messages"][1]
    assistant.pop("reasoning_content")
    assistant["reasoning"] = blocks
    body["stream"] = stream
    # Keep the fixture response intact while supplying explicit synthetic history expectations.
    expected_body = deepcopy(_read_cassette(path)["request"]["body"])
    if upstream == "responses":
        reasoning = expected_body["input"][1]
        expected_body["input"][1:2] = [
            {**reasoning, "content": [{"type": "reasoning_text", "text": block}]}
            for block in blocks
        ]
    else:
        expected_body["messages"][1]["content"][:1] = [
            {"type": "thinking", "thinking": block} for block in blocks
        ]
    sdk_events = await _sdk_stream_events(
        reference_path, "chat/completions", {**body, "stream": True}
    )
    requests: list[Request] = []
    async with _get_http_client(
        [path], upstream, expected_body=expected_body, requests=requests
    ) as http_client:
        result = await Molto(http_client=http_client).chat_completions(
            body=body,
            endpoint=upstream,
            provider="vllm",
            api_key="test-api-key",
        )
        if stream:
            events = [event async for event in result]
            actual = [
                chunk
                for event in events
                for chunk in reconstructable_chat_completions_chunks(dumps_sse(event), body=body)
            ]
            expected = [
                chunk
                for event in sdk_events
                for chunk in reconstructable_chat_completions_chunks(dumps_sse(event), body=body)
            ]
            if upstream == "messages":
                for chunk in actual + expected:
                    chunk.pop("created", None)
            assert_chat_completions_chunks_match(actual, expected)
        else:
            assert type(result) is OpenAIChatCompletionsResponse
            assert result.id == sdk_events[0].id
            assert result.model == sdk_events[0].model
            message = result.choices[0].message
            assert message.content == "".join(
                choice.delta.content or "" for event in sdk_events for choice in event.choices
            )
            assert message.reasoning_content == "".join(
                getattr(choice.delta, "reasoning_content", "") or ""
                for event in sdk_events
                for choice in event.choices
            )
            assert result.choices[0].finish_reason == "stop"
            sdk_usage = next(event.usage for event in sdk_events if event.usage is not None)
            assert result.usage.prompt_tokens == sdk_usage.prompt_tokens
            assert result.usage.completion_tokens == sdk_usage.completion_tokens
            assert result.usage.total_tokens == sdk_usage.total_tokens
    assert len(requests) == 1
