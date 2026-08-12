from collections.abc import AsyncIterator

import httpx
import pytest

from molto.types.normalized import (
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    TextDelta,
    TextOutput,
    Thinking,
    ThinkingDelta,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesStreamEvent,
    validate_openai_responses_stream_event,
)
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


def _unexpected_http(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"events() performed independent HTTP I/O: {request.url}")


def _response(*, usage: bool = False) -> dict[str, object]:
    response: dict[str, object] = {
        "id": "resp_123",
        "created_at": 1.9,
        "model": "reasoning-model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "service_tier": "default",
        "tool_choice": "auto",
        "tools": [],
    }
    if usage:
        response["usage"] = {
            "input_tokens": 11,
            "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 4},
            "total_tokens": 18,
        }
    return response


def _response_event(data: dict[str, object]) -> OpenAIResponsesStreamEvent:
    return validate_openai_responses_stream_event(data)


def _chat_chunk(
    *,
    choices: list[dict[str, object]],
    usage: dict[str, object] | None = None,
) -> OpenAIChatCompletionsStreamChunk:
    return OpenAIChatCompletionsStreamChunk.model_validate(
        {
            "id": "chatcmpl_123",
            "choices": choices,
            "created": 2,
            "model": "reasoning-model",
            "object": "chat.completion.chunk",
            "service_tier": "default",
            "usage": usage,
        }
    )


async def test_responses_events_translate_native_sdk_events(monkeypatch: pytest.MonkeyPatch):
    native_events = [
        _response_event(
            {
                "type": "response.created",
                "response": _response(),
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "summary": [],
                    "content": [],
                },
                "output_index": 0,
                "sequence_number": 1,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_text.delta",
                "content_index": 0,
                "delta": "Think",
                "item_id": "rs_123",
                "output_index": 0,
                "sequence_number": 2,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_123",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Summary"}],
                    "content": [{"type": "reasoning_text", "text": "Think"}],
                },
                "output_index": 0,
                "sequence_number": 3,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
                "output_index": 1,
                "sequence_number": 4,
            }
        ),
        _response_event(
            {
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "Answer",
                "item_id": "msg_123",
                "logprobs": [],
                "output_index": 1,
                "sequence_number": 5,
            }
        ),
        _response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Answer",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                "output_index": 1,
                "sequence_number": 6,
            }
        ),
        _response_event(
            {
                "type": "response.completed",
                "response": _response(usage=True),
                "sequence_number": 7,
            }
        ),
    ]
    captured_requests: list[OpenAIResponsesCreateRequest] = []

    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        captured_requests.append(native_request)
        for event in native_events:
            yield event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        events = [event async for event in client.events(NormalizedRequest(input="Hi", model="m"))]

    assert len(captured_requests) == 1
    assert isinstance(captured_requests[0], OpenAIResponsesCreateRequest)
    assert captured_requests[0].model_dump() == {"input": "Hi", "model": "m"}
    assert events == [
        ResponseStartEvent(
            id="resp_123", created_at=1, model="reasoning-model", service_tier="default"
        ),
        PartStartEvent(index=0, part=Thinking(content="", summary="", id="rs_123")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="Think")),
        PartEndEvent(
            index=0,
            part=Thinking(content="Think", summary="Summary", id="rs_123"),
        ),
        PartStartEvent(index=1, part=TextOutput(content="", id="msg_123")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="Answer")),
        PartEndEvent(index=1, part=TextOutput(content="Answer", id="msg_123")),
        ResponseEndEvent(
            id="resp_123",
            created_at=1,
            model="reasoning-model",
            service_tier="default",
            usage={
                "input_tokens": 11,
                "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 18,
            },
        ),
    ]


@pytest.mark.parametrize(
    "native_event",
    [
        _response_event(
            {
                "type": "response.in_progress",
                "response": _response(),
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.content_part.added",
                "content_index": 0,
                "item_id": "msg_123",
                "output_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.content_part.done",
                "content_index": 0,
                "item_id": "msg_123",
                "output_index": 0,
                "part": {"type": "output_text", "text": "Answer", "annotations": []},
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.reasoning_text.done",
                "content_index": 0,
                "item_id": "rs_123",
                "output_index": 0,
                "text": "Think",
                "sequence_number": 0,
            }
        ),
        _response_event(
            {
                "type": "response.output_text.done",
                "content_index": 0,
                "item_id": "msg_123",
                "logprobs": [],
                "output_index": 0,
                "text": "Answer",
                "sequence_number": 0,
            }
        ),
    ],
)
async def test_responses_events_ignore_native_sdk_events(
    monkeypatch: pytest.MonkeyPatch,
    native_event: OpenAIResponsesStreamEvent,
):
    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        yield native_event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        assert [event async for event in client.events(NormalizedRequest())] == []


async def test_responses_events_reject_unsupported_native_sdk_event(
    monkeypatch: pytest.MonkeyPatch,
):
    native_event = _response_event(
        {
            "type": "response.function_call_arguments.delta",
            "delta": "{}",
            "item_id": "call_123",
            "output_index": 0,
            "sequence_number": 0,
        }
    )

    async def stub_native_events(
        native_request: OpenAIResponsesCreateRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        yield native_event

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIResponsesClient(http_client)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        with pytest.raises(
            ValueError,
            match="Unsupported Responses stream event: response.function_call_arguments.delta",
        ):
            [event async for event in client.events(NormalizedRequest(model="m"))]


async def test_chat_completions_events_translate_native_sdk_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    native_chunks = [
        _chat_chunk(choices=[{"index": 0, "delta": {"role": "assistant"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"reasoning_content": "Think"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"content": "Answer"}}]),
        _chat_chunk(choices=[{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]),
        _chat_chunk(
            choices=[],
            usage={
                "prompt_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 18,
            },
        ),
    ]
    captured_requests: list[OpenAIChatCompletionsCreateRequest] = []

    async def stub_native_events(
        native_request: OpenAIChatCompletionsCreateRequest,
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        captured_requests.append(native_request)
        for chunk in native_chunks:
            yield chunk

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIChatCompletionsClient(http_client)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        events = [
            event
            async for event in client.events(NormalizedRequest(input="Hi", model="m", stream=True))
        ]

    assert len(captured_requests) == 1
    assert isinstance(captured_requests[0], OpenAIChatCompletionsCreateRequest)
    assert captured_requests[0].model_dump() == {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "m",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert events == [
        ResponseStartEvent(
            id="chatcmpl_123", created_at=2, model="reasoning-model", service_tier="default"
        ),
        PartStartEvent(index=0, part=Thinking(content="", summary="")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="Think")),
        PartEndEvent(index=0, part=Thinking(content="Think", summary="")),
        PartStartEvent(index=1, part=TextOutput(content="")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="Answer")),
        PartEndEvent(index=1, part=TextOutput(content="Answer")),
        ResponseEndEvent(
            id="chatcmpl_123",
            created_at=2,
            model="reasoning-model",
            service_tier="default",
            usage={
                "input_tokens": 11,
                "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 18,
            },
        ),
    ]


async def test_chat_completions_events_reject_empty_native_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    async def stub_native_events(
        native_request: OpenAIChatCompletionsCreateRequest,
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        if False:
            yield

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as http_client:
        client = OpenAIChatCompletionsClient(http_client)
        monkeypatch.setattr(client, "native_events", stub_native_events)

        with pytest.raises(
            ValueError,
            match="Chat Completions stream did not contain a response chunk",
        ):
            [event async for event in client.events(NormalizedRequest(model="m"))]
