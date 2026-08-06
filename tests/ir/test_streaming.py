import asyncio
import json

from molto.providers.openai.output_translation.chat import chat_stream_to_output_events
from molto.types.ir.events import PreservedOpenAIStreamEvent
from molto.types.openai.chat_completions.from_ir import (
    from_output_event_stream as chat_from_output_event_stream,
)
from molto.types.openai.cross_endpoint import (
    chat_from_responses_output_event_stream,
    responses_from_chat_output_event_stream,
)
from molto.types.openai.responses.from_ir import (
    from_output_event_stream as responses_from_output_event_stream,
)

CHAT_FRAME = {
    "id": "chat_1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "gpt-test",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": "hello"},
            "finish_reason": None,
        }
    ],
}


def _preserved(
    payload: dict[str, object],
    *,
    endpoint: str,
) -> PreservedOpenAIStreamEvent:
    return PreservedOpenAIStreamEvent(
        response_id="response_1",
        sequence_number=1,
        endpoint=endpoint,
        event_type=str(payload["type"] if "type" in payload else payload["object"]),
        payload_json=json.dumps(payload),
    )


async def test_chat_provider_translation_yields_first_frame_before_stream_end() -> None:
    release = asyncio.Event()

    async def payloads():
        yield CHAT_FRAME
        await release.wait()

    translated = chat_stream_to_output_events(payloads())
    await anext(translated)
    frame = await asyncio.wait_for(anext(translated), timeout=0.2)

    assert isinstance(frame, PreservedOpenAIStreamEvent)
    assert json.loads(frame.payload_json) == CHAT_FRAME
    release.set()


async def test_chat_output_composer_yields_native_frame_before_stream_end() -> None:
    release = asyncio.Event()

    async def events():
        yield _preserved(CHAT_FRAME, endpoint="chat_completions")
        await release.wait()

    stream = chat_from_output_event_stream(events())
    chunk = await asyncio.wait_for(anext(stream), timeout=0.2)

    assert chunk.choices[0].delta.content == "hello"
    release.set()


async def test_responses_output_composer_yields_native_event_before_stream_end() -> None:
    release = asyncio.Event()
    frame = {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hello",
    }

    async def events():
        yield _preserved(frame, endpoint="responses")
        await release.wait()

    stream = responses_from_output_event_stream(events())
    event = await asyncio.wait_for(anext(stream), timeout=0.2)

    assert event.type == "response.output_text.delta"
    assert event.delta == "hello"
    release.set()


async def test_responses_to_chat_composer_yields_text_delta_before_stream_end() -> None:
    release = asyncio.Event()
    frame = {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hello",
    }

    async def events():
        yield _preserved(frame, endpoint="responses")
        await release.wait()

    stream = chat_from_responses_output_event_stream(events())
    chunk = await asyncio.wait_for(anext(stream), timeout=0.2)

    assert chunk.choices[0].delta.content == "hello"
    release.set()


async def test_chat_to_responses_composer_yields_events_before_stream_end() -> None:
    release = asyncio.Event()

    async def events():
        yield _preserved(CHAT_FRAME, endpoint="chat_completions")
        await release.wait()

    stream = responses_from_chat_output_event_stream(events())
    created = await asyncio.wait_for(anext(stream), timeout=0.2)
    in_progress = await asyncio.wait_for(anext(stream), timeout=0.2)
    item_added = await asyncio.wait_for(anext(stream), timeout=0.2)
    part_added = await asyncio.wait_for(anext(stream), timeout=0.2)
    delta = await asyncio.wait_for(anext(stream), timeout=0.2)

    assert created.type == "response.created"
    assert in_progress.type == "response.in_progress"
    assert item_added.type == "response.output_item.added"
    assert part_added.type == "response.content_part.added"
    assert delta.type == "response.output_text.delta"
    assert delta.delta == "hello"
    release.set()


async def test_chat_to_responses_composer_preserves_function_call_events() -> None:
    frames = [
        {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"id":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]

    async def events():
        for frame in frames:
            yield _preserved(frame, endpoint="chat_completions")

    response_events = [event async for event in responses_from_chat_output_event_stream(events())]

    assert [event.type for event in response_events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert response_events[-1].response.output[0].arguments == '{"id":1}'


CHAT_USAGE_FRAMES = [
    CHAT_FRAME,
    {
        "id": "chat_1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": None,
    },
    # OpenAI sends the real usage in a trailing chunk with no choices.
    {
        "id": "chat_1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    },
]


async def test_chat_to_responses_composer_emits_trailing_usage_when_requested() -> None:
    async def events():
        for frame in CHAT_USAGE_FRAMES:
            yield _preserved(frame, endpoint="chat_completions")

    response_events = [
        event
        async for event in responses_from_chat_output_event_stream(events(), include_usage=True)
    ]

    terminal = response_events[-1]
    assert terminal.type == "response.completed"
    assert terminal.response.usage is not None
    assert terminal.response.usage["input_tokens"] == 5
    assert terminal.response.usage["output_tokens"] == 3
    assert terminal.response.usage["total_tokens"] == 8


async def test_chat_to_responses_composer_omits_usage_when_not_requested() -> None:
    async def events():
        for frame in CHAT_USAGE_FRAMES:
            yield _preserved(frame, endpoint="chat_completions")

    response_events = [
        event
        async for event in responses_from_chat_output_event_stream(events(), include_usage=False)
    ]

    terminal = response_events[-1]
    assert terminal.type == "response.completed"
    assert terminal.response.usage is None


RESPONSES_USAGE_FRAMES = [
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hello",
    },
    {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-test",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        },
    },
]


async def test_responses_to_chat_composer_emits_terminal_usage_when_requested() -> None:
    async def events():
        for frame in RESPONSES_USAGE_FRAMES:
            yield _preserved(frame, endpoint="responses")

    chunks = [
        chunk
        async for chunk in chat_from_responses_output_event_stream(events(), include_usage=True)
    ]

    terminal = chunks[-1]
    assert terminal.choices[0].finish_reason == "stop"
    assert terminal.usage is not None
    assert terminal.usage.prompt_tokens == 5
    assert terminal.usage.completion_tokens == 3
    assert terminal.usage.total_tokens == 8


async def test_responses_to_chat_composer_strips_terminal_usage_when_not_requested() -> None:
    async def events():
        for frame in RESPONSES_USAGE_FRAMES:
            yield _preserved(frame, endpoint="responses")

    chunks = [
        chunk
        async for chunk in chat_from_responses_output_event_stream(events(), include_usage=False)
    ]

    terminal = chunks[-1]
    assert terminal.choices[0].finish_reason == "stop"
    assert terminal.usage is None
