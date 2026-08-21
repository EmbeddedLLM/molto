from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.base import BaseAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.normalized import (
    ModelResponseStreamEvent,
    NormalizedAPIStatusError,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    TextInput,
    Thinking,
    ThinkingDelta,
    UserInput,
)
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest


class NativeRequest(BaseModel):
    value: str


class Adapter(BaseAdapter[NativeRequest, ModelResponseStreamEvent]):
    def _normalize_request(self, request: NativeRequest) -> NormalizedRequest:
        return NormalizedRequest(input=[UserInput(content=[TextInput(content=request.value)])])

    def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        del request
        return events

    def _project_api_status_error(self, error: NormalizedAPIStatusError) -> Exception:
        return RuntimeError(error.message)


class EmptyEvents(AsyncIterator[ModelResponseStreamEvent]):
    async def __anext__(self) -> ModelResponseStreamEvent:
        raise StopAsyncIteration


async def test_adapter_remembers_native_request_before_projecting_events():
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])
    native_request = NativeRequest(value="hello")
    normalized_request = adapter.normalize_request(native_request)
    events = EmptyEvents()

    assert adapter.native_request is native_request
    assert normalized_request.input == [UserInput(content=[TextInput(content="hello")])]
    projected = adapter.project_events(events, normalized_request)
    assert [event async for event in projected] == []


def test_adapter_rejects_event_projection_before_request_normalization():
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])

    with pytest.raises(RuntimeError, match=r"normalize_request\(\).+project_events\(\)"):
        adapter.project_events(EmptyEvents(), NormalizedRequest())


def test_base_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        BaseAdapter()


async def _projected(
    adapter: BaseAdapter,
    native_request: BaseModel,
    events: list[ModelResponseStreamEvent],
) -> list:
    normalized_request = adapter.normalize_request(native_request)

    async def stream() -> AsyncIterator[ModelResponseStreamEvent]:
        for event in events:
            yield event

    return [event async for event in adapter.project_events(stream(), normalized_request)]


async def test_reasoning_start_snapshot_projects_consistently_across_protocols() -> None:
    # One normalized stream with a non-empty reasoning start must project the
    # same full reasoning text through every downstream protocol, keeping each
    # stream internally consistent with its terminal snapshot.
    ir_events: list[ModelResponseStreamEvent] = [
        ResponseStartEvent(id="resp_123", created_at=1, model="m"),
        PartStartEvent(index=0, part=Thinking(content="Initial", summary="", id="rs_1")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta=" more")),
        PartEndEvent(index=0, part=Thinking(content="Initial more", summary="", id="rs_1")),
        ResponseEndEvent(id="resp_123", created_at=1, model="m", termination_reason="stop"),
    ]
    capabilities = PROVIDER_CAPABILITIES["vllm"]

    responses_events = await _projected(
        OpenAIResponsesAdapter(capabilities),
        OpenAIResponsesCreateRequest(input="Question", model="m", stream=True),
        ir_events,
    )
    part_add = next(
        event for event in responses_events if event.type == "response.content_part.added"
    )
    text_deltas = [
        event.delta for event in responses_events if event.type == "response.reasoning_text.delta"
    ]
    item_done = next(
        event for event in responses_events if event.type == "response.output_item.done"
    )
    streamed = part_add.part.text + "".join(text_deltas)
    assert streamed == "Initial more"
    assert [part.text for part in item_done.item.content] == [streamed]

    chat_events = await _projected(
        OpenAIChatCompletionsAdapter(capabilities),
        OpenAIChatCompletionsCreateRequest(
            messages=[{"role": "user", "content": "Hi"}], model="m", stream=True
        ),
        ir_events,
    )
    assert (
        "".join(
            reasoning
            for event in chat_events
            if event.choices
            and (reasoning := getattr(event.choices[0].delta, "reasoning_content", None))
        )
        == "Initial more"
    )

    message_events = await _projected(
        AnthropicMessagesAdapter(capabilities),
        AnthropicMessagesCreateRequest(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            model="m",
            stream=True,
        ),
        ir_events,
    )
    block_start = next(
        event
        for event in message_events
        if event.type == "content_block_start" and event.content_block.type == "thinking"
    )
    assert (
        block_start.content_block.thinking
        + "".join(
            event.delta.thinking
            for event in message_events
            if event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        )
        == "Initial more"
    )
