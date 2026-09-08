import inspect
from collections.abc import AsyncIterator
from typing import NoReturn

import pytest
from pydantic import BaseModel

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.base import BaseAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    ModelResponseStreamEvent,
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


class NativeResponse(BaseModel):
    value: str


class Adapter(BaseAdapter[NativeRequest, ModelResponseStreamEvent, NativeResponse]):
    def _normalize_request(self, request: NativeRequest) -> NormalizedRequest:
        return NormalizedRequest(input=[UserInput(content=[TextInput(content=request.value)])])

    def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        del request
        return events

    async def _project_response(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> NativeResponse:
        del request
        response = await anext(events)
        if not isinstance(response, ResponseStartEvent):
            raise ValueError("Expected a response start event.")
        if response.id == "invalid":
            raise ValueError("Invalid normalized response.")
        return NativeResponse(value=response.id)

    def _handle_error(self, error: Exception) -> NoReturn:
        raise RuntimeError("Native projection failed.") from error


class InvalidNormalizedAdapter(Adapter):
    def _normalize_request(self, request: NativeRequest) -> NormalizedRequest:
        del request
        return NormalizedRequest.model_validate({"unknown": True})


class BrokenAdapter(Adapter):
    def _normalize_request(self, request: NativeRequest) -> NormalizedRequest:
        del request
        raise ValueError("Unexpected implementation error.")


class EmptyEvents(AsyncIterator[ModelResponseStreamEvent]):
    async def __anext__(self) -> ModelResponseStreamEvent:
        raise StopAsyncIteration


class CloseTrackingEvents(AsyncIterator[ModelResponseStreamEvent]):
    def __init__(self, event: ModelResponseStreamEvent) -> None:
        self.event = event
        self.closed = False

    async def __anext__(self) -> ModelResponseStreamEvent:
        return self.event

    async def aclose(self) -> None:
        self.closed = True


async def test_adapter_remembers_native_request_before_projecting_events():
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])
    native_request = NativeRequest(value="hello")
    normalized_request = adapter.normalize_request(native_request)
    events = EmptyEvents()

    assert adapter.native_request is native_request
    assert normalized_request.input == [UserInput(content=[TextInput(content="hello")])]
    assert [event async for event in adapter.project_events(events, normalized_request)] == []


def test_adapter_rejects_event_projection_before_request_normalization():
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])

    with pytest.raises(RuntimeError, match=r"normalize_request\(\).+project_events\(\)"):
        adapter.project_events(EmptyEvents(), NormalizedRequest())


def test_adapter_classifies_only_pydantic_request_validation_errors():
    native_request = NativeRequest(value="hello")

    with pytest.raises(UnsupportedInput):
        InvalidNormalizedAdapter(PROVIDER_CAPABILITIES["vllm"]).normalize_request(native_request)

    with pytest.raises(ValueError, match="Unexpected implementation error") as exc_info:
        BrokenAdapter(PROVIDER_CAPABILITIES["vllm"]).normalize_request(native_request)

    assert type(exc_info.value) is ValueError


def test_base_adapter_defines_abstract_translation_contract():
    assert inspect.isabstract(BaseAdapter)
    assert BaseAdapter.__abstractmethods__ == {
        "_handle_error",
        "_normalize_request",
        "_project_events",
        "_project_response",
    }
    assert not inspect.isabstract(OpenAIResponsesAdapter)
    assert not inspect.isabstract(OpenAIChatCompletionsAdapter)
    assert not inspect.isabstract(AnthropicMessagesAdapter)


async def test_adapter_rejects_response_projection_before_request_normalization():
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])

    with pytest.raises(RuntimeError, match=r"normalize_request\(\).+project_response\(\)"):
        await adapter.project_response(EmptyEvents(), NormalizedRequest())


async def test_adapter_collects_projected_events_into_response():
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])
    normalized_request = adapter.normalize_request(NativeRequest(value="hello"))

    async def events() -> AsyncIterator[ModelResponseStreamEvent]:
        yield ResponseStartEvent(id="response-id", created_at=1, model="m")

    assert await adapter.project_response(events(), normalized_request) == NativeResponse(
        value="response-id"
    )


async def test_adapter_closes_projected_events_after_response_collection():
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])
    normalized_request = adapter.normalize_request(NativeRequest(value="hello"))
    events = CloseTrackingEvents(ResponseStartEvent(id="response-id", created_at=1, model="m"))

    await adapter.project_response(events, normalized_request)

    assert events.closed


async def test_adapter_translates_response_projection_errors():
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])
    normalized_request = adapter.normalize_request(NativeRequest(value="hello"))

    async def events() -> AsyncIterator[ModelResponseStreamEvent]:
        yield ResponseStartEvent(id="invalid", created_at=1, model="m")

    with pytest.raises(RuntimeError, match="Native projection failed") as exc_info:
        await adapter.project_response(events(), normalized_request)

    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_adapter_translates_projection_errors() -> None:
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])
    normalized_request = adapter.normalize_request(NativeRequest(value="hello"))

    async def events() -> AsyncIterator[ModelResponseStreamEvent]:
        raise ValueError("Invalid IR stream.")
        yield

    with pytest.raises(RuntimeError, match="Native projection failed") as exc_info:
        await anext(adapter.project_events(events(), normalized_request))

    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_adapter_native_passthrough_translates_only_errors() -> None:
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])
    event = ResponseStartEvent(id="resp_123", created_at=1, model="m")

    async def native_events() -> AsyncIterator[ModelResponseStreamEvent]:
        yield event

    # Same-protocol passthrough requires no prior request normalization and
    # forwards events unchanged.
    assert [event async for event in adapter.project_native_events(native_events())] == [event]

    async def failing_events() -> AsyncIterator[ModelResponseStreamEvent]:
        raise ValueError("Invalid native stream.")
        yield

    with pytest.raises(RuntimeError, match="Native projection failed") as exc_info:
        await anext(adapter.project_native_events(failing_events()))

    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_adapter_native_response_passthrough_translates_only_errors() -> None:
    adapter = Adapter(PROVIDER_CAPABILITIES["vllm"])
    response = NativeResponse(value="hello")

    async def native_response() -> NativeResponse:
        return response

    assert await adapter.project_native_response(native_response()) is response

    async def failing_response() -> NativeResponse:
        raise ValueError("Invalid native response.")

    with pytest.raises(RuntimeError, match="Native projection failed") as exc_info:
        await adapter.project_native_response(failing_response())

    assert isinstance(exc_info.value.__cause__, ValueError)


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
