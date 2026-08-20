from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from molto.adapters.base import BaseAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.normalized import (
    ModelResponseStreamEvent,
    NormalizedAPIStatusError,
    NormalizedRequest,
    TextInput,
    UserInput,
)


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
