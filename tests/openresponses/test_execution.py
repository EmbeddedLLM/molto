from collections.abc import AsyncIterator

import pytest

import molto.core
from molto import Molto
from molto.types.ir.portable import (
    AssistantMessageItem,
    ContentPartAddedEvent,
    ContentPartDeltaEvent,
    ContentPartDoneEvent,
    ItemStatus,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextPart,
    ResponseCompletedEvent,
    ResponseMetadata,
    ResponseStartedEvent,
    Usage,
)
from molto.types.openai import OpenAIRequest
from molto.types.openresponses import OpenResponsesRequest
from molto.utils.exceptions import ModelCapabilityError


async def _text_output_events() -> AsyncIterator:
    started = AssistantMessageItem(id="msg_1", status=ItemStatus.IN_PROGRESS, content=())
    done = AssistantMessageItem(
        id="msg_1",
        status=ItemStatus.COMPLETED,
        content=(OutputTextPart(text="Hello"),),
    )
    yield ResponseStartedEvent(
        response_id="resp_1",
        sequence_number=0,
        metadata=ResponseMetadata(model="gpt-test", object="response"),
    )
    yield OutputItemAddedEvent(response_id="resp_1", sequence_number=1, item_index=0, item=started)
    yield ContentPartAddedEvent(
        response_id="resp_1",
        sequence_number=2,
        item_id="msg_1",
        item_index=0,
        content_index=0,
        part=OutputTextPart(text=""),
    )
    yield ContentPartDeltaEvent(
        response_id="resp_1",
        sequence_number=3,
        item_id="msg_1",
        item_index=0,
        content_index=0,
        delta="Hello",
    )
    yield ContentPartDoneEvent(
        response_id="resp_1",
        sequence_number=4,
        item_id="msg_1",
        item_index=0,
        content_index=0,
        part=done.content[0],
    )
    yield OutputItemDoneEvent(response_id="resp_1", sequence_number=5, item_index=0, item=done)
    yield ResponseCompletedEvent(
        response_id="resp_1",
        sequence_number=6,
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class _FakeOpenAIProvider:
    constructed_endpoint = None

    def __init__(self, endpoint, **kwargs) -> None:
        del kwargs
        self.__class__.constructed_endpoint = endpoint

    def run(self, request) -> AsyncIterator:
        del request
        return _text_output_events()


async def test_anthropic_models_use_the_openai_compatible_chat_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(molto.core, "OpenAIProvider", _FakeOpenAIProvider)
    request = OpenResponsesRequest(model="anthropic/claude-test", input="Say hello.")

    response = await Molto().responses(request, api_key="test")

    assert _FakeOpenAIProvider.constructed_endpoint == "chat_completions"
    assert response.id == "resp_1"
    assert response.output[0].content[0].text == "Hello"


async def test_openai_responses_route_uses_the_responses_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(molto.core, "OpenAIProvider", _FakeOpenAIProvider)
    request = OpenAIRequest.responses(
        model="openai/gpt-test",
        input="Say hello.",
        stream=True,
    )

    response = await Molto().responses(request, api_key="test")

    assert _FakeOpenAIProvider.constructed_endpoint == "responses"
    assert [event async for event in response][-1].type == "response.completed"


async def test_anthropic_compatible_contract_rejects_images_before_provider_construction(
    monkeypatch,
) -> None:
    _FakeOpenAIProvider.constructed_endpoint = None
    monkeypatch.setattr(molto.core, "OpenAIProvider", _FakeOpenAIProvider)
    request = OpenResponsesRequest(
        model="anthropic/claude-test",
        input=(
            {
                "type": "message",
                "role": "user",
                "content": ({"type": "input_image", "image_url": "https://example.test"},),
            },
        ),
    )

    with pytest.raises(ModelCapabilityError, match="input.image"):
        await Molto().responses(request, api_key="test")

    assert _FakeOpenAIProvider.constructed_endpoint is None
