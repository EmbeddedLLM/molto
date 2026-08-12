import inspect
from typing import get_args

import pytest
from openai.resources.responses.responses import AsyncResponses
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputItemParam
from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam
from openai.types.responses.response_stream_event import ResponseStreamEvent
from pydantic import TypeAdapter, ValidationError

from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesInputItemParam,
    OpenAIResponsesResponse,
    OpenAIResponsesStreamEvent,
    ResponseOutputMessageOptionalIDParam,
    ResponseReasoningItemOptionalIDParam,
)


def test_request_fields_match_responses_create_body_parameters():
    signature = inspect.signature(AsyncResponses.create)
    transport_parameters = {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
    sdk_parameters = {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in transport_parameters
    }

    assert set(OpenAIResponsesCreateRequest.model_fields) == sdk_parameters


def test_request_input_replaces_only_sdk_history_types():
    sdk_items = set(get_args(ResponseInputItemParam))
    molto_items = set(get_args(OpenAIResponsesInputItemParam))

    assert molto_items == sdk_items - {
        ResponseReasoningItemParam,
        ResponseOutputMessageParam,
    } | {
        ResponseReasoningItemOptionalIDParam,
        ResponseOutputMessageOptionalIDParam,
    }


def test_request_dump_is_ready_for_responses_create():
    body = OpenAIResponsesCreateRequest(
        input=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
        stream=True,
        tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
    )

    assert body.model_dump() == {
        "input": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4o-mini",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
    }


def test_request_validates_idless_provider_response_history():
    history = [
        {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "2+2=4"}],
            "status": "completed",
        },
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "4",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
    ]

    request = OpenAIResponsesCreateRequest(input=history, model="reasoning-model", stream=True)

    assert isinstance(request.input[0], ResponseReasoningItemOptionalIDParam)
    assert isinstance(request.input[1], ResponseOutputMessageOptionalIDParam)
    assert request.model_dump()["input"] == history


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "2+2=4"}],
        },
        {
            "type": "reasoning",
            "summary": "not-a-list",
        },
        {
            "id": "reasoning_123",
            "type": "reasoning",
            "summary": "not-a-list",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "4", "annotations": []}],
        },
        {"type": "unsupported"},
    ],
)
def test_request_rejects_invalid_response_history(item):
    with pytest.raises(ValidationError):
        OpenAIResponsesCreateRequest(input=[item], model="reasoning-model", stream=True)


def test_non_streaming_response_uses_sdk_model():
    assert OpenAIResponsesResponse is Response
    assert set(OpenAIResponsesResponse.model_fields) == {
        "id",
        "created_at",
        "error",
        "incomplete_details",
        "instructions",
        "metadata",
        "model",
        "object",
        "output",
        "parallel_tool_calls",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
        "background",
        "completed_at",
        "conversation",
        "max_output_tokens",
        "max_tool_calls",
        "moderation",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "status",
        "text",
        "top_logprobs",
        "truncation",
        "usage",
        "user",
    }

    response = OpenAIResponsesResponse.model_validate(
        {
            "id": "resp_123",
            "created_at": 1.0,
            "model": "gpt-4o-mini",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )

    assert response.id == "resp_123"
    assert response.model_dump()["object"] == "response"


def test_stream_event_union_wraps_every_sdk_event_and_field():
    molto_events = get_args(get_args(OpenAIResponsesStreamEvent)[0])
    sdk_events = get_args(get_args(ResponseStreamEvent)[0])

    assert len(molto_events) == len(sdk_events)
    assert all(
        any(issubclass(molto_event, sdk_event) for molto_event in molto_events)
        for sdk_event in sdk_events
    )
    assert all("sequence_number" in event.model_fields for event in molto_events)


def test_streaming_event_union_validates_by_type():
    event = TypeAdapter(OpenAIResponsesStreamEvent).validate_python(
        {
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "Hello",
            "item_id": "msg_123",
            "logprobs": [],
            "output_index": 0,
            "sequence_number": 1,
        }
    )

    assert event.type == "response.output_text.delta"
    assert event.delta == "Hello"
