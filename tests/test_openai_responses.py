import inspect

from openai.resources.responses.responses import AsyncResponses
from pydantic import TypeAdapter

from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStreamEvent,
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


def test_non_streaming_response_is_a_pydantic_model():
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
