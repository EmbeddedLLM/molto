import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
)

from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    _OpenAIStreamEvent,
)
from molto.utils.io import json_loads
from molto.utils.test import (
    STREAM_PARAMS,
    assert_output_item_order,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    stream_reference_cassette_suffix,
)

INSTRUCTIONS = "You are patrick jane, a consultant for the CBI"

STRUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "job": {"type": "string"},
    },
    "required": ["name", "job"],
    "additionalProperties": False,
}

EXPECTED_NAME = "Patrick Jane"


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_text_config_json_schema(stream: bool, cassette_manager):
    """Compare JSON schema output across clients, not request encoding details."""

    req_body = dict(
        model="gpt-4.1-nano",
        input=[{"role": "user", "content": "Who are you?"}],
        instructions=INSTRUCTIONS,
        text={
            "format": {
                "type": "json_schema",
                "name": "final_result",
                "description": "The final response which ends this conversation",
                "schema": STRUCT_SCHEMA,
                "strict": True,
            },
        },
        stream=stream,
    )

    assert_attrs = ["model", "input", "instructions"]

    # -- OpenAI SDK (parametrized stream) --------------------------------
    sdk_client = make_openai_client(
        cassette_manager,
        expected_request_body=req_body,
        assert_attrs=assert_attrs,
    )
    sdk_response = await sdk_client.responses.create(**req_body)
    if stream:
        sdk_events = [e async for e in sdk_response]
        sdk_event_types = [e.type for e in sdk_events]
        assert isinstance(sdk_events[-1], ResponseCompletedEvent)
        assert sdk_event_types[-1] == "response.completed"

        assert_output_item_order(sdk_events)

        sdk_response = sdk_events[-1].response
    assert isinstance(sdk_response, Response)
    assert sdk_response.status == "completed"

    assert sdk_response.text.format.type == "json_schema"
    assert sdk_response.text.verbosity == "medium"

    sdk_msg = next(o for o in sdk_response.output if isinstance(o, ResponseOutputMessage))
    assert sdk_msg.status == "completed"
    sdk_text = next(c for c in sdk_msg.content if isinstance(c, ResponseOutputText))
    sdk_output = json_loads(sdk_text.text)
    assert sdk_output["name"] == EXPECTED_NAME
    assert "job" in sdk_output

    stream_ref_suffix = stream_reference_cassette_suffix(stream)

    # -- Stream SDK reference (matches Molto's internal streaming) -------
    stream_req_body = {**req_body, "stream": True}
    if stream:
        sdk_ref = sdk_response
    else:
        ref_client = make_openai_client(
            cassette_manager,
            cassette_suffix=stream_ref_suffix,
            expected_request_body=stream_req_body,
            assert_attrs=assert_attrs,
        )
        ref_events = [e async for e in await ref_client.responses.create(**stream_req_body)]
        assert isinstance(ref_events[-1], ResponseCompletedEvent)
        sdk_ref = ref_events[-1].response

    # -- Molto client ----------------------------------------------------
    molto_req_body = dict(
        model="gpt-4.1-nano",
        input=[{"role": "user", "content": "Who are you?"}],
        instructions=INSTRUCTIONS,
        text={
            "format": {
                "type": "json_schema",
                "name": "final_result",
                "schema": STRUCT_SCHEMA,
                "strict": True,
                "description": "The final response which ends this conversation",
            }
        },
        stream=stream,
    )

    client = make_molto_client(
        cassette_manager,
        cassette_suffix=stream_ref_suffix,
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(molto_req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        event_types = [e.type for e in events]
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert event_types[-1] == "response.completed"

        assert_output_item_order(events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == sdk_ref.id
    assert response.status == "completed"

    messages = [o for o in response.output if isinstance(o, OpenAIOutputItem)]
    assert len(messages) == 1
    msg = messages[-1]
    assert msg.role == "assistant"
    assert msg.status == "completed"
    text_part = msg.content[0]
    assert isinstance(text_part, OpenAIOutputTextContent)
    molto_output = json_loads(text_part.text)
    assert molto_output["name"] == sdk_output["name"] == EXPECTED_NAME
    assert isinstance(molto_output.get("job"), str) and molto_output["job"]
    assert isinstance(sdk_output.get("job"), str) and sdk_output["job"]
