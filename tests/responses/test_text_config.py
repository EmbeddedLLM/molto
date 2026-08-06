import pytest
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
)

from molto.types.openai.responses import (
    OpenAIJsonSchemaFormat,
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAITextConfig,
)
from molto.utils.io import json_loads
from tests.support import (
    STREAM_PARAMS,
    run_responses_parity,
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
async def test_json_schema_text_format_round_trip_through_sdk(stream: bool, cassette_manager):
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

    molto_req_body = dict(
        model="openai/gpt-4.1-nano",
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

    parity = await run_responses_parity(
        cassette_manager, req_body, molto_request_body=molto_req_body
    )
    sdk_response = parity.sdk_response
    response = parity.molto_response

    assert sdk_response.status == "completed"

    assert sdk_response.text.format.type == "json_schema"
    assert sdk_response.text.verbosity == "medium"

    sdk_msg = next(o for o in sdk_response.output if isinstance(o, ResponseOutputMessage))
    assert sdk_msg.status == "completed"
    sdk_text = next(c for c in sdk_msg.content if isinstance(c, ResponseOutputText))
    sdk_output = json_loads(sdk_text.text)
    assert sdk_output["name"] == EXPECTED_NAME
    assert "job" in sdk_output

    # -- Molto client ----------------------------------------------------
    assert response.id == sdk_response.id
    assert response.status == "completed"

    assert isinstance(response.text, OpenAITextConfig)
    assert response.text.verbosity == "medium"
    assert isinstance(response.text.format, OpenAIJsonSchemaFormat)
    assert response.text.format.type == "json_schema"
    assert response.text.format.name == "final_result"
    assert response.text.format.strict is True
    for key in ("type", "properties", "required", "additionalProperties"):
        assert response.text.format.schema_[key] == STRUCT_SCHEMA[key]

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
