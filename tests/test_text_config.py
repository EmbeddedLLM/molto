import pytest

from molto.types.openai.responses import (
    OpenAIJsonSchemaFormat,
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAITextConfig,
    _OpenAIStreamEvent,
)
from molto.utils.io import json_loads
from molto.utils.test import (
    CASSETTE_DIR,
    STREAM_PARAMS,
    extract_cassette_response,
    extract_stream_cassette_response,
    make_handler,
    make_molto_client,
    prepend_model_prefix,
)

CASSETTE_PATH = CASSETTE_DIR / "openai" / "responses" / "text_config.yaml"

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
async def test_text_config_json_schema(stream: bool):
    """Round-trip a `text.format = json_schema` request through Molto.

    pydantic-ai turns the json_schema text format into a
    function tool with `tool_choice=required` on the wire — the body captured
    in the cassette. The response's structured payload comes back as the
    function-call arguments and must round-trip into the assistant message.
    """

    req_body = dict(
        model="gpt-4.1-nano",
        input=[{"role": "user", "content": "Who are you?"}],
        instructions=INSTRUCTIONS,
        tool_choice="required",
        text={
            "type": "json_schema",
            "name": "final_result",
            "description": "The final response which ends this conversation",
            "parameters": STRUCT_SCHEMA,
            "strict": True,
        },
        stream=stream,
    )

    handler = make_handler(
        CASSETTE_PATH,
        req_body,
        ["model", "input", "instructions"],
    )

    cassette_response = extract_cassette_response(CASSETTE_PATH, stream=stream)

    assert cassette_response["text"]["format"]["type"] == "text"
    assert cassette_response["text"]["verbosity"] == "medium"
    assert cassette_response["output"][0]["type"] == "function_call"
    assert cassette_response["output"][0]["name"] == "final_result"
    cassette_args = json_loads(cassette_response["output"][0]["arguments"])
    assert cassette_args["name"] == EXPECTED_NAME
    assert "job" in cassette_args

    stream_resp = extract_stream_cassette_response(CASSETTE_PATH)
    expected_molto_response_id = stream_resp["id"]

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

    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(molto_req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.completed"
        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == expected_molto_response_id
    assert response.status == "completed"

    assert isinstance(response.text, OpenAITextConfig)
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
