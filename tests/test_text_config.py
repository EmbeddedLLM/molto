import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
)

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
    make_handler,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    read_cassette,
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

    # body recorded in the cassette: pydantic-ai converts the
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

    # ── (1) Verify the cassette response carries the expected payload ──
    cassette_data = read_cassette(CASSETTE_PATH, stream=stream)
    if stream:
        completed_frames = [
            line
            for line in cassette_data
            if line.startswith("data: ") and '"type":"response.completed"' in line
        ]
        assert len(completed_frames) == 1
        cassette_response = json_loads(completed_frames[0][len("data: ") :])["response"]
    else:
        cassette_response = cassette_data

    assert cassette_response["text"]["format"]["type"] == "text"
    assert cassette_response["text"]["verbosity"] == "medium"
    assert cassette_response["output"][0]["type"] == "function_call"
    assert cassette_response["output"][0]["name"] == "final_result"
    cassette_args = json_loads(cassette_response["output"][0]["arguments"])
    assert cassette_args["name"] == EXPECTED_NAME
    assert "job" in cassette_args

    # ── (2) OpenAI SDK client — sends the wire body verbatim ───────────
    client = make_openai_client(handler, load_stream_cassette=stream)
    response = await client.responses.create(**req_body)
    if stream:
        events = [r async for r in response]
        assert ResponseFunctionCallArgumentsDeltaEvent in {type(e) for e in events}
        assert "response.function_call_arguments.delta" in {e.type for e in events}
        assert "response.function_call_arguments.done" in {e.type for e in events}
        assert isinstance(events[-1], ResponseCompletedEvent)
        response = events[-1].response
    assert isinstance(response, Response)

    assert response.status == "completed"

    sdk_calls = [o for o in response.output if isinstance(o, ResponseFunctionToolCall)]
    assert len(sdk_calls) == 1
    assert sdk_calls[0].name == "final_result"
    assert sdk_calls[0].status == "completed"
    sdk_args = json_loads(sdk_calls[0].arguments)
    assert sdk_args["name"] == EXPECTED_NAME
    assert "job" in sdk_args

    # ── (3) Molto client — caller passes `text.format = json_schema`;
    # pydantic-ai translates that into the `final_result` tool on the wire.
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

    assert response.status == "completed"

    # The caller's json_schema text format must round-trip back through from_ir.
    assert isinstance(response.text, OpenAITextConfig)
    assert isinstance(response.text.format, OpenAIJsonSchemaFormat)
    assert response.text.format.type == "json_schema"
    assert response.text.format.name == "final_result"
    assert response.text.format.strict is True
    # The schema may be augmented with `title`/`description` derived from the
    # text-format metadata, but the structural fields must round-trip verbatim.
    for key in ("type", "properties", "required", "additionalProperties"):
        assert response.text.format.schema_[key] == STRUCT_SCHEMA[key]

    # The structured payload is surfaced as the assistant message body — the
    # json_schema text format is the *response* contract, so Molto unwraps the
    # function-call args back into the final output text item.
    messages = [o for o in response.output if isinstance(o, OpenAIOutputItem)]
    assert len(messages) == 1
    msg = messages[-1]
    assert msg.role == "assistant"
    assert msg.status == "completed"
    text_part = msg.content[0]
    assert isinstance(text_part, OpenAIOutputTextContent)
