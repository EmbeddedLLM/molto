import pytest
from openai.types.responses import Response, ResponseCompletedEvent, ResponseFunctionToolCall

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
    assert_output_item_order,
    make_handler,
    make_molto_client,
    make_openai_client,
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

    pydantic-ai converts json_schema into a function tool with `tool_choice=required`
    on the wire, so the cassette captures a function-call request/response — not a
    native structured-output one. As a result, the OpenAI SDK sees `text.format.type="text"`
    in the response (the API default, since no json_schema field was sent). Molto
    reconstructs the original json_schema format from the IR so callers always get
    back the format they requested.
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

    # ── OpenAI SDK (parametrized stream) ───────────────────────────────
    sdk_client = make_openai_client(handler, load_stream_cassette=stream)
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

    assert sdk_response.text.format.type == "text"
    assert sdk_response.text.verbosity == "medium"

    sdk_fc = next(o for o in sdk_response.output if isinstance(o, ResponseFunctionToolCall))
    assert isinstance(sdk_fc, ResponseFunctionToolCall)
    assert sdk_fc.name == "final_result"
    sdk_args = json_loads(sdk_fc.arguments)
    assert sdk_args["name"] == EXPECTED_NAME
    assert "job" in sdk_args

    # ── Stream SDK reference (matches Molto's internal streaming) ──────
    ref_client = make_openai_client(handler, load_stream_cassette=True)
    ref_events = [
        e async for e in await ref_client.responses.create(**{**req_body, "stream": True})
    ]
    assert isinstance(ref_events[-1], ResponseCompletedEvent)
    sdk_ref = ref_events[-1].response

    # ── Molto client ───────────────────────────────────────────────────
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
        event_types = [e.type for e in events]
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert event_types[-1] == "response.completed"

        assert_output_item_order(events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == sdk_ref.id
    assert response.status == "completed"

    # Molto translates the function_call response back to json_schema text format
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
    assert isinstance(msg, OpenAIOutputItem)
    assert msg.role == "assistant"
    assert msg.status == "completed"
    text_part = msg.content[0]
    assert isinstance(text_part, OpenAIOutputTextContent)
