import json

import pytest

from molto.types.ir.portable import JsonObjectOutputFormat, JsonSchemaOutputFormat
from molto.types.openai import OpenAIRequest
from tests.support import STREAM_PARAMS, run_chat_parity, stream_chat_text

STRUCT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


RESPONSE_FORMATS = [
    pytest.param(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "final_result",
                "description": "A deterministic JSON response.",
                "schema": STRUCT_SCHEMA,
                "strict": True,
            },
        },
        id="json-schema",
    ),
    pytest.param({"type": "json_object"}, id="json-object"),
]

JSON_PROMPT = 'Return exactly the JSON object {"name":"Molto"}. Do not include markdown.'


def test_json_schema_response_format_maps_to_ir() -> None:
    request = OpenAIRequest.chat(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Who are you?"}],
        response_format={
            "type": "json_schema",
            "json_schema": STRUCT_SCHEMA,
            "name": "final_result",
            "description": "The final response.",
            "strict": True,
        },
    )

    text_format = request.to_ir().output_format

    assert isinstance(text_format, JsonSchemaOutputFormat)
    assert text_format.name == "final_result"
    assert text_format.schema_ == STRUCT_SCHEMA
    assert text_format.strict is True
    assert text_format.description == "The final response."


def test_json_object_response_format_maps_to_ir() -> None:
    request = OpenAIRequest.chat(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Return JSON."}],
        response_format={"type": "json_object"},
    )

    assert request.to_ir().output_format == JsonObjectOutputFormat()


@pytest.mark.parametrize("response_format", RESPONSE_FORMATS)
@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_response_format_round_trip_through_sdk(
    response_format: dict, stream: bool, cassette_manager
) -> None:
    req_body = {
        "model": "gpt-4.1-nano",
        "messages": [{"role": "user", "content": JSON_PROMPT}],
        "response_format": response_format,
        "stream": stream,
    }
    if stream:
        req_body["stream_options"] = {"include_usage": True}
    parity = await run_chat_parity(cassette_manager, req_body)
    if stream:
        assert parity.sdk_chunks[-1].usage is not None
        sdk_text = stream_chat_text(parity.sdk_chunks)
        molto_text = stream_chat_text(parity.molto_chunks)
    else:
        assert parity.sdk_response.usage is not None
        sdk_text = parity.sdk_response.choices[0].message.content
        assert sdk_text is not None
        molto_text = parity.molto_response.choices[0].message.content
        assert molto_text is not None
    json.loads(sdk_text)
    assert molto_text == sdk_text
