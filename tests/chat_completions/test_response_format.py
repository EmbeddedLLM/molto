import json

import pytest

from molto.types.ir import JsonObjectFormat, JsonSchemaFormat
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.utils.test import (
    STREAM_PARAMS,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    stream_reference_cassette_suffix,
)

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
    request = OpenAIChatCompletionRequest(
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

    text_format = request.to_ir().text_format

    assert isinstance(text_format, JsonSchemaFormat)
    assert text_format.name == "final_result"
    assert text_format.json_schema == STRUCT_SCHEMA
    assert text_format.strict is True
    assert text_format.description == "The final response."


def test_json_object_response_format_maps_to_ir() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Return JSON."}],
        response_format={"type": "json_object"},
    )

    assert request.to_ir().text_format == JsonObjectFormat()


def _stream_text(chunks) -> str:
    return "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)


def _assert_usage_matches(sdk_usage, molto_usage) -> None:
    assert sdk_usage is not None
    assert molto_usage is not None
    assert molto_usage.model_dump() == sdk_usage.model_dump()


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
    assert_attrs = ["model", "messages", "response_format"]
    sdk_client = make_openai_client(
        cassette_manager,
        cassette_endpoint="chat_completions",
        expected_request_body=req_body,
        assert_attrs=assert_attrs,
    )
    sdk_response = await sdk_client.chat.completions.create(**req_body)
    if stream:
        sdk_chunks = [chunk async for chunk in sdk_response]
        assert sdk_chunks[-1].usage is not None
        sdk_text = _stream_text(sdk_chunks)
        stream_reference = sdk_chunks
    else:
        assert sdk_response.usage is not None
        sdk_text = sdk_response.choices[0].message.content
        assert sdk_text is not None
        stream_req_body = {**req_body, "stream": True, "stream_options": {"include_usage": True}}
        reference_client = make_openai_client(
            cassette_manager,
            cassette_suffix=stream_reference_cassette_suffix(stream),
            cassette_endpoint="chat_completions",
            expected_request_body=stream_req_body,
            assert_attrs=assert_attrs,
        )
        stream_reference = [
            chunk
            async for chunk in await reference_client.chat.completions.create(**stream_req_body)
        ]
    json.loads(sdk_text)
    stream_req_body = {**req_body, "stream": True, "stream_options": {"include_usage": True}}
    molto = make_molto_client(
        cassette_manager,
        cassette_suffix=stream_reference_cassette_suffix(stream),
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    molto_response = await molto.chat(
        OpenAIChatCompletionRequest(**prepend_model_prefix(req_body))
    )
    if stream:
        molto_chunks = [chunk async for chunk in molto_response]
        molto_text = _stream_text(molto_chunks)
        assert molto_text == sdk_text
        assert molto_chunks[-1].id == stream_reference[-1].id
        _assert_usage_matches(stream_reference[-1].usage, molto_chunks[-1].usage)
    else:
        molto_text = molto_response.choices[0].message.content
        assert molto_text is not None
        assert molto_text == sdk_text
        assert molto_response.id == stream_reference[-1].id
        _assert_usage_matches(stream_reference[-1].usage, molto_response.usage)
