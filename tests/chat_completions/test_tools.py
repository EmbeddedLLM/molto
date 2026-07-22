import pytest

from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.utils.test import (
    STREAM_PARAMS,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    stream_reference_cassette_suffix,
)

REQUEST_BODY = {
    "model": "gpt-4.1-nano",
    "messages": [{"role": "user", "content": "What number am I thinking of?"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "guess_number",
                "description": "Guess a number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer", "description": "The guessed number."}
                    },
                    "required": ["number"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ],
    "tool_choice": "required",
}


def _choice_chunks(chunks):
    return [chunk for chunk in chunks if chunk.choices]


def _tool_call_chunks(chunks):
    return [chunk for chunk in _choice_chunks(chunks) if chunk.choices[0].delta.tool_calls]


def _assert_usage_matches(sdk_usage, molto_usage) -> None:
    assert sdk_usage is not None
    assert molto_usage is not None
    assert molto_usage.model_dump() == sdk_usage.model_dump()


def _assert_tool_call_deltas_match(sdk_chunk, molto_chunk) -> None:
    sdk_tool_calls = sdk_chunk.choices[0].delta.tool_calls or []
    molto_tool_calls = molto_chunk.choices[0].delta.tool_calls or []
    assert len(molto_tool_calls) == len(sdk_tool_calls)
    for sdk_tool_call, molto_tool_call in zip(sdk_tool_calls, molto_tool_calls, strict=True):
        assert molto_tool_call.index == sdk_tool_call.index
        assert molto_tool_call.id == sdk_tool_call.id
        assert molto_tool_call.type == sdk_tool_call.type
        assert molto_tool_call.function is not None
        assert sdk_tool_call.function is not None
        assert molto_tool_call.function.name == sdk_tool_call.function.name
        assert molto_tool_call.function.arguments == sdk_tool_call.function.arguments


def _assert_stream_chunks_match(sdk_chunks, molto_chunks) -> None:
    assert len(molto_chunks) == len(sdk_chunks)
    for sdk_chunk, molto_chunk in zip(sdk_chunks, molto_chunks, strict=True):
        if sdk_chunk.choices and sdk_chunk.choices[0].finish_reason is None:
            assert molto_chunk.id is None
        else:
            assert molto_chunk.id == sdk_chunk.id
        assert molto_chunk.object == sdk_chunk.object
        if sdk_chunk.usage is None:
            assert molto_chunk.usage is None
        else:
            _assert_usage_matches(sdk_chunk.usage, molto_chunk.usage)
        assert len(molto_chunk.choices) == len(sdk_chunk.choices)
        for sdk_choice, molto_choice in zip(sdk_chunk.choices, molto_chunk.choices, strict=True):
            assert molto_choice.index == sdk_choice.index
            assert molto_choice.finish_reason == sdk_choice.finish_reason
            assert molto_choice.delta.role == sdk_choice.delta.role
            assert molto_choice.delta.content == sdk_choice.delta.content
            assert molto_choice.delta.refusal == sdk_choice.delta.refusal
        if sdk_chunk.choices:
            _assert_tool_call_deltas_match(sdk_chunk, molto_chunk)


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_tool_choice_required(stream: bool, cassette_manager) -> None:
    req_body = {**REQUEST_BODY, "stream": stream}
    if stream:
        req_body["stream_options"] = {"include_usage": True}
    assert_attrs = ["model", "messages", "tool_choice"]

    sdk_client = make_openai_client(
        cassette_manager,
        cassette_endpoint="chat_completions",
        expected_request_body=req_body,
        assert_attrs=assert_attrs,
    )
    sdk_response = await sdk_client.chat.completions.create(**req_body)
    if stream:
        sdk_chunks = [chunk async for chunk in sdk_response]
        assert _choice_chunks(sdk_chunks)[-1].choices[0].finish_reason == "tool_calls"
        sdk_tool_chunks = _tool_call_chunks(sdk_chunks)
        assert sdk_tool_chunks
    else:
        assert sdk_response.choices[0].finish_reason == "tool_calls"

    stream_req_body = {
        **REQUEST_BODY,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    reference_chunks = None
    if not stream:
        reference_client = make_openai_client(
            cassette_manager,
            cassette_suffix=stream_reference_cassette_suffix(stream),
            cassette_endpoint="chat_completions",
            expected_request_body=stream_req_body,
            assert_attrs=assert_attrs,
        )
        reference_chunks = [
            chunk
            async for chunk in await reference_client.chat.completions.create(**stream_req_body)
        ]
    molto = make_molto_client(
        cassette_manager,
        cassette_suffix=stream_reference_cassette_suffix(stream),
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    response = await molto.chat(OpenAIChatCompletionRequest(**prepend_model_prefix(req_body)))
    if stream:
        molto_chunks = [chunk async for chunk in response]
        _assert_stream_chunks_match(sdk_chunks, molto_chunks)
    else:
        molto_tool_call = response.choices[0].message.tool_calls[0]
        assert reference_chunks is not None
        reference_tool_chunks = _tool_call_chunks(reference_chunks)
        assert response.id == reference_chunks[-1].id
        assert response.choices[0].finish_reason == sdk_response.choices[0].finish_reason
        assert molto_tool_call.id == reference_tool_chunks[0].choices[0].delta.tool_calls[0].id
        assert response.object == sdk_response.object
        _assert_usage_matches(reference_chunks[-1].usage, response.usage)
        assert (
            molto_tool_call.function.name
            == reference_tool_chunks[0].choices[0].delta.tool_calls[0].function.name
        )
        assert molto_tool_call.function.arguments == "".join(
            chunk.choices[0].delta.tool_calls[0].function.arguments or ""
            for chunk in reference_tool_chunks
        )
