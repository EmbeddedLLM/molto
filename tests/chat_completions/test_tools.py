import pytest

from tests.support import (
    STREAM_PARAMS,
    run_chat_parity,
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


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_required_function_tool_round_trip_through_sdk(
    stream: bool, cassette_manager
) -> None:
    req_body = {**REQUEST_BODY, "stream": stream}
    if stream:
        req_body["stream_options"] = {"include_usage": True}

    parity = await run_chat_parity(cassette_manager, req_body)

    if stream:
        sdk_chunks = parity.sdk_chunks
        assert _choice_chunks(sdk_chunks)[-1].choices[0].finish_reason == "tool_calls"
        sdk_tool_chunks = _tool_call_chunks(sdk_chunks)
        assert sdk_tool_chunks
    else:
        sdk_response = parity.sdk_response
        response = parity.molto_response
        assert sdk_response.choices[0].finish_reason == "tool_calls"
        molto_tool_call = response.choices[0].message.tool_calls[0]
        sdk_tool_call = sdk_response.choices[0].message.tool_calls[0]
        assert response.choices[0].finish_reason == sdk_response.choices[0].finish_reason
        assert molto_tool_call.id == sdk_tool_call.id
        assert molto_tool_call.function.name == sdk_tool_call.function.name
        assert molto_tool_call.function.arguments == sdk_tool_call.function.arguments
