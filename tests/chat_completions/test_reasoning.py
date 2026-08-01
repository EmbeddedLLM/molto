from typing import Any

import pytest

from tests.support import (
    STREAM_PARAMS,
    assert_usage,
    run_chat_parity,
    stream_chat_text,
)

REQUEST_BODY = {
    "model": "gpt-5.1",
    "messages": [
        {
            "role": "user",
            "content": "How many Rs are in strawberry? Explain briefly.",
        }
    ],
    "reasoning_effort": "high",
    "max_completion_tokens": 512,
}


def _single_stream_usage(chunks: list[Any]) -> Any:
    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].choices == []
    return usage_chunks[0].usage


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_reasoning_output_round_trip_through_sdk(
    stream: bool,
    cassette_manager,
) -> None:
    """Preserve positive reasoning usage from parsed SDK Chat output."""
    req_body = {**REQUEST_BODY, "stream": stream}
    if stream:
        req_body["stream_options"] = {"include_usage": True}

    parity = await run_chat_parity(cassette_manager, req_body)

    if stream:
        sdk_chunks = parity.sdk_chunks
        assert any(chunk.choices for chunk in sdk_chunks)
        assert stream_chat_text(sdk_chunks).strip()
        sdk_usage = _single_stream_usage(sdk_chunks)
        assert_usage(sdk_usage, endpoint="chat", expect_reasoning=True)
        choice_chunks = [chunk for chunk in sdk_chunks if chunk.choices]
        assert choice_chunks[-1].choices[0].finish_reason is not None

        molto_chunks = parity.molto_chunks
        assert stream_chat_text(molto_chunks) == stream_chat_text(sdk_chunks)
        molto_usage = _single_stream_usage(molto_chunks)
        assert_usage(molto_usage, endpoint="chat", expect_reasoning=True)
    else:
        sdk_response = parity.sdk_response
        assert len(sdk_response.choices) == 1
        assert sdk_response.choices[0].index == 0
        assert sdk_response.choices[0].message.role == "assistant"
        assert sdk_response.choices[0].message.content
        assert sdk_response.choices[0].finish_reason is not None
        sdk_usage = sdk_response.usage
        assert_usage(sdk_usage, endpoint="chat", expect_reasoning=True)

        response = parity.molto_response
        assert response.choices[0].message.content == sdk_response.choices[0].message.content
        molto_usage = response.usage
        assert_usage(molto_usage, endpoint="chat", expect_reasoning=True)

    assert sdk_usage is not None
    assert molto_usage.completion_tokens_details.reasoning_tokens == (
        sdk_usage.completion_tokens_details.reasoning_tokens
    )
