import pytest

from molto.providers import OpenAIChatCompletionsProvider
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.utils.test import (
    STREAM_PARAMS,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    stream_reference_cassette_suffix,
)

CHAT_SETTINGS = {
    "frequency_penalty": 0.0,
    "logprobs": True,
    "max_completion_tokens": 20,
    "presence_penalty": 0.0,
    "prompt_cache_key": "chat-cache",
    "prompt_cache_retention": "24h",
    "store": False,
    "temperature": 0.45,
    "top_logprobs": 1,
    "top_p": 0.55,
}


def test_chat_model_settings_map_to_shared_ir_and_provider_settings() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-5.1",
        messages=[{"role": "user", "content": "hi"}],
        frequency_penalty=0.0,
        logprobs=True,
        max_completion_tokens=20,
        parallel_tool_calls=True,
        presence_penalty=0.0,
        prompt_cache_key="chat-cache",
        prompt_cache_retention="24h",
        reasoning_effort="high",
        service_tier="flex",
        store=False,
        temperature=0.45,
        top_logprobs=1,
        top_p=0.55,
    )

    ir = request.to_ir()

    assert ir.max_output_tokens == 20
    assert ir.reasoning_effort == "high"
    assert ir.openai.store is False
    assert ir.openai.prompt_cache_key == "chat-cache"
    assert ir.openai.prompt_cache_retention == "24h"
    assert ir.openai.service_tier == "flex"
    assert OpenAIChatCompletionsProvider._build_model_settings(ir) == {
        "temperature": 0.45,
        "top_p": 0.55,
        "max_tokens": 20,
        "openai_logprobs": True,
        "openai_top_logprobs": 1,
        "parallel_tool_calls": True,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "openai_reasoning_effort": "high",
        "openai_store": False,
        "openai_service_tier": "flex",
        "openai_prompt_cache_key": "chat-cache",
        "openai_prompt_cache_retention": "24h",
    }


def test_chat_max_tokens_is_the_legacy_fallback() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=11,
    )

    assert request.to_ir().max_output_tokens == 11


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_chat_model_settings_round_trip_through_sdk(stream: bool, cassette_manager) -> None:
    """Verify public Chat settings using parsed SDK responses and replayed Molto requests."""
    req_body = {
        "model": "gpt-4.1-nano",
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        **CHAT_SETTINGS,
        "stream": stream,
    }
    if stream:
        req_body["stream_options"] = {"include_usage": True}
    assert_attrs = ["model", "messages", *CHAT_SETTINGS]

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
        assert any(chunk.choices for chunk in sdk_chunks)
    else:
        assert sdk_response.choices[0].message.content == "pong"
        assert sdk_response.usage is not None

    stream_req_body = {
        **CHAT_SETTINGS,
        "model": "gpt-4.1-nano",
        "messages": req_body["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    stream_reference = None
    if not stream:
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
        assert molto_chunks[-1].usage is not None
        assert any(chunk.choices for chunk in molto_chunks)
    else:
        assert stream_reference is not None
        assert molto_response.id == stream_reference[-1].id
        assert molto_response.choices[0].message.content == "".join(
            chunk.choices[0].delta.content or "" for chunk in stream_reference if chunk.choices
        )
        assert molto_response.usage is not None
