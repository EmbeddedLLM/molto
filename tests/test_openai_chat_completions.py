import inspect

from openai.resources.chat.completions.completions import AsyncCompletions

from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsResponse,
    OpenAIChatCompletionsStreamChunk,
)


def test_request_fields_match_chat_completions_create_parameters():
    signature = inspect.signature(AsyncCompletions.create)
    transport_parameters = {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
    sdk_parameters = {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in transport_parameters
    }

    assert set(OpenAIChatCompletionsCreateRequest.model_fields) == sdk_parameters


def test_request_dump_is_ready_for_chat_completions_create():
    body = OpenAIChatCompletionsCreateRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
        temperature=0.2,
    )

    assert body.model_dump() == {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4o-mini",
        "temperature": 0.2,
    }


def test_non_streaming_chat_completion_is_a_pydantic_model():
    assert OpenAIChatCompletionsResponse.__module__ == "molto.types.openai_chat_completions"
    assert set(OpenAIChatCompletionsResponse.model_fields) == {
        "id",
        "choices",
        "created",
        "model",
        "object",
        "moderation",
        "service_tier",
        "system_fingerprint",
        "usage",
    }

    response = OpenAIChatCompletionsResponse.model_validate(
        {
            "id": "chatcmpl_123",
            "choices": [],
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion",
        }
    )

    assert response.id == "chatcmpl_123"


def test_streaming_chat_completion_chunk_is_a_pydantic_model():
    assert OpenAIChatCompletionsStreamChunk.__module__ == "molto.types.openai_chat_completions"
    assert set(OpenAIChatCompletionsStreamChunk.model_fields) == {
        "id",
        "choices",
        "created",
        "model",
        "object",
        "moderation",
        "service_tier",
        "system_fingerprint",
        "usage",
    }

    chunk = OpenAIChatCompletionsStreamChunk.model_validate(
        {
            "id": "chatcmpl_123",
            "choices": [],
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
        }
    )

    assert chunk.object == "chat.completion.chunk"
