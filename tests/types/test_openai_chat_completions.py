import inspect

import pytest
from openai.resources.chat.completions.completions import AsyncCompletions
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import ValidationError

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


@pytest.mark.parametrize(
    "reasoning_field, value",
    [
        ("reasoning_content", "  Think\n\ncarefully.\t "),
        ("reasoning", "  Think\n\ncarefully.\t "),
        ("reasoning", ["  Think\n\ncarefully.\t ", "", "Another thought"]),
        ("reasoning", []),
    ],
)
def test_assistant_reasoning_extension_round_trips_sdk_history(reasoning_field, value):
    """Ensures SDK assistant reasoning survives strict request validation and serialization."""
    response = ChatCompletion.model_validate(
        {
            "id": "chatcmpl_123",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Answer",
                        reasoning_field: value,
                    },
                }
            ],
            "created": 1,
            "model": "m",
            "object": "chat.completion",
        }
    )
    message = response.choices[0].message.model_dump(exclude_none=True)
    request = OpenAIChatCompletionsCreateRequest(model="m", messages=[message])

    assert request.model_dump()["messages"] == [message]
    assert request.model_dump()["messages"][0][reasoning_field] == value


@pytest.mark.parametrize("role", ["system", "developer", "user", "tool"])
def test_reasoning_extension_is_restricted_to_assistant_history(role):
    """Asserts that Together reasoning does not weaken validation of other message roles."""
    message = {"role": role, "content": "Text", "reasoning": "Thought"}
    if role == "tool":
        message["tool_call_id"] = "call_1"

    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(model="m", messages=[message])


@pytest.mark.parametrize(
    "value", [None, 3, {}, [3], [None], [{}], [["Thought"]], ["Thought", False]]
)
def test_assistant_reasoning_extension_rejects_malformed_values(value):
    """Asserts that assistant reasoning accepts only strings and arrays of strings."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="m", messages=[{"role": "assistant", "reasoning": value}]
        )


@pytest.mark.parametrize("value", [[], ["Thought"]])
def test_assistant_reasoning_content_remains_string_only(value):
    """Ensures the reasoning array extension does not change reasoning_content validation."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="m", messages=[{"role": "assistant", "reasoning_content": value}]
        )


def test_assistant_reasoning_extension_preserves_unknown_field_rejection():
    """Ensures adding reasoning leaves unsupported assistant fields invalid."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="m",
            messages=[{"role": "assistant", "reasoning": "Thought", "unknown": "value"}],
        )


@pytest.mark.parametrize(
    "tool_type, input_field, input_value",
    [("function", "arguments", '{"command":"pwd"}'), ("custom", "input", "pwd")],
)
@pytest.mark.parametrize("malformation", ["stream-index", "unknown-field", "missing-input"])
def test_assistant_tool_calls_are_validated_at_construction(
    tool_type, input_field, input_value, malformation
):
    """Ensures malformed function and custom calls fail before lazy serialization can lose them."""
    details = {"name": "shell", input_field: input_value}
    tool_call = {"id": "call_1", "type": tool_type, tool_type: details}
    if malformation == "stream-index":
        tool_call["index"] = 0
    elif malformation == "unknown-field":
        details["cache_control"] = {"type": "ephemeral"}
    else:
        del details[input_field]

    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="m", messages=[{"role": "assistant", "tool_calls": [tool_call]}]
        )


def test_request_dump_preserves_assistant_tool_calls():
    """Ensures repeated serialization retains function and custom calls with assistant reasoning."""
    message = {
        "role": "assistant",
        "content": "Running both tools.",
        "reasoning": "Use each tool once.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
            },
            {
                "id": "call_2",
                "type": "custom",
                "custom": {"name": "shell", "input": "pwd"},
            },
        ],
    }
    body = OpenAIChatCompletionsCreateRequest(model="m", messages=[message])
    expected = {"model": "m", "messages": [message]}

    assert body.model_dump() == expected
    assert body.model_dump() == expected


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
def test_request_dump_preserves_content_part_lists(role):
    """Ensures validation and repeated serialization preserve message and part boundaries."""
    body = OpenAIChatCompletionsCreateRequest(
        messages=[
            {
                "role": role,
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "text", "text": "Part two"},
                ],
            },
            {
                "role": role,
                "content": [{"type": "text", "text": "Follow-up"}],
            },
        ],
        model="gpt-4o-mini",
    )

    expected = {
        "messages": [
            {
                "role": role,
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "text", "text": "Part two"},
                ],
            },
            {
                "role": role,
                "content": [{"type": "text", "text": "Follow-up"}],
            },
        ],
        "model": "gpt-4o-mini",
    }
    assert body.model_dump() == expected
    assert body.model_dump() == expected


@pytest.mark.parametrize("role", ["system", "developer", "assistant"])
def test_request_rejects_malformed_instruction_content_parts(role):
    """Asserts that local schema extensions reject instruction and assistant parts lacking text."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            messages=[{"role": role, "content": [{"type": "text"}]}],
            model="gpt-4o-mini",
        )


def test_request_dump_preserves_tool_content_part_lists():
    body = OpenAIChatCompletionsCreateRequest(
        messages=[
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "text", "text": "Part two"},
                ],
            }
        ],
        model="gpt-4o-mini",
    )

    expected = {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "text", "text": "Part two"},
                ],
            }
        ],
        "model": "gpt-4o-mini",
    }
    assert body.model_dump() == expected
    assert body.model_dump() == expected


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {}}],
        [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.test/image.png", "detail": "original"},
            }
        ],
    ],
    ids=["malformed-image", "unsupported-image-detail"],
)
def test_request_rejects_unsupported_user_content(content):
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            messages=[{"role": "user", "content": content}],
            model="gpt-4o-mini",
        )


def test_non_streaming_chat_completion_uses_sdk_model():
    assert OpenAIChatCompletionsResponse is ChatCompletion
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


def test_streaming_chat_completion_chunk_uses_sdk_model():
    assert OpenAIChatCompletionsStreamChunk is ChatCompletionChunk
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
