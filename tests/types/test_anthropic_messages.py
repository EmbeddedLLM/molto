import inspect

import pytest
from anthropic.resources.messages.messages import AsyncMessages
from anthropic.types import Message, RawMessageStreamEvent
from pydantic import ValidationError

from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesResponse,
    AnthropicMessagesStreamEvent,
    AnthropicMessagesStreamEventAdapter,
)


def test_request_fields_match_messages_create_body_parameters():
    signature = inspect.signature(AsyncMessages.create)
    transport_parameters = {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
    sdk_parameters = {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in transport_parameters
    }

    assert set(AnthropicMessagesCreateRequest.model_fields) == sdk_parameters


def test_request_dump_accepts_compatible_thinking_history_without_signature():
    body = {
        "max_tokens": 2048,
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "2+2=4"},
                    {"type": "text", "text": "4"},
                ],
            },
        ],
        "model": "reasoning-model",
        "stream": True,
        "thinking": {"type": "adaptive", "display": "summarized"},
    }

    assert AnthropicMessagesCreateRequest.model_validate(body).model_dump() == body


def test_result_types_use_sdk_models():
    assert AnthropicMessagesResponse is Message
    assert AnthropicMessagesStreamEvent is RawMessageStreamEvent


def test_stream_adapter_uses_official_response_service_tiers():
    event = {
        "type": "message_start",
        "message": {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "model",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 0,
                "service_tier": "standard",
            },
        },
    }

    parsed = AnthropicMessagesStreamEventAdapter.validate_python(event)
    assert parsed.message.usage.service_tier == "standard"

    event["message"]["usage"]["service_tier"] = "auto"
    with pytest.raises(ValidationError):
        AnthropicMessagesStreamEventAdapter.validate_python(event)
