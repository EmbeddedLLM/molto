from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from molto.utils.io import dumps_sse
from molto.utils.test import (
    reconstructable_chat_completions_chunks,
    reconstructable_responses_event,
)

CHAT_BODY = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "test-model",
    "stream": True,
}


def _chat_chunk(*, choices: list[dict[str, Any]], usage: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "id": "chatcmpl_123",
        "object": "chat.completion.chunk",
        "created": 123,
        "model": "test-model",
        "service_tier": "default",
        "system_fingerprint": "provider-only",
        "choices": choices,
        "usage": usage,
    }


def test_reconstructable_chat_completions_chunks_normalize_terminal_usage():
    finish_choice = {
        "index": 0,
        "delta": {"content": "", "reasoning_content": None},
        "logprobs": None,
        "finish_reason": "stop",
    }
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 3, "audio_tokens": 0},
        "prompt_cache_hit_tokens": 0,
    }
    combined = reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[finish_choice], usage=usage)),
        CHAT_BODY,
    )
    separate = [
        *reconstructable_chat_completions_chunks(
            dumps_sse(_chat_chunk(choices=[finish_choice], usage=None)),
            CHAT_BODY,
        ),
        *reconstructable_chat_completions_chunks(
            dumps_sse(_chat_chunk(choices=[], usage=usage)),
            CHAT_BODY,
        ),
    ]

    assert combined == separate

    different_usage = deepcopy(usage)
    different_usage["completion_tokens"] = 6
    different_usage["total_tokens"] = 16
    assert combined != reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[finish_choice], usage=different_usage)),
        CHAT_BODY,
    )


def test_reconstructable_chat_completions_chunks_preserve_each_delta():
    chunk = _chat_chunk(
        choices=[
            {
                "index": 0,
                "delta": {"content": "Hello"},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
        usage=None,
    )
    provider_variant = deepcopy(chunk)
    provider_variant["service_tier"] = "auto"
    provider_variant["system_fingerprint"] = "different"

    expected = reconstructable_chat_completions_chunks(dumps_sse(chunk), CHAT_BODY)
    assert expected == reconstructable_chat_completions_chunks(
        dumps_sse(provider_variant), CHAT_BODY
    )

    provider_variant["choices"][0]["delta"]["content"] = "Different"
    assert expected != reconstructable_chat_completions_chunks(
        dumps_sse(provider_variant), CHAT_BODY
    )


def test_reconstructable_chat_completions_chunks_validate_chunk_and_request():
    chunk = _chat_chunk(choices=[], usage=None)

    with pytest.raises(ValidationError):
        reconstructable_chat_completions_chunks(dumps_sse({}), CHAT_BODY)
    with pytest.raises(ValidationError):
        reconstructable_chat_completions_chunks(dumps_sse(chunk), {})


def _responses_events() -> list[dict[str, Any]]:
    output_item = {
        "id": "msg_expected",
        "type": "message",
        "status": "completed",
        "content": [],
        "role": "assistant",
    }
    return [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**output_item, "status": "in_progress"},
            "sequence_number": 0,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_expected",
                "object": "response",
                "created_at": 123,
                "status": "completed",
                "background": False,
                "completed_at": 123,
                "model": "expected-model",
                "output": [output_item],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": None,
            },
            "sequence_number": 1,
        },
    ]


def _reconstructable(events: list[dict[str, Any]], body: dict[str, Any]) -> list[dict[str, Any]]:
    return [reconstructable_responses_event(dumps_sse(event), body) for event in events]


def test_reconstructable_responses_event_preserves_reconstructable_fields():
    expected_events = _responses_events()
    actual_events = deepcopy(expected_events)
    actual_events[0]["item"]["id"] = "msg_generated"
    actual_events[1]["response"].pop("background")
    actual_events[1]["response"]["completed_at"] = 456
    actual_events[1]["response"]["output"][0]["id"] = "msg_generated"

    assert _reconstructable(actual_events, {}) == _reconstructable(expected_events, {})
    assert _reconstructable(actual_events, {"background": False}) != _reconstructable(
        expected_events, {"background": False}
    )

    actual_events[1]["response"]["model"] = "different-model"
    assert _reconstructable(actual_events, {}) != _reconstructable(expected_events, {})


def test_reconstructable_responses_event_validates_event_and_request():
    event = _responses_events()[0]

    with pytest.raises(ValidationError):
        reconstructable_responses_event(dumps_sse({"type": "invalid"}), {})
    with pytest.raises(ValidationError):
        reconstructable_responses_event(dumps_sse(event), {"invalid": True})
