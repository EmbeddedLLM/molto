from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from molto.utils.io import dumps_sse
from molto.utils.test import (
    assert_contains_expected,
    reconstructable_anthropic_event,
    reconstructable_chat_completions_chunks,
    reconstructable_responses_event,
)

CHAT_BODY = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "test-model",
    "stream": True,
}


def test_assert_contains_expected_allows_extra_dictionary_fields():
    expected = {
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                }
            ],
        },
    }
    actual = {
        "type": "response.completed",
        "sequence_number": 10,
        "response": {
            "id": "response_123",
            "output": [
                {
                    "id": "message_123",
                    "type": "message",
                    "status": "completed",
                }
            ],
        },
    }

    assert_contains_expected(actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected", "message"),
    [
        ({"response": []}, {"response": {}}, r"Expected an object at \$\.response"),
        ({"response": {}}, {"response": []}, r"Expected a list at \$\.response"),
        (
            {"response": {}},
            {"response": {"id": "response_123"}},
            r"Missing keys at \$\.response: \['id'\]",
        ),
        ({"output": [1]}, {"output": [1, 2]}, r"Expected 2 items at \$\.output, got 1"),
        (
            {"output": [{"type": "actual"}]},
            {"output": [{"type": "expected"}]},
            r"Value mismatch at \$\.output\[0\]\.type: 'actual' != 'expected'",
        ),
    ],
)
def test_assert_contains_expected_reports_mismatch_path(
    actual: Any,
    expected: Any,
    message: str,
):
    with pytest.raises(AssertionError, match=message):
        assert_contains_expected(actual, expected)


def test_assert_contains_expected_uses_custom_root_path():
    with pytest.raises(AssertionError, match=r"Value mismatch at event\[2\]\.type"):
        assert_contains_expected("actual", "expected", path="event[2].type")


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

    different_finish_choice = {**finish_choice, "finish_reason": "length"}
    assert combined != reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[different_finish_choice], usage=usage)),
        CHAT_BODY,
    )


def test_reconstructable_anthropic_event_preserves_stop_reason():
    event = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 1},
    }
    different_reason = deepcopy(event)
    different_reason["delta"]["stop_reason"] = "max_tokens"

    assert reconstructable_anthropic_event(dumps_sse(event)) != (
        reconstructable_anthropic_event(dumps_sse(different_reason))
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
    provider_variant["system_fingerprint"] = "different"

    expected = reconstructable_chat_completions_chunks(dumps_sse(chunk), CHAT_BODY)
    assert expected == reconstructable_chat_completions_chunks(
        dumps_sse(provider_variant), CHAT_BODY
    )

    provider_variant["service_tier"] = "auto"
    assert expected != reconstructable_chat_completions_chunks(
        dumps_sse(provider_variant), CHAT_BODY
    )

    provider_variant["service_tier"] = chunk["service_tier"]
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
