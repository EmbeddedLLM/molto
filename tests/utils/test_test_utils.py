from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from molto.utils.io import dumps_sse, read_yaml
from molto.utils.test import (
    assert_contains_expected,
    filter_headers,
    mask_authorization,
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
    """Ensure generated metadata and absent/null incomplete details do not break parity."""
    expected_events = _responses_events()
    expected_events[0]["item"]["phase"] = "final_answer"
    expected_events[1]["response"]["incomplete_details"] = None
    expected_events[1]["response"]["output"][0]["phase"] = "final_answer"
    actual_events = deepcopy(expected_events)
    actual_events[0]["item"].pop("phase")
    actual_events[1]["response"].pop("incomplete_details")
    actual_events[1]["response"]["output"][0].pop("phase")
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
    """Ensures malformed requests and incomplete events still raise SDK validation errors."""
    event = _responses_events()[0]

    for invalid in (
        {"type": "invalid"},
        {"type": "response.incomplete"},
        {"type": "response.incomplete", "response": {"usage": {}}},
    ):
        with pytest.raises(ValidationError):
            reconstructable_responses_event(dumps_sse(invalid), {})
    with pytest.raises(ValidationError):
        reconstructable_responses_event(dumps_sse(event), {"invalid": True})


@pytest.fixture
async def incomplete_responses_event() -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse the recorded incomplete stream through the OpenAI SDK independently."""
    cassette = read_yaml(
        Path(__file__).parents[1]
        / "cassettes/errors/incomplete_response/openai-responses-stream-00.yaml"
    )

    def replay(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            cassette["response"]["status_code"],
            headers={"content-type": "text/event-stream"},
            content="".join(cassette["response"]["sse"]),
        )

    body = cassette["request"]["body"]
    async with AsyncOpenAI(
        api_key="test-api-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(replay)),
    ) as sdk:
        events = [event async for event in await sdk.responses.create(**body)]
    assert events[-1].type == "response.incomplete"
    return events[-1].model_dump(mode="json", exclude_unset=True), body


def test_reconstructable_responses_event_preserves_incomplete_result(
    incomplete_responses_event: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Ensure incomplete SDK results retain their recorded status, output, and usage."""
    event, body = incomplete_responses_event
    expected = event["response"]
    actual = reconstructable_responses_event(dumps_sse(event), body)

    assert actual["type"] == "response.incomplete"
    response = actual["response"]
    assert response["status"] == expected["status"] == "incomplete"
    assert (
        response["incomplete_details"]
        == expected["incomplete_details"]
        == {"reason": "max_output_tokens"}
    )
    assert response["output"] == [
        {key: value for key, value in item.items() if key != "id"} for item in expected["output"]
    ]
    assert response["output"][0]["content"][0]["text"]
    assert response["usage"] == expected["usage"]


def test_reconstructable_responses_event_requires_recorded_cache_write_count(
    incomplete_responses_event: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Ensure comparison validation cannot invent a missing incomplete cache-write count."""
    event, body = deepcopy(incomplete_responses_event)
    event["response"]["usage"]["input_tokens_details"].pop("cache_write_tokens", None)
    with pytest.raises(ValidationError, match="cache_write_tokens"):
        reconstructable_responses_event(dumps_sse(event), body)


@pytest.mark.parametrize(
    "mutation", ["incomplete_reason", "status", "partial_text", "usage", "cache_usage"]
)
def test_reconstructable_responses_event_rejects_changed_incomplete_result(
    incomplete_responses_event: tuple[dict[str, Any], dict[str, Any]],
    mutation: str,
) -> None:
    """Ensure an incorrect incomplete reason, partial result, or token count fails parity."""
    event, body = incomplete_responses_event
    expected = reconstructable_responses_event(dumps_sse(event), body)
    changed = deepcopy(event)
    response = changed["response"]
    if mutation == "incomplete_reason":
        response["incomplete_details"]["reason"] = "content_filter"
    elif mutation == "status":
        response["status"] = "completed"
    elif mutation == "partial_text":
        response["output"][0]["content"][0]["text"] += " changed"
    elif mutation == "cache_usage":
        response["usage"]["input_tokens_details"]["cache_write_tokens"] += 1
    else:
        response["usage"]["output_tokens"] += 1

    with pytest.raises(AssertionError, match="Value mismatch"):
        assert_contains_expected(
            reconstructable_responses_event(dumps_sse(changed), body),
            expected,
        )


@pytest.mark.parametrize(
    ("value", "masked"),
    [
        ("Bearer sk-secret", "Bearer ***"),
        ("bearer sk-secret", "Bearer ***"),
        ("Basic dXNlcjpzZWNyZXQ=", "***"),
        ("sk-secret", "***"),
        ("", ""),
    ],
)
def test_mask_authorization_never_retains_credentials(value: str, masked: str):
    """Ensures recorded Authorization values keep at most the scheme, never key material."""
    assert mask_authorization(value) == masked


def test_filter_headers_retains_only_whitelisted_headers_and_masks_authorization():
    """Ensures recordings drop unlisted headers, such as API keys, and mask Authorization."""
    headers = httpx.Headers(
        {
            "Authorization": "Bearer sk-secret",
            "Content-Type": "application/json",
            "X-Run-Id": "run-1",
            "X-API-Key": "sk-ant-secret",
            "OpenAI-Organization": "org-secret",
            "Cookie": "session=secret",
        }
    )

    assert filter_headers(headers) == {
        "authorization": "Bearer ***",
        "content-type": "application/json",
        "x-run-id": "run-1",
    }
