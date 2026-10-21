from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from molto.utils.io import dumps_sse
from molto.utils.test import (
    assert_chat_choice_matches,
    assert_chat_completions_chunks_match,
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


@pytest.mark.parametrize("text", ["A thought", "\n\n"])
def test_reconstructable_chat_completions_chunks_compare_provider_reasoning_aliases(text):
    """Ensures provider reasoning aliases compare equal without hiding different text."""
    choice = {"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}
    expected = reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
    )
    choice["delta"] = {"reasoning": text}
    assert expected == reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
    )
    choice["delta"] = {"reasoning": "Different thought"}
    assert expected != reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
    )


@pytest.mark.parametrize(
    ("reasoning", "reasoning_content"),
    [("Together", "OpenAI"), ("Thought", "Thought")],
)
def test_reconstructable_chat_completions_chunks_preserve_both_reasoning_fields(
    reasoning, reasoning_content
):
    """Ensures simultaneous reasoning fields retain both values during comparison."""
    delta = {"reasoning": reasoning, "reasoning_content": reasoning_content}
    choice = {"index": 0, "delta": delta, "finish_reason": None}
    result = reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
    )

    assert result[0]["choices"][0]["delta"] == delta


@pytest.mark.parametrize("text", ["Hello", "\n\n", ""])
def test_reconstructable_chat_completions_chunks_preserve_duplicate_text(text):
    """Ensures provider text remains available for independent consistency assertions."""
    choice = {"index": 0, "delta": {"content": text}, "text": text, "finish_reason": None}
    result = reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
    )

    assert result[0]["choices"][0]["text"] == text


@pytest.mark.parametrize(
    "delta", [{"content": "Hello"}, {"content": "\n\n"}, {"content": None}, {}]
)
@pytest.mark.parametrize("text_sides", ["actual", "expected", "both"])
def test_assert_chat_choice_matches_allows_consistent_duplicate_text(delta, text_sides):
    """Ensures redundant text on either side is accepted without changing either choice."""
    actual = {"index": 0, "delta": delta, "finish_reason": None}
    expected = deepcopy(actual)
    for side, choice in (("actual", actual), ("expected", expected)):
        if text_sides in (side, "both"):
            choice["text"] = delta.get("content") or ""
    original = deepcopy((actual, expected))

    assert_chat_choice_matches(actual, expected)

    assert (actual, expected) == original


@pytest.mark.parametrize("text", ["Wrong", "Hello ", None])
@pytest.mark.parametrize("text_sides", ["actual", "expected", "both"])
def test_assert_chat_choice_matches_rejects_contradictory_text(text, text_sides):
    """Ensures even identical duplicated text cannot contradict the delta on either side."""
    actual = {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
    expected = deepcopy(actual)
    for side, choice in (("actual", actual), ("expected", expected)):
        if text_sides in (side, "both"):
            choice["text"] = text
    original = deepcopy((actual, expected))

    with pytest.raises(AssertionError):
        assert_chat_choice_matches(actual, expected)

    assert (actual, expected) == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta", {}),
        ("delta", {"content": "Different"}),
        ("delta", None),
        ("index", 1),
        ("finish_reason", "length"),
        ("provider_extension", "unexpected"),
        ("finish_reason", None),
    ],
)
@pytest.mark.parametrize("swap", [False, True])
def test_assert_chat_choice_matches_rejects_other_differences(field, value, swap):
    """Ensures non-text fields must be present and equal in both choices."""
    expected = {"index": 0, "delta": {"content": "Hello"}, "finish_reason": "stop"}
    actual = {**expected, field: value}
    if value is None:
        del actual[field]
    if swap:
        actual, expected = expected, actual

    with pytest.raises(AssertionError):
        assert_chat_choice_matches(actual, expected)


@pytest.mark.parametrize("both", [False, True])
def test_assert_chat_completions_chunks_match_rejects_normalized_contradictory_text(both):
    """Ensures reconstruction cannot hide contradictory text before parity comparison."""
    choice = {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
    actual = reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[{**choice, "text": "Wrong"}], usage=None)), CHAT_BODY
    )
    if both:
        choice["text"] = "Wrong"
    expected = reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
    )

    with pytest.raises(AssertionError):
        assert_chat_completions_chunks_match(actual, expected)


def test_assert_chat_completions_chunks_match_preserves_inputs():
    """Ensures chunk comparison accepts redundant text without mutating nested payloads."""
    choices = [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]
    actual = [_chat_chunk(choices=choices, usage=None), _chat_chunk(choices=[], usage=None)]
    expected = deepcopy(actual)
    expected[0]["choices"][0]["text"] = "Hello"
    original = deepcopy((actual, expected))

    assert_chat_completions_chunks_match(actual, expected)

    assert (actual, expected) == original


@pytest.mark.parametrize(
    "difference",
    ["metadata", "extra_metadata", "chunk_count", "chunk_order", "choice_count", "choice_order"],
)
def test_assert_chat_completions_chunks_match_rejects_other_differences(difference):
    """Ensures chunk metadata, counts, and chunk and choice ordering remain exact."""
    choices = [
        {"index": index, "delta": {"content": text}, "finish_reason": None}
        for index, text in enumerate(["Hello", "World"])
    ]
    expected = [_chat_chunk(choices=choices, usage=None), _chat_chunk(choices=[], usage=None)]
    actual = deepcopy(expected)
    if difference == "metadata":
        actual[0]["model"] = "different-model"
    elif difference == "extra_metadata":
        actual[0]["provider_extension"] = "unexpected"
    elif difference == "chunk_count":
        actual.pop()
    elif difference == "chunk_order":
        actual.reverse()
    elif difference == "choice_count":
        actual[0]["choices"].pop()
    else:
        actual[0]["choices"].reverse()

    with pytest.raises(AssertionError):
        assert_chat_completions_chunks_match(actual, expected)


@pytest.mark.parametrize("placeholder", [None, ""])
@pytest.mark.parametrize(
    "delta",
    [
        {"role": "assistant"},
        {"content": "Hello"},
        {"reasoning_content": "A thought"},
        {"reasoning": "A thought"},
        {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]},
    ],
)
def test_reconstructable_chat_completions_chunks_normalize_optional_placeholders(
    delta, placeholder
):
    """Ensures empty optional fields compare equally without changing populated deltas."""
    delta = deepcopy(delta)
    choice = {"index": 0, "delta": delta, "finish_reason": None}
    chunk = _chat_chunk(choices=[choice], usage=None)
    expected = reconstructable_chat_completions_chunks(dumps_sse(chunk), CHAT_BODY)

    chunk.pop("usage")
    for field in ("content", "reasoning", "reasoning_content", "refusal", "token_id"):
        delta.setdefault(field, placeholder)
    assert expected == reconstructable_chat_completions_chunks(dumps_sse(chunk), CHAT_BODY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", "\n\n"),
        ("reasoning", "\n\n"),
        ("reasoning_content", "\n\n"),
        ("refusal", "Cannot comply"),
        ("token_id", 42),
        ("provider_extension", "unrecognized"),
    ],
)
def test_reconstructable_chat_completions_chunks_preserve_populated_fields(field, value):
    """Ensures placeholder normalization cannot conceal text or populated extensions."""
    choice = {"index": 0, "delta": {}, "finish_reason": None}
    chunk = _chat_chunk(choices=[choice], usage=None)
    expected = reconstructable_chat_completions_chunks(dumps_sse(chunk), CHAT_BODY)

    choice["delta"][field] = value
    assert expected != reconstructable_chat_completions_chunks(dumps_sse(chunk), CHAT_BODY)


def test_reconstructable_chat_completions_chunks_normalize_tool_continuation_placeholders():
    """Ensures optional repeated tool fields normalize without hiding identity or arguments."""
    tool = {"index": 0, "function": {"arguments": '{"path":'}}
    choice = {"index": 0, "delta": {"tool_calls": [tool]}, "finish_reason": None}
    chunk = _chat_chunk(choices=[choice], usage=None)
    expected = reconstructable_chat_completions_chunks(dumps_sse(chunk), CHAT_BODY)

    tool.update(id=None, type="function")
    tool["function"]["name"] = None
    assert expected == reconstructable_chat_completions_chunks(dumps_sse(chunk), CHAT_BODY)

    for field, value in (("id", "new-call"), ("index", 1)):
        changed = deepcopy(chunk)
        changed["choices"][0]["delta"]["tool_calls"][0][field] = value
        assert expected != reconstructable_chat_completions_chunks(dumps_sse(changed), CHAT_BODY)
    for field, value in (("name", "another_function"), ("arguments", '"different"}')):
        changed = deepcopy(chunk)
        changed["choices"][0]["delta"]["tool_calls"][0]["function"][field] = value
        assert expected != reconstructable_chat_completions_chunks(dumps_sse(changed), CHAT_BODY)


@pytest.mark.parametrize("field", ["content", "reasoning", "tool_calls"])
def test_reconstructable_chat_completions_chunks_preserve_delta_boundaries(field):
    """Ensures equal final content cannot hide different observed text or tool chunks."""

    def normalize(parts):
        chunks = []
        for part in parts:
            value = (
                [{"index": 0, "function": {"arguments": part}}] if field == "tool_calls" else part
            )
            choice = {"index": 0, "delta": {field: value}, "finish_reason": None}
            chunks.extend(
                reconstructable_chat_completions_chunks(
                    dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
                )
            )
        return chunks

    assert normalize(["Hello", " world"]) != normalize(["Hello world"])
    assert normalize(["Hello", " world"]) != normalize(["Hell", "o world"])


@pytest.mark.parametrize(
    "fields", [("reasoning",), ("reasoning_content",), ("reasoning", "reasoning_content")]
)
@pytest.mark.parametrize("placeholder", [None, ""])
def test_reconstructable_chat_completions_chunks_prune_terminal_reasoning_placeholders(
    fields, placeholder
):
    """Ensures terminal empty or null reasoning aliases compare like absent fields."""
    choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
    expected = reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
    )
    choice["delta"] = dict.fromkeys(fields, placeholder)
    assert expected == reconstructable_chat_completions_chunks(
        dumps_sse(_chat_chunk(choices=[choice], usage=None)), CHAT_BODY
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
    expected_events[0]["item"]["phase"] = "final_answer"
    expected_events[1]["response"]["output"][0]["phase"] = "final_answer"
    actual_events = deepcopy(expected_events)
    actual_events[0]["item"].pop("phase")
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
    event = _responses_events()[0]

    with pytest.raises(ValidationError):
        reconstructable_responses_event(dumps_sse({"type": "invalid"}), {})
    with pytest.raises(ValidationError):
        reconstructable_responses_event(dumps_sse(event), {"invalid": True})
