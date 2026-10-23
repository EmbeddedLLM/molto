from copy import deepcopy
from typing import Any

import pytest
from openai.types.chat import ChatCompletion

from tests.core.cassette_helpers import PROTOCOLS, Protocol
from tests.core.history_helpers import (
    _assert_history_matches,
    _continue_history,
    _response_history,
)


def test_chat_empty_calls_match_omission_without_mutating_history():
    """Ensures Together's empty call list stays in submitted history while matching omitted calls."""
    actual = [{"role": "assistant", "content": "Hello", "tool_calls": []}]
    expected = [{"role": "assistant", "content": "Hello"}]
    _assert_history_matches(actual, expected, "chat/completions")
    assert actual == [{**expected[0], "tool_calls": []}]


def _assistant_history(protocol: Protocol) -> list[dict[str, Any]]:
    if protocol == "responses":
        return [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Thinking."}],
                "summary": [],
                "encrypted_content": "opaque-signature",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "First."},
                    {"type": "output_text", "text": "Second."},
                ],
            },
            *[
                {"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"}
                for call_id in ("call_a", "call_b")
            ],
        ]
    if protocol == "chat/completions":
        return [
            {
                "role": "assistant",
                "reasoning_content": "Thinking.",
                "content": "First. Second.",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": call_id,
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                    for call_id in ("call_a", "call_b")
                ],
            }
        ]
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Thinking.", "signature": "opaque-signature"},
                {"type": "text", "text": "First."},
                {"type": "text", "text": "Second."},
                *[
                    {"type": "tool_use", "id": call_id, "name": "lookup", "input": {}}
                    for call_id in ("call_a", "call_b")
                ],
            ],
        }
    ]


@pytest.mark.parametrize(
    ("protocol", "path", "replacement"),
    [
        ("responses", (0, "content", 0, "text"), "Changed thinking."),
        ("responses", (0, "encrypted_content"), "changed-signature"),
        ("responses", (1, "content", 0, "text"), "Changed text."),
        ("responses", (1, "content", 0, "text"), "First. "),
        ("responses", (1, "content"), [{"type": "output_text", "text": "First.Second."}]),
        ("responses", (2, "call_id"), "different-call"),
        ("responses", (2, "name"), "different-tool"),
        ("responses", (2, "arguments"), '{"changed":true}'),
        ("chat/completions", (0, "reasoning_content"), "Changed thinking."),
        ("chat/completions", (0, "content"), "Changed text."),
        ("chat/completions", (0, "content"), "First. Second. "),
        ("chat/completions", (0, "tool_calls"), []),
        ("chat/completions", (0, "tool_calls", 0, "id"), "different-call"),
        ("chat/completions", (0, "tool_calls", 0, "function", "name"), "different-tool"),
        ("chat/completions", (0, "tool_calls", 0, "function", "arguments"), '{"changed":true}'),
        ("messages", (0, "content", 0, "thinking"), "Changed thinking."),
        ("messages", (0, "content", 0, "signature"), "changed-signature"),
        ("messages", (0, "content", 1, "text"), "Changed text."),
        ("messages", (0, "content", 1, "text"), "First. "),
        ("messages", (0, "content", 3, "id"), "different-call"),
        ("messages", (0, "content", 3, "name"), "different-tool"),
        ("messages", (0, "content", 3, "input"), {"changed": True}),
    ],
)
def test_history_comparison_rejects_changed_semantics(
    protocol: Protocol, path: tuple[str | int, ...], replacement: Any
):
    """Ensures metadata allowances never hide text, reasoning, calls, signatures, or part loss."""
    expected = _assistant_history(protocol)
    actual = deepcopy(expected)
    target = actual
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = replacement
    with pytest.raises(AssertionError):
        _assert_history_matches(actual, expected, protocol, translated=True)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_history_comparison_preserves_call_order(protocol: Protocol):
    """Ensures parallel calls cannot change order while matching next-turn history."""
    expected = _assistant_history(protocol)
    actual = deepcopy(expected)
    if protocol == "responses":
        actual[2:] = reversed(actual[2:])
    elif protocol == "chat/completions":
        actual[0]["tool_calls"].reverse()
    else:
        actual[0]["content"][3:] = reversed(actual[0]["content"][3:])
    with pytest.raises(AssertionError):
        _assert_history_matches(actual, expected, protocol, translated=True)


@pytest.mark.parametrize(
    ("role", "field", "translated"),
    [("user", "id", True), ("assistant", "id", False), ("assistant", "phase", True)],
)
def test_history_comparison_preserves_available_metadata(role: str, field: str, translated: bool):
    """Ensures only unavailable output IDs and omitted phase receive metadata allowances."""
    expected = [{"type": "message", "role": role, "content": [], field: "recorded"}]
    actual = [{**expected[0], field: "changed"}]
    with pytest.raises(AssertionError):
        _assert_history_matches(actual, expected, "responses", translated=translated)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("corruption", ["wrong_id", "missing_result", "reordered", "assistant"])
def test_continuation_rejects_invalid_new_input(protocol: Protocol, corruption: str):
    """Ensures continuation cannot replace actual output or lose ordered tool-result linkage."""
    output = _assistant_history(protocol)
    if protocol == "responses":
        results = [
            {"type": "function_call_output", "call_id": call_id, "output": "done"}
            for call_id in ("call_a", "call_b")
        ]
        id_field = "call_id"
    elif protocol == "chat/completions":
        results = [
            {"role": "tool", "tool_call_id": call_id, "content": "done"}
            for call_id in ("call_a", "call_b")
        ]
        id_field = "tool_call_id"
    else:
        results = [
            {"type": "tool_result", "tool_use_id": call_id, "content": "done"}
            for call_id in ("call_a", "call_b")
        ]
        id_field = "tool_use_id"
    if corruption == "wrong_id":
        results[0][id_field] = "different-call"
    elif corruption == "missing_result":
        results.pop()
    elif corruption == "reordered":
        results.reverse()
    continuation = [{"role": "user", "content": results}] if protocol == "messages" else results
    if corruption == "assistant":
        continuation.append({"role": "assistant", "content": "Substituted output."})
    with pytest.raises(AssertionError):
        _continue_history([], output, [*deepcopy(output), *continuation], protocol)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_history_comparison_keeps_actual_output_unchanged(protocol: Protocol):
    """Ensures allowed comparison differences never rewrite the actual submitted history."""
    if protocol == "responses":
        actual = [{"type": "message", "role": "assistant", "content": "First.", "id": "generated"}]
        expected = [{"role": "assistant", "content": "First.", "id": "recorded"}]
    elif protocol == "chat/completions":
        actual = [{"role": "assistant", "reasoning_content": "Thinking."}]
        expected = [{"role": "assistant", "reasoning": "Thinking."}]
    else:
        actual = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Thinking.", "signature": ""}],
            }
        ]
        expected = [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Thinking."}]}
        ]
    original_actual, original_expected = deepcopy(actual), deepcopy(expected)
    _assert_history_matches(actual, expected, protocol, translated=True)
    assert actual == original_actual
    assert expected == original_expected


@pytest.mark.parametrize("populated", [False, True])
def test_chat_history_omits_only_empty_output_annotations(populated: bool):
    """Ensures empty SDK annotations are omitted while populated output cannot silently disappear."""
    annotations = (
        [
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": 0,
                    "end_index": 1,
                    "title": "Source",
                    "url": "https://example.org",
                },
            }
        ]
        if populated
        else []
    )
    response = ChatCompletion.model_validate(
        {
            "id": "completion",
            "object": "chat.completion",
            "model": "test-model",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Text.",
                        "annotations": annotations,
                    },
                }
            ],
        }
    )
    history = _response_history(response, "chat/completions")
    expected = {"role": "assistant", "content": "Text."}
    if populated:
        expected["annotations"] = annotations
    assert history == [expected]
    assert response.choices[0].message.model_dump()["annotations"] == annotations
