"""Collect public SDK output and compare conversation history without Molto translation."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from anthropic.lib.streaming._messages import accumulate_event
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import BaseModel

if TYPE_CHECKING:
    from tests.core.cassette_helpers import Protocol


def _collect_response(events: list[BaseModel], protocol: Protocol) -> BaseModel:
    """Accumulate downstream output using SDK behavior independently of Molto collectors."""
    if protocol == "chat/completions":
        state = ChatCompletionStreamState()
        for event in events:
            state.handle_chunk(event)
        return state.get_final_completion()
    if protocol == "messages":
        message = None
        for event in events:
            message = accumulate_event(event=event, current_snapshot=message)
        assert message is not None and events[-1].type == "message_stop"
        return message
    terminal = [
        event.response
        for event in events
        if event.type in {"response.completed", "response.incomplete"}
    ]
    assert len(terminal) == 1
    return terminal[0]


def _response_history(response: BaseModel, protocol: Protocol) -> list[dict[str, Any]]:
    """Convert a public response into request history, removing SDK-only bookkeeping."""
    if protocol == "responses":
        return [item.model_dump(exclude_none=True) for item in response.output]
    if protocol == "chat/completions":
        assert len(response.choices) == 1
        message = response.choices[0].message.model_dump(
            exclude_none=True, exclude={"tool_calls": {"__all__": {"index"}}}
        )
        # Chat annotations are output-only; an empty list carries no history state.
        # Keep populated annotations so unsupported history fails instead of losing data.
        if message.get("annotations") == []:
            message.pop("annotations")
        return [message]
    return [
        {
            "role": response.role,
            "content": [
                # The SDK stores its incremental JSON parser buffer on tool blocks.
                block.model_dump(exclude_none=True, exclude={"__json_buf", "parsed_output"})
                for block in response.content
            ],
        }
    ]


def _assert_history_matches(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    protocol: Protocol,
    *,
    translated: bool = False,
) -> None:
    """Compare copies using only protocol representation and unavailable metadata allowances."""
    actual, expected = deepcopy(actual), deepcopy(expected)
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        if protocol == "responses":
            for item in (left, right):
                if item.get("role") == "assistant":
                    item.setdefault("type", "message")
                    if isinstance(item.get("content"), str):
                        item["content"] = [{"type": "output_text", "text": item["content"]}]
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        for field in ("annotations", "logprobs"):
                            if part.get(field) == []:
                                part.pop(field)
                if item.get("type") == "reasoning" and item.get("content") == []:
                    item.pop("content")
            output_items = all(
                item.get("type") in {"reasoning", "function_call"}
                or (item.get("type") == "message" and item.get("role") == "assistant")
                for item in (left, right)
            )
            # Only output-item IDs may be generated at a translated protocol boundary.
            if output_items and (translated or "id" not in left or "id" not in right):
                left.pop("id", None)
                right.pop("id", None)
            # Phase has no Chat/Messages equivalent, but populated values must still agree.
            if output_items and ("phase" not in left or "phase" not in right):
                left.pop("phase", None)
                right.pop("phase", None)
            if "status" not in left or "status" not in right:
                for item in (left, right):
                    if item.get("status") == "completed":
                        item.pop("status")
        elif protocol == "chat/completions":
            for item in (left, right):
                if "reasoning" in item and "reasoning_content" not in item:
                    item["reasoning_content"] = item.pop("reasoning")
                if item.get("tool_calls") and item.get("content") in (None, ""):
                    item.pop("content", None)
                if item.get("tool_calls") == []:
                    item.pop("tool_calls")
        else:
            for item in (left, right):
                if isinstance(item.get("content"), str):
                    item["content"] = [{"type": "text", "text": item["content"]}]
                for block in item.get("content") or []:
                    if block.get("type") == "tool_use" and block.get("caller") == {
                        "type": "direct"
                    }:
                        block.pop("caller")
                    if block.get("type") == "thinking" and block.get("signature") == "":
                        block.pop("signature")
        assert left == right


def _continue_history(
    history: list[dict[str, Any]],
    output_history: list[dict[str, Any]],
    next_recorded_history: list[dict[str, Any]],
    protocol: Protocol,
    *,
    translated: bool = False,
) -> list[dict[str, Any]]:
    """Retain actual output and append only new caller inputs after checking the next YAML."""
    retained = [*history, *output_history]
    _assert_history_matches(
        retained,
        next_recorded_history[: len(retained)],
        protocol,
        translated=translated,
    )
    continuation = deepcopy(next_recorded_history[len(retained) :])
    assert continuation, "The next conversation turn must contain new caller input."
    if protocol == "responses":
        assert all(
            item.get("type") == "function_call_output" or item.get("role") == "user"
            for item in continuation
        )
        calls = [item["call_id"] for item in output_history if item["type"] == "function_call"]
        results = [
            item["call_id"] for item in continuation if item.get("type") == "function_call_output"
        ]
    elif protocol == "chat/completions":
        assert all(item["role"] in {"tool", "user"} for item in continuation)
        calls = [call["id"] for item in output_history for call in item.get("tool_calls", [])]
        results = [item["tool_call_id"] for item in continuation if item["role"] == "tool"]
    else:
        assert all(item["role"] == "user" for item in continuation)
        calls = [
            block["id"]
            for item in output_history
            for block in item["content"]
            if block["type"] == "tool_use"
        ]
        results = [
            block["tool_use_id"]
            for item in continuation
            for block in item["content"]
            if isinstance(block, dict) and block["type"] == "tool_result"
        ]
    assert results == calls
    return [*retained, *continuation]
