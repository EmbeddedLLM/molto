#!/usr/bin/env python3
"""Prepare a Chat Completions cassette for CC-to-Responses stream parity tests.

The Responses cassette is the canonical expected output. This script rewrites the
Chat Completions SSE stream so a converter receives one corresponding CC delta for
every `response.output_text.delta` and `response.reasoning_text.delta` event,
and one corresponding tool-call argument delta for every
`response.function_call_arguments.delta` event, in the same order and with the
same content. It also synchronizes fields that can pass through the conversion,
including the response ID, creation time, model, target-default service tier, tool
calls, and usage. Cache requests also copy the canonical cache options, keys,
retention, and content-part breakpoints without changing their prefix boundaries.

Matched reasoning requests carry the canonical Responses effort as
`reasoning_effort`. When a continuation contains reasoning history but omits a
current reasoning configuration, the matcher uses `high` to match the default
used by the normalized Anthropic thinking policy.

The Responses terminal lifecycle controls the matched Chat finish reason:
`response.completed` maps to `stop`, while incomplete output-token and
content-filter limits map to `length` and `content_filter` respectively.

The result remains a Chat Completions wire-format cassette. Responses-only envelope
fields such as message item IDs, sequence numbers, and a distinct completion time
cannot be encoded in CC chunks; the converter must generate those fields
deterministically for exact output parity. Canonical Responses raw reasoning and
tool-call history are translated into Chat Completions messages. Other request
fields and transport metadata are intentionally left unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from molto.utils.io import dump_yaml, json_dumps, json_loads, read_yaml


def _request_body(cassette: dict[str, Any]) -> dict[str, Any]:
    try:
        request_body = cassette["request"]["body"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cassette must contain request.body.") from exc
    if not isinstance(request_body, dict):
        raise ValueError("Cassette request.body must be an object.")
    return request_body


def _response_message_content(
    content: Any, *, preserve_parts: bool = False
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        if preserve_parts:
            return [{"text": content, "type": "text"}]
        return content
    if not isinstance(content, list):
        raise ValueError("Responses message content must be text or a list of text parts.")

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {
            "input_text",
            "output_text",
            "text",
        }:
            raise ValueError("Only Responses text message content can map to Chat Completions.")
        part_text = part.get("text")
        if not isinstance(part_text, str):
            raise ValueError("Responses text content must contain a string text field.")
        mapped_part = {"text": part_text, "type": "text"}
        if part.get("prompt_cache_breakpoint") is not None:
            mapped_part["prompt_cache_breakpoint"] = deepcopy(part["prompt_cache_breakpoint"])
            preserve_parts = True
        parts.append(mapped_part)
    if preserve_parts:
        return parts
    return "".join(part["text"] for part in parts)


def _raw_reasoning_content(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        raise ValueError("Responses reasoning item must contain a content list.")
    if item.get("summary") not in (None, []):
        raise ValueError("Responses reasoning summaries cannot map to Chat Completions.")
    if item.get("encrypted_content") is not None:
        raise ValueError("Encrypted Responses reasoning cannot map to Chat Completions.")

    reasoning: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "reasoning_text":
            raise ValueError("Only raw Responses reasoning can map to Chat Completions.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Responses reasoning text must be a string.")
        reasoning.append(text)
    return "".join(reasoning)


def _raw_reasoning_content(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        raise ValueError("Responses reasoning item must contain a content list.")
    if item.get("summary") not in (None, []):
        raise ValueError("Responses reasoning summaries cannot map to Chat Completions.")
    if item.get("encrypted_content") is not None:
        raise ValueError("Encrypted Responses reasoning cannot map to Chat Completions.")

    reasoning: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "reasoning_text":
            raise ValueError("Only raw Responses reasoning can map to Chat Completions.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Responses reasoning text must be a string.")
        reasoning.append(text)
    return "".join(reasoning)


def _chat_messages_from_response_input(response_input: Any) -> list[dict[str, Any]]:
    if isinstance(response_input, str):
        return [{"content": response_input, "role": "user"}]
    if not isinstance(response_input, list):
        raise ValueError("Responses input must be a string or an input-item list.")

    messages: list[dict[str, Any]] = []
    tool_call_ids: set[str] = set()
    returned_tool_call_ids: set[str] = set()
    for item in response_input:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects.")

        item_type = item.get("type", "message")
        if item_type == "reasoning":
            messages.append(
                {
                    "reasoning_content": _raw_reasoning_content(item),
                    "role": "assistant",
                }
            )
            continue

        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not all(isinstance(value, str) for value in (call_id, name, arguments)):
                raise ValueError(
                    "Responses function calls require string call_id, name, and arguments."
                )
            if call_id in tool_call_ids:
                raise ValueError(f"Duplicate Responses function call ID: {call_id!r}.")
            if not messages or messages[-1].get("role") != "assistant":
                messages.append({"role": "assistant"})
            messages[-1].setdefault("tool_calls", []).append(
                {
                    "function": {"arguments": arguments, "name": name},
                    "id": call_id,
                    "type": "function",
                }
            )
            tool_call_ids.add(call_id)
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            if not isinstance(call_id, str):
                raise ValueError("Responses function call outputs require a string call_id.")
            if call_id not in tool_call_ids:
                raise ValueError(f"Responses function call output has unknown ID: {call_id!r}.")
            if call_id in returned_tool_call_ids:
                raise ValueError(f"Duplicate Responses function call output ID: {call_id!r}.")
            messages.append(
                {
                    "content": _response_message_content(output, preserve_parts=True),
                    "role": "tool",
                    "tool_call_id": call_id,
                }
            )
            returned_tool_call_ids.add(call_id)
            continue

        if item_type != "message":
            raise ValueError(f"Responses input type {item_type!r} cannot map to Chat Completions.")
        role = item.get("role")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"Responses input role {role!r} cannot map to Chat Completions.")
        message = {
            "content": _response_message_content(
                item.get("content"), preserve_parts=role in {"system", "developer", "user"}
            ),
            "role": role,
        }
        if (
            role == "assistant"
            and messages
            and messages[-1].get("role") == "assistant"
            and ("content" not in messages[-1])
        ):
            messages[-1].update(message)
        else:
            messages.append(message)
    return messages


def _match_request_history(
    chat_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    chat_request = _request_body(chat_cassette)
    responses_request = _request_body(responses_cassette)
    response_input = responses_request.get("input")
    has_cache = any(
        field in responses_request
        for field in ("prompt_cache_options", "prompt_cache_key", "prompt_cache_retention")
    ) or _has_cache_breakpoint(response_input)
    if not has_cache and (
        not isinstance(response_input, list)
        or not any(
            isinstance(item, dict)
            and item.get("type") in {"reasoning", "function_call", "function_call_output"}
            for item in response_input
        )
    ):
        return

    chat_request["messages"] = _chat_messages_from_response_input(response_input)
    if has_cache:
        chat_request["model"] = responses_request["model"]
        for field in ("prompt_cache_options", "prompt_cache_key", "prompt_cache_retention"):
            if field in responses_request:
                chat_request[field] = deepcopy(responses_request[field])
            else:
                chat_request.pop(field, None)


def _has_cache_breakpoint(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("prompt_cache_breakpoint") is not None or any(
            _has_cache_breakpoint(item) for item in value.values()
        )
    return isinstance(value, list) and any(_has_cache_breakpoint(item) for item in value)


def _parse_data_event(sse_entry: str) -> dict[str, Any] | None:
    entry = sse_entry.strip()
    if not entry.startswith("data:"):
        return None

    data = entry.removeprefix("data:").lstrip()
    if data == "[DONE]":
        return None

    event = json_loads(data)
    if not isinstance(event, dict):
        raise ValueError("SSE data must contain a JSON object.")
    return event


def _data_events(cassette: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    try:
        sse_entries = cassette["response"]["sse"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cassette must contain response.sse.") from exc

    if not isinstance(sse_entries, list) or not all(
        isinstance(entry, str) for entry in sse_entries
    ):
        raise ValueError("Cassette response.sse must be a list of strings.")

    return [
        (index, event)
        for index, entry in enumerate(sse_entries)
        if (event := _parse_data_event(entry)) is not None
    ]


def _response_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("type") == event_type]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {event_type!r} event, found {len(matches)}.")
    return matches[0]


def _response_terminal(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    terminal_events = [
        event
        for event in events
        if event.get("type") in {"response.completed", "response.incomplete"}
    ]
    if len(terminal_events) != 1:
        raise ValueError(
            "Expected exactly one completed or incomplete Responses event, "
            f"found {len(terminal_events)}."
        )

    terminal_event = terminal_events[0]
    response = terminal_event.get("response")
    if not isinstance(response, dict):
        raise ValueError("Responses terminal event must contain a response object.")
    if terminal_event["type"] == "response.completed":
        return response, "stop"

    incomplete_details = response.get("incomplete_details")
    if not isinstance(incomplete_details, dict):
        raise ValueError("response.incomplete must contain incomplete_details.")
    reason = incomplete_details.get("reason")
    finish_reasons = {
        "content_filter": "content_filter",
        "max_output_tokens": "length",
    }
    return response, finish_reasons.get(reason, "stop")


def _match_request_reasoning_effort(
    chat_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    chat_request = chat_cassette["request"]["body"]
    responses_request = responses_cassette["request"]["body"]
    reasoning = responses_request.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    response_input = responses_request.get("input")
    has_reasoning_history = isinstance(response_input, list) and any(
        isinstance(item, dict) and item.get("type") == "reasoning" for item in response_input
    )
    if effort is not None or has_reasoning_history:
        chat_request["reasoning_effort"] = effort or "high"
    else:
        chat_request.pop("reasoning_effort", None)


def _serialize_data_event(event: dict[str, Any], template: str) -> str:
    if _parse_data_event(template) == event:
        return template
    suffix = template[len(template.rstrip("\r\n")) :]
    return f"data: {json_dumps(event)}{suffix}"


def _response_text_deltas(
    events: list[dict[str, Any]], event_prefix: str, *, optional: bool = False
) -> list[str]:
    delta_type = f"{event_prefix}.delta"
    done_type = f"{event_prefix}.done"
    deltas = [event["delta"] for event in events if event.get("type") == delta_type]
    if not all(isinstance(delta, str) for delta in deltas):
        raise ValueError(f"{event_prefix} deltas must be strings.")

    done_events = [event for event in events if event.get("type") == done_type]
    if optional and not deltas and not done_events:
        return []
    if len(done_events) != 1:
        raise ValueError(f"Expected exactly one {done_type!r} event, found {len(done_events)}.")

    done_text = done_events[0].get("text")
    if "".join(deltas) != done_text:
        raise ValueError(f"{event_prefix} deltas do not match {event_prefix}.done.")
    return deltas


def _response_tool_calls(events: list[dict[str, Any]]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    calls_by_item_id: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []

    for event in events:
        event_type = event.get("type")
        item = event.get("item")
        if (
            event_type == "response.output_item.added"
            and isinstance(item, dict)
            and item.get("type") == "function_call"
        ):
            for field in ("id", "call_id", "name", "arguments"):
                if not isinstance(item.get(field), str):
                    raise ValueError(f"Responses function call {field!r} must be a string.")
            output_index = event.get("output_index")
            if not isinstance(output_index, int):
                raise ValueError("Responses function call output_index must be an integer.")
            item_id = item["id"]
            if item_id in calls_by_item_id:
                raise ValueError(f"Duplicate Responses function call item ID: {item_id!r}.")
            call = {
                "arguments": item["arguments"],
                "argument_deltas": [],
                "arguments_done": [],
                "call_id": item["call_id"],
                "done_items": [],
                "index": len(calls),
                "item_id": item_id,
                "name": item["name"],
                "output_index": output_index,
            }
            calls.append(call)
            calls_by_item_id[item_id] = call
            chunks.append({"arguments": item["arguments"], "call": call, "initial": True})
            continue

        if event_type == "response.function_call_arguments.delta":
            item_id = event.get("item_id")
            call = calls_by_item_id.get(item_id) if isinstance(item_id, str) else None
            if call is None:
                raise ValueError("Responses function call argument delta targets an unknown item.")
            if event.get("output_index") != call["output_index"]:
                raise ValueError("Responses function call argument delta targets the wrong item.")
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise ValueError("Responses function call argument deltas must be strings.")
            call["argument_deltas"].append(delta)
            chunks.append({"arguments": delta, "call": call, "initial": False})
            continue

        if event_type == "response.function_call_arguments.done":
            item_id = event.get("item_id")
            call = calls_by_item_id.get(item_id) if isinstance(item_id, str) else None
            if call is None:
                raise ValueError(
                    "Completed Responses function call arguments target an unknown item."
                )
            if event.get("output_index") != call["output_index"]:
                raise ValueError(
                    "Completed Responses function call arguments target the wrong item."
                )
            call["arguments_done"].append(event)
            continue

        if (
            event_type == "response.output_item.done"
            and isinstance(item, dict)
            and item.get("type") == "function_call"
        ):
            item_id = item.get("id")
            call = calls_by_item_id.get(item_id) if isinstance(item_id, str) else None
            if call is None:
                raise ValueError("Completed Responses function call targets an unknown item.")
            call["done_items"].append(event)

    for call in calls:
        if len(call["arguments_done"]) != 1:
            raise ValueError(
                "Expected exactly one 'response.function_call_arguments.done' event for "
                f"function call {call['item_id']!r}, found {len(call['arguments_done'])}."
            )
        if len(call["done_items"]) != 1:
            raise ValueError(
                "Expected exactly one completed Responses output item for function call "
                f"{call['item_id']!r}, found {len(call['done_items'])}."
            )
        done_event = call["done_items"][0]
        done_item = done_event["item"]
        if done_event.get("output_index") != call["output_index"]:
            raise ValueError("Responses function call lifecycle disagrees on 'output_index'.")
        for field in ("id", "call_id", "name"):
            if done_item.get(field) != call[field if field != "id" else "item_id"]:
                raise ValueError(f"Responses function call lifecycle disagrees on {field!r}.")

        arguments = call["arguments"] + "".join(call["argument_deltas"])
        arguments_done = call["arguments_done"][0]
        if arguments_done.get("arguments") != arguments or done_item.get("arguments") != arguments:
            raise ValueError(
                "Responses function call argument deltas do not match completed arguments."
            )

    return {"calls": calls, "chunks": chunks}


def _chat_delta_events(
    indexed_events: list[tuple[int, dict[str, Any]]], delta_field: str
) -> list[tuple[int, int, dict[str, Any]]]:
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for event_position, (sse_index, event) in enumerate(indexed_events):
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue

        choice = choices[0]
        delta = choice.get("delta")
        if (
            isinstance(delta, dict)
            and "role" not in delta
            and choice.get("finish_reason") is None
            and isinstance(delta.get(delta_field), str)
        ):
            matches.append((event_position, sse_index, event))
    return matches


def _chat_tool_call_events(
    indexed_events: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, int, dict[str, Any]]]:
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for event_position, (sse_index, event) in enumerate(indexed_events):
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict) or "tool_calls" not in delta:
            continue
        tool_calls = delta["tool_calls"]
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError("Chat Completions tool_calls must be a non-empty list.")
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or not isinstance(tool_call.get("index"), int):
                raise ValueError("Chat Completions tool call indices must be integers.")
            function = tool_call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
                raise ValueError("Chat Completions tool call arguments must be strings.")
        matches.append((event_position, sse_index, event))
    return matches


def _map_usage(response_usage: dict[str, Any], chat_usage: dict[str, Any]) -> None:
    chat_usage["prompt_tokens"] = response_usage["input_tokens"]
    chat_usage["completion_tokens"] = response_usage["output_tokens"]
    chat_usage["total_tokens"] = response_usage["total_tokens"]

    input_details = response_usage.get("input_tokens_details") or {}
    if input_details:
        prompt_details = chat_usage.get("prompt_tokens_details") or {}
        for field in ("cache_write_tokens", "cached_tokens"):
            if field in input_details:
                prompt_details[field] = input_details[field]
        chat_usage["prompt_tokens_details"] = prompt_details

    output_details = response_usage.get("output_tokens_details") or {}
    if "reasoning_tokens" in output_details:
        completion_details = chat_usage.get("completion_tokens_details") or {}
        completion_details["reasoning_tokens"] = output_details["reasoning_tokens"]
        chat_usage["completion_tokens_details"] = completion_details


def _replace_delta_events(
    chat_cassette: dict[str, Any],
    indexed_chat_events: list[tuple[int, dict[str, Any]]],
    delta_events: list[tuple[int, int, dict[str, Any]]],
    response_deltas: list[str],
    delta_field: str,
) -> None:
    if not delta_events and not response_deltas:
        return
    if not delta_events:
        raise ValueError(
            f"Cannot insert {delta_field} deltas without a Chat Completions template."
        )

    sse_entries = chat_cassette["response"]["sse"]
    event_positions = [event_position for event_position, _, _ in delta_events]
    if event_positions != list(range(event_positions[0], event_positions[-1] + 1)):
        raise ValueError(f"Chat Completions {delta_field} delta events must be contiguous.")

    first_event_position, segment_start, _ = delta_events[0]
    last_event_position = delta_events[-1][0]
    segment_end = (
        indexed_chat_events[last_event_position + 1][0]
        if last_event_position + 1 < len(indexed_chat_events)
        else len(sse_entries)
    )
    next_event_index = (
        indexed_chat_events[first_event_position + 1][0]
        if first_event_position + 1 < len(indexed_chat_events)
        else len(sse_entries)
    )
    separator_entries = sse_entries[segment_start + 1 : next_event_index]
    templates = [(sse_index, event) for _, sse_index, event in delta_events]

    replacement_entries: list[str] = []
    for index, delta_value in enumerate(response_deltas):
        template_index, template_event = templates[min(index, len(templates) - 1)]
        event = deepcopy(template_event)
        choice = event["choices"][0]
        choice["delta"] = {delta_field: delta_value}
        choice["finish_reason"] = None
        event["usage"] = None
        replacement_entries.append(_serialize_data_event(event, sse_entries[template_index]))
        replacement_entries.extend(separator_entries)

    sse_entries[segment_start:segment_end] = replacement_entries


def _replace_tool_call_events(
    chat_cassette: dict[str, Any],
    indexed_chat_events: list[tuple[int, dict[str, Any]]],
    response_tool_calls: dict[str, Any],
) -> None:
    tool_call_events = _chat_tool_call_events(indexed_chat_events)
    tool_finish_events = [
        event
        for _, event in indexed_chat_events
        if isinstance(event.get("choices"), list)
        and event["choices"]
        and isinstance(event["choices"][0], dict)
        and event["choices"][0].get("finish_reason") == "tool_calls"
    ]
    if not response_tool_calls["calls"]:
        if tool_call_events or tool_finish_events:
            raise ValueError(
                "Responses and Chat Completions streams disagree on the presence of a tool call."
            )
        return
    if not tool_call_events:
        raise ValueError(
            "Responses and Chat Completions streams disagree on the presence of a tool call."
        )
    if len(tool_finish_events) != 1:
        raise ValueError(
            "Expected exactly one Chat Completions tool_calls finish event, "
            f"found {len(tool_finish_events)}."
        )

    event_positions = [position for position, _, _ in tool_call_events]
    if event_positions != list(range(event_positions[0], event_positions[-1] + 1)):
        raise ValueError("Chat Completions tool call delta events must be contiguous.")

    sse_entries = chat_cassette["response"]["sse"]
    first_event_position, segment_start, _ = tool_call_events[0]
    last_event_position = tool_call_events[-1][0]
    segment_end = (
        indexed_chat_events[last_event_position + 1][0]
        if last_event_position + 1 < len(indexed_chat_events)
        else len(sse_entries)
    )
    next_event_index = (
        indexed_chat_events[first_event_position + 1][0]
        if first_event_position + 1 < len(indexed_chat_events)
        else len(sse_entries)
    )
    separator_entries = sse_entries[segment_start + 1 : next_event_index]
    templates = [(sse_index, event) for _, sse_index, event in tool_call_events]

    replacement_entries: list[str] = []
    preceding_deltas: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for sse_index, event in indexed_chat_events[:first_event_position]:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta")
        if isinstance(delta, dict):
            preceding_deltas.append((sse_index, event, delta))

    has_visible_output = any(
        any(
            isinstance(delta.get(field), str) and bool(delta[field])
            for field in ("content", "reasoning_content")
        )
        for _, _, delta in preceding_deltas
    )
    if not has_visible_output:
        role_events = [
            (sse_index, event)
            for sse_index, event, delta in preceding_deltas
            if delta.get("role") == "assistant"
        ]
        role_delta = {"role": "assistant", "content": None, "refusal": None}
        if role_events:
            role_index, role_event = role_events[-1]
            role_event["choices"][0]["delta"] = role_delta
            sse_entries[role_index] = _serialize_data_event(role_event, sse_entries[role_index])
        else:
            template_index, template_event = templates[0]
            role_event = deepcopy(template_event)
            role_choice = role_event["choices"][0]
            role_choice["delta"] = role_delta
            role_choice["finish_reason"] = None
            role_event["usage"] = None
            replacement_entries.append(
                _serialize_data_event(role_event, sse_entries[template_index])
            )
            replacement_entries.extend(separator_entries)

    for index, response_chunk in enumerate(response_tool_calls["chunks"]):
        call = response_chunk["call"]
        template_index, template_event = templates[min(index, len(templates) - 1)]
        event = deepcopy(template_event)
        choice = event["choices"][0]
        if response_chunk["initial"]:
            tool_call = {
                "index": call["index"],
                "id": call["call_id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": response_chunk["arguments"],
                },
            }
            choice["delta"] = {"content": None, "tool_calls": [tool_call]}
        else:
            choice["delta"] = {
                "tool_calls": [
                    {
                        "index": call["index"],
                        "function": {"arguments": response_chunk["arguments"]},
                    }
                ]
            }
        choice["finish_reason"] = None
        event["usage"] = None
        replacement_entries.append(_serialize_data_event(event, sse_entries[template_index]))
        replacement_entries.extend(separator_entries)

    sse_entries[segment_start:segment_end] = replacement_entries


def match_chat_completions_events(
    chat_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    """Rewrite CC SSE events into the canonical input for Responses parity."""
    _match_request_reasoning_effort(chat_cassette, responses_cassette)
    indexed_chat_events = _data_events(chat_cassette)
    response_events = [event for _, event in _data_events(responses_cassette)]

    created_event = _response_event(response_events, "response.created")
    created_response = created_event.get("response")
    terminal_response, finish_reason = _response_terminal(response_events)
    if not isinstance(created_response, dict):
        raise ValueError("Responses lifecycle events must contain response objects.")
    for field in ("id", "created_at", "model"):
        if created_response.get(field) != terminal_response.get(field):
            raise ValueError(f"Responses lifecycle events disagree on {field!r}.")

    response_id = created_response["id"]
    created_at = created_response["created_at"]
    model = terminal_response["model"]
    response_usage = terminal_response.get("usage")
    if not isinstance(response_usage, dict):
        raise ValueError("Responses terminal event must contain usage for stream matching.")

    response_content_deltas = _response_text_deltas(
        response_events, "response.output_text", optional=True
    )
    response_reasoning_deltas = _response_text_deltas(
        response_events, "response.reasoning_text", optional=True
    )
    response_tool_calls = _response_tool_calls(response_events)
    if response_tool_calls["calls"] and terminal_response.get("status") == "completed":
        finish_reason = "tool_calls"

    usage_events = [
        usage
        for _, event in indexed_chat_events
        if isinstance((usage := event.get("usage")), dict)
    ]
    finish_events = [
        event
        for _, event in indexed_chat_events
        if event.get("choices") and event["choices"][0].get("finish_reason") is not None
    ]

    if len(usage_events) != 1:
        raise ValueError(
            f"Expected exactly one Chat Completions usage event, found {len(usage_events)}."
        )
    if len(finish_events) != 1:
        raise ValueError(
            f"Expected exactly one Chat Completions finish event, found {len(finish_events)}."
        )

    _replace_delta_events(
        chat_cassette,
        indexed_chat_events,
        _chat_delta_events(indexed_chat_events, "reasoning_content"),
        response_reasoning_deltas,
        "reasoning_content",
    )
    indexed_chat_events = _data_events(chat_cassette)
    _replace_delta_events(
        chat_cassette,
        indexed_chat_events,
        _chat_delta_events(indexed_chat_events, "content"),
        response_content_deltas,
        "content",
    )
    indexed_chat_events = _data_events(chat_cassette)
    _replace_tool_call_events(chat_cassette, indexed_chat_events, response_tool_calls)
    indexed_chat_events = _data_events(chat_cassette)
    usage_events = [
        usage
        for _, event in indexed_chat_events
        if isinstance((usage := event.get("usage")), dict)
    ]
    for _, event in indexed_chat_events:
        event["id"] = response_id
        event["created"] = created_at
        event["model"] = model
        event["service_tier"] = "default"
    finish_events = [
        event
        for _, event in indexed_chat_events
        if event.get("choices") and event["choices"][0].get("finish_reason") is not None
    ]
    finish_events[0]["choices"][0]["finish_reason"] = finish_reason
    _map_usage(response_usage, usage_events[0])

    sse_entries = chat_cassette["response"]["sse"]
    for index, event in indexed_chat_events:
        sse_entries[index] = _serialize_data_event(event, sse_entries[index])
    _match_request_history(chat_cassette, responses_cassette)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Chat Completions input cassette for exact CC-to-Responses stream "
            "parity. The Chat Completions cassette is updated in place unless --output "
            "is provided."
        )
    )
    parser.add_argument("chat_completions_cassette", type=Path)
    parser.add_argument("responses_cassette", type=Path)
    parser.add_argument("--output", type=Path, help="Write to a different YAML file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    chat_cassette = read_yaml(str(args.chat_completions_cassette))
    responses_cassette = read_yaml(str(args.responses_cassette))
    if not isinstance(chat_cassette, dict) or not isinstance(responses_cassette, dict):
        raise ValueError("Cassette roots must be YAML mappings.")

    match_chat_completions_events(chat_cassette, responses_cassette)
    output_path = args.output or args.chat_completions_cassette
    dump_yaml(chat_cassette, str(output_path), sort_keys=True)
    print(f"Matched Chat Completions events and wrote {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
