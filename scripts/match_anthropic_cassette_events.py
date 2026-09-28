#!/usr/bin/env python3
"""Prepare a Messages cassette for Messages-to-Responses stream parity tests.

The Responses cassette is the canonical expected output. This script rewrites the
Anthropic Messages SSE stream so a converter receives one corresponding thinking,
text, or tool-input delta for every equivalent Responses delta, in the same order
and with the same content. It also synchronizes fields that can pass through the
conversion, including request history, the request model, response ID and model,
tool call IDs and names, same-provider opaque reasoning state, and terminal usage.
Compatible function definitions are mapped from Responses `tools` into Messages
`tools`. Responses tool selection and `parallel_tool_calls` are represented by
Messages `tool_choice` and its inverse `disable_parallel_tool_use` flag.
Translated Messages responses use the target default service tier
`standard` because OpenAI tier labels have no faithful Anthropic equivalent.
Responses normal completion maps to Messages `end_turn`, or `tool_use` when
the response contains tool calls. An incomplete output-token limit maps to
`max_tokens`. A Responses content-filter limit is coerced to `end_turn`
because Messages has no corresponding stop reason.
Matched reasoning requests use adaptive summarized thinking and carry normalized
reasoning effort in `output_config.effort`; reasoning history defaults that
effort to `high` when the canonical request omits it.
System/developer input becomes top-level system content, preserving marked block
boundaries. OpenAI cache options and markers map to automatic/block cache controls;
30m maps to 1h under Molto's documented lifetime approximation. Cache usage,
including an existing creation-lifetime breakdown, follows the Responses counts.

The matched fixtures assume OpenAI serves both the Responses and Messages
endpoints. Under that same-provider contract, Responses `encrypted_content`
and Messages thinking `signature` carry the same OpenAI-issued opaque state.
The matcher maps the `response.output_item.added` snapshot to the Messages
thinking-block start signature and the `response.output_item.done` snapshot to
the later replacement signature. Messages has no third lifecycle slot for the
terminal response-output snapshot, so matching intentionally preserves the first
two snapshots rather than buffering for terminal byte parity. This mapping must
not be generalized to an Anthropic-issued signature.

An encrypted-only Responses reasoning item has no visible reasoning deltas. The
matcher preserves an existing Messages thinking block without its unrelated text,
or inserts an empty thinking block when the recorded Messages response omitted one,
so the two opaque snapshots still have native Messages lifecycle slots.

The result remains an Anthropic Messages wire-format cassette. Responses-only
envelope fields such as item IDs, sequence numbers, and timestamps cannot be
encoded in Messages events; the converter must generate those fields
deterministically for parity. Messages-only lifecycle and transport events remain
native, and all transport metadata is intentionally left unchanged.

Hosted search uses a reduced Messages projection with observed sources,
answer-range citations and N/A state placeholders. Completed searches lacking
source data fail matching. Cited text parts wait for complete annotations;
unannotated text, reasoning, and client function deltas retain their boundaries.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from molto.utils.io import dump_yaml, json_dumps, json_loads, read_yaml

if __package__:
    from scripts.match_anthropic_cassette_responses import (
        _cited_text_content,
        _map_search_usage,
        _messages_tool_choice,
        _response_content,
        _search_content,
        _web_search_tool,
    )
else:
    from match_anthropic_cassette_responses import (
        _cited_text_content,
        _map_search_usage,
        _messages_tool_choice,
        _response_content,
        _search_content,
        _web_search_tool,
    )


@dataclass
class _SSEEventUnit:
    entries: list[str]
    data_entry_index: int
    event: dict[str, Any]


@dataclass(frozen=True)
class _ResponseToolCall:
    call_id: str
    name: str
    argument_deltas: list[str]


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


def _sse_entries(cassette: dict[str, Any]) -> list[str]:
    try:
        sse_entries = cassette["response"]["sse"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cassette must contain response.sse.") from exc

    if not isinstance(sse_entries, list) or not all(
        isinstance(entry, str) for entry in sse_entries
    ):
        raise ValueError("Cassette response.sse must be a list of strings.")
    return sse_entries


def _request_body(cassette: dict[str, Any]) -> dict[str, Any]:
    try:
        request_body = cassette["request"]["body"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cassette must contain request.body.") from exc
    if not isinstance(request_body, dict):
        raise ValueError("Cassette request.body must be an object.")
    return request_body


def _image_content(image_url: Any) -> dict[str, Any]:
    if not isinstance(image_url, str):
        raise ValueError("Responses image content must contain a string image_url field.")

    if not image_url.startswith("data:"):
        return {"type": "image", "source": {"type": "url", "url": image_url}}

    metadata, separator, data = image_url.partition(",")
    if not separator or not metadata.endswith(";base64"):
        raise ValueError("Responses image data URLs must contain base64-encoded data.")
    media_type = metadata.removeprefix("data:").removesuffix(";base64")
    if not media_type or not data:
        raise ValueError("Responses image data URLs must contain a media type and data.")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _message_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("Responses message content must be text or a list of content parts.")

    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Responses message content parts must be objects.")
        if part.get("type") == "input_image":
            blocks.append(_image_content(part.get("image_url")))
        elif part.get("type") in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Responses text content must contain a string text field.")
            blocks.extend(
                _cited_text_content(part)
                if part.get("annotations")
                else [{"type": "text", "text": text}]
            )
        else:
            raise ValueError("Responses message content cannot map to Messages.")
        if part.get("prompt_cache_breakpoint") is not None:
            blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    return blocks


def _reasoning_content(item: dict[str, Any]) -> dict[str, str]:
    content = item.get("content")
    if content is None:
        content = []
    if not isinstance(content, list):
        raise ValueError("Responses reasoning input must contain a content list.")
    summary = item.get("summary", [])
    if not isinstance(summary, list):
        raise ValueError("Responses reasoning input summary must be a list.")

    reasoning_text: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "reasoning_text":
            raise ValueError("Only raw Responses reasoning text can map to Messages thinking.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Responses reasoning text must be a string.")
        reasoning_text.append(text)

    summary_text: list[str] = []
    for part in summary:
        if not isinstance(part, dict) or part.get("type") != "summary_text":
            raise ValueError("Only Responses reasoning summaries can map to Messages thinking.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Responses reasoning summary text must be a string.")
        summary_text.append(text)

    thinking = {
        "type": "thinking",
        "thinking": "".join(reasoning_text or summary_text),
    }
    encrypted_content = item.get("encrypted_content")
    if encrypted_content is not None:
        if not isinstance(encrypted_content, str):
            raise ValueError("Responses encrypted reasoning content must be a string.")
        thinking["signature"] = encrypted_content
    return thinking


def _function_call_content(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise ValueError("Responses function calls must contain string call_id and name fields.")
    if not isinstance(arguments, str):
        raise ValueError("Responses function call arguments must be a string.")
    try:
        input_object = json_loads(arguments)
    except ValueError as exc:
        raise ValueError("Responses function call arguments must contain valid JSON.") from exc
    if not isinstance(input_object, dict):
        raise ValueError("Responses function call arguments must contain a JSON object.")
    return {
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": input_object,
    }


def _function_call_output_content(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    if not isinstance(call_id, str):
        raise ValueError("Responses function call outputs must contain a string call_id field.")
    output = item.get("output")
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": _message_content(output),
    }


def _append_assistant_content(
    messages: list[dict[str, Any]], content: list[dict[str, Any]]
) -> None:
    if messages and messages[-1].get("role") == "assistant":
        existing_content = messages[-1].get("content")
        if not isinstance(existing_content, list):
            raise ValueError("Messages assistant content must be a list.")
        existing_content.extend(content)
    else:
        messages.append({"content": content, "role": "assistant"})


def _append_tool_result(messages: list[dict[str, Any]], tool_result: dict[str, Any]) -> None:
    if messages and messages[-1].get("role") == "user":
        existing_content = messages[-1].get("content")
        if (
            isinstance(existing_content, list)
            and existing_content
            and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in existing_content
            )
        ):
            existing_content.append(tool_result)
            return
    messages.append({"content": [tool_result], "role": "user"})


def _messages_from_response_input(response_input: Any) -> list[dict[str, Any]]:
    if isinstance(response_input, str):
        return [
            {
                "content": [{"type": "text", "text": response_input}],
                "role": "user",
            }
        ]
    if not isinstance(response_input, list):
        raise ValueError("Responses input must be a string or an input-item list.")

    messages: list[dict[str, Any]] = []
    pending_assistant_content: list[dict[str, Any]] = []

    for item in response_input:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects.")
        item_type = item.get("type")
        if item_type == "reasoning":
            pending_assistant_content.append(_reasoning_content(item))
            continue
        if item_type == "web_search_call":
            _append_assistant_content(
                messages, [*pending_assistant_content, *_search_content(item)]
            )
            pending_assistant_content = []
            continue
        if item_type == "function_call":
            _append_assistant_content(
                messages,
                [*pending_assistant_content, _function_call_content(item)],
            )
            pending_assistant_content = []
            continue
        if item_type == "function_call_output":
            if pending_assistant_content:
                raise ValueError(
                    "Responses reasoning input must precede an assistant output item."
                )
            _append_tool_result(messages, _function_call_output_content(item))
            continue

        role = item.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"Responses input role {role!r} cannot map to Messages.")
        content = _message_content(item.get("content"))
        if role == "assistant":
            messages.append(
                {"content": [*pending_assistant_content, *content], "role": "assistant"}
            )
            pending_assistant_content = []
        else:
            if pending_assistant_content:
                raise ValueError("Responses reasoning input must precede an assistant message.")
            messages.append({"content": content, "role": "user"})

    if pending_assistant_content:
        raise ValueError("Responses reasoning input must precede an assistant message.")
    return messages


def _messages_tools(response_tools: Any) -> list[dict[str, Any]] | None:
    if response_tools is None:
        return None
    if not isinstance(response_tools, list):
        raise ValueError("Responses tools must be a list.")

    tools: list[dict[str, Any]] = []
    for tool in response_tools:
        if isinstance(tool, dict) and tool.get("type") in {"web_search"}:
            tools.append(_web_search_tool(tool))
            continue
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Only Responses function tools can map to Messages tools.")

        name = tool.get("name")
        parameters = tool.get("parameters")
        if not isinstance(name, str):
            raise ValueError("Responses function tools must contain a string name.")
        if not isinstance(parameters, dict):
            raise ValueError("Responses function tools must contain an object parameters schema.")

        messages_tool: dict[str, Any] = {
            "input_schema": deepcopy(parameters),
            "name": name,
        }
        description = tool.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError("Responses function tool descriptions must be strings.")
            messages_tool["description"] = description
        strict = tool.get("strict")
        if strict is not None:
            if not isinstance(strict, bool):
                raise ValueError("Responses function tool strict values must be booleans.")
            if strict:
                messages_tool["strict"] = True

        unsupported = {
            key
            for key, value in tool.items()
            if key
            not in {
                "description",
                "name",
                "parameters",
                "strict",
                "type",
            }
            and value not in (None, False)
        }
        if unsupported:
            raise ValueError(
                f"Unsupported Responses function tool fields for Messages: {sorted(unsupported)}"
            )
        tools.append(messages_tool)
    return tools


def _matched_request_body(
    messages_request: dict[str, Any], responses_request: dict[str, Any]
) -> dict[str, Any]:
    request_model = responses_request.get("model")
    if not isinstance(request_model, str):
        raise ValueError("Responses request must contain a string model.")
    stream = responses_request.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise ValueError("Responses request stream must be a boolean when provided.")

    matched_request = deepcopy(messages_request)
    response_input = responses_request.get("input")
    system = (
        _message_content(responses_request["instructions"])
        if responses_request.get("instructions")
        else []
    )
    has_markers = _has_cache_breakpoint(response_input)
    if isinstance(response_input, list):
        message_input = []
        for item in response_input:
            if isinstance(item, dict) and item.get("role") in {"system", "developer"}:
                if message_input and has_markers:
                    raise ValueError(
                        "Cannot hoist mid-conversation instructions with cache breakpoints."
                    )
                system.extend(_message_content(item.get("content")))
            else:
                message_input.append(item)
        response_input = message_input
    if system:
        matched_request["system"] = (
            system
            if any("cache_control" in block for block in system)
            else "\n\n".join(block["text"] for block in system)
        )
    else:
        matched_request.pop("system", None)
    matched_request["messages"] = _messages_from_response_input(response_input)
    if responses_request.get("prompt_cache_retention") is not None:
        raise ValueError("Messages cannot represent prompt cache retention.")
    options = responses_request.get("prompt_cache_options") or {}
    matched_request.pop("cache_control", None)
    if options.get("mode") != "explicit" and (
        options.get("mode") == "implicit" or options.get("ttl") is not None or has_markers
    ):
        matched_request["cache_control"] = {"type": "ephemeral"}
        if options.get("ttl") is not None or has_markers:
            matched_request["cache_control"]["ttl"] = "1h"
    matched_request["model"] = request_model
    if stream is None:
        matched_request.pop("stream", None)
    else:
        matched_request["stream"] = stream
    if responses_request.get("max_output_tokens") is not None:
        matched_request["max_tokens"] = responses_request["max_output_tokens"]

    matched_tools = _messages_tools(responses_request.get("tools"))
    if matched_tools is None:
        matched_request.pop("tools", None)
    else:
        matched_request["tools"] = matched_tools
    if matched_tools and responses_request.get("max_tool_calls") is not None:
        for tool in matched_tools:
            if tool.get("type") == "web_search_20250305":
                tool["max_uses"] = responses_request["max_tool_calls"]
    matched_tool_choice = _messages_tool_choice(responses_request)
    if matched_tool_choice is None:
        matched_request.pop("tool_choice", None)
    else:
        matched_request["tool_choice"] = matched_tool_choice

    response_input = responses_request.get("input")
    has_thinking_history = isinstance(response_input, list) and any(
        isinstance(item, dict) and item.get("type") == "reasoning" for item in response_input
    )
    reasoning = responses_request.get("reasoning")
    reasoning_effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    reasoning_summary = reasoning.get("summary") if isinstance(reasoning, dict) else None
    if reasoning_effort == "none":
        matched_request["thinking"] = {"type": "disabled"}
        matched_request.pop("output_config", None)
    elif (
        reasoning_effort is not None
        or reasoning_summary is not None
        or has_thinking_history
        or (
            isinstance(matched_request.get("thinking"), dict)
            and matched_request["thinking"].get("type") == "adaptive"
        )
    ):
        effort = "high" if reasoning_effort is None else reasoning_effort
        if effort == "minimal":
            effort = "low"
        matched_request["thinking"] = {"type": "adaptive", "display": "summarized"}
        existing_output_config = matched_request.get("output_config")
        output_config = (
            {"format": existing_output_config["format"]}
            if isinstance(existing_output_config, dict) and "format" in existing_output_config
            else {}
        )
        output_config["effort"] = effort
        matched_request["output_config"] = output_config
    return matched_request


def _has_cache_breakpoint(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("prompt_cache_breakpoint") is not None or any(
            _has_cache_breakpoint(item) for item in value.values()
        )
    return isinstance(value, list) and any(_has_cache_breakpoint(item) for item in value)


def _data_events(cassette: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, event)
        for index, entry in enumerate(_sse_entries(cassette))
        if (event := _parse_data_event(entry)) is not None
    ]


def _message_event_units(
    cassette: dict[str, Any],
) -> tuple[list[str], list[_SSEEventUnit]]:
    sse_entries = _sse_entries(cassette)
    indexed_events = _data_events(cassette)
    if not indexed_events:
        raise ValueError("Messages cassette must contain SSE data events.")

    unit_starts: list[int] = []
    previous_data_index = -1
    for data_index, _ in indexed_events:
        event_header_indices = [
            index
            for index in range(previous_data_index + 1, data_index)
            if sse_entries[index].strip().startswith("event:")
        ]
        unit_starts.append(event_header_indices[-1] if event_header_indices else data_index)
        previous_data_index = data_index

    prefix = sse_entries[: unit_starts[0]]
    units: list[_SSEEventUnit] = []
    for position, ((data_index, event), unit_start) in enumerate(
        zip(indexed_events, unit_starts, strict=True)
    ):
        unit_end = (
            unit_starts[position + 1] if position + 1 < len(unit_starts) else len(sse_entries)
        )
        units.append(
            _SSEEventUnit(
                entries=sse_entries[unit_start:unit_end],
                data_entry_index=data_index - unit_start,
                event=event,
            )
        )
    return prefix, units


def _response_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("type") == event_type]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {event_type!r} event, found {len(matches)}.")
    return matches[0]


def _response_terminal(
    events: list[dict[str, Any]],
    *,
    has_search: bool = False,
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
        return response, "end_turn"

    incomplete_details = response.get("incomplete_details")
    if incomplete_details is None and (
        has_search
        or any(item.get("type") == "web_search_call" for item in response.get("output", []))
    ):
        return response, "pause_turn"
    if not isinstance(incomplete_details, dict):
        raise ValueError("response.incomplete must contain incomplete_details.")
    reason = incomplete_details.get("reason")
    if reason == "max_output_tokens":
        return response, "max_tokens"
    return response, "end_turn"


def _serialize_data_event(event: dict[str, Any], template: str) -> str:
    if _parse_data_event(template) == event:
        return template
    suffix = template[len(template.rstrip("\r\n")) :]
    return f"data: {json_dumps(event)}{suffix}"


def _serialize_event_name(event_type: str, template: str) -> str:
    suffix = template[len(template.rstrip("\r\n")) :]
    return f"event: {event_type}{suffix}"


def _response_text_deltas(
    events: list[dict[str, Any]], event_prefix: str, *, optional: bool = False
) -> list[str] | None:
    delta_type = f"{event_prefix}.delta"
    done_type = f"{event_prefix}.done"
    deltas = [event["delta"] for event in events if event.get("type") == delta_type]
    if not all(isinstance(delta, str) for delta in deltas):
        raise ValueError(f"{event_prefix} deltas must be strings.")

    done_events = [event for event in events if event.get("type") == done_type]
    if optional and not deltas and not done_events:
        return None
    if len(done_events) != 1:
        raise ValueError(f"Expected exactly one {done_type!r} event, found {len(done_events)}.")

    done_text = done_events[0].get("text")
    if "".join(deltas) != done_text:
        raise ValueError(f"{event_prefix} deltas do not match {event_prefix}.done.")
    return deltas


def _response_reasoning_deltas(events: list[dict[str, Any]]) -> list[str] | None:
    summary_deltas = _response_text_deltas(
        events, "response.reasoning_summary_text", optional=True
    )
    reasoning_deltas = _response_text_deltas(events, "response.reasoning_text", optional=True)
    deltas = summary_deltas if summary_deltas is not None else reasoning_deltas
    if deltas is not None:
        return deltas

    has_reasoning_item = any(
        event.get("type") in {"response.output_item.added", "response.output_item.done"}
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "reasoning"
        for event in events
    )
    return [] if has_reasoning_item else None


def _response_reasoning_encrypted_content(
    events: list[dict[str, Any]], event_type: str
) -> str | None:
    reasoning_items = [
        event.get("item")
        for event in events
        if event.get("type") == event_type
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "reasoning"
    ]
    if len(reasoning_items) != 1:
        raise ValueError(
            f"Expected exactly one Responses reasoning item in {event_type!r}, "
            f"found {len(reasoning_items)}."
        )

    encrypted_content = reasoning_items[0].get("encrypted_content")
    if encrypted_content is not None and not isinstance(encrypted_content, str):
        raise ValueError("Responses encrypted reasoning content must be a string.")
    return encrypted_content


def _response_tool_calls(events: list[dict[str, Any]]) -> list[_ResponseToolCall]:
    completed_calls: list[tuple[int, dict[str, Any]]] = []
    for event in events:
        item = event.get("item")
        if (
            event.get("type") != "response.output_item.done"
            or not isinstance(item, dict)
            or item.get("type") != "function_call"
        ):
            continue
        output_index = event.get("output_index")
        if not isinstance(output_index, int):
            raise ValueError(
                "Completed Responses function calls must have an integer output_index."
            )
        completed_calls.append((output_index, item))

    output_indices = [output_index for output_index, _ in completed_calls]
    if len(output_indices) != len(set(output_indices)):
        raise ValueError("Completed Responses function call output indices must be unique.")
    completed_calls.sort(key=lambda value: value[0])

    calls: list[_ResponseToolCall] = []
    item_ids: set[str] = set()
    for _, item in completed_calls:
        item_id = item.get("id")
        call_id = item.get("call_id")
        name = item.get("name")
        arguments = item.get("arguments")
        if not all(isinstance(value, str) for value in (item_id, call_id, name, arguments)):
            raise ValueError(
                "Completed Responses function calls must contain string id, call_id, name, "
                "and arguments fields."
            )
        if item_id in item_ids:
            raise ValueError(f"Responses function call item ID {item_id!r} is duplicated.")
        item_ids.add(item_id)

        argument_deltas = [
            event.get("delta")
            for event in events
            if event.get("type") == "response.function_call_arguments.delta"
            and event.get("item_id") == item_id
        ]
        if not all(isinstance(delta, str) for delta in argument_deltas):
            raise ValueError("Responses function call argument deltas must be strings.")
        done_events = [
            event
            for event in events
            if event.get("type") == "response.function_call_arguments.done"
            and event.get("item_id") == item_id
        ]
        if len(done_events) != 1:
            raise ValueError(
                "Expected exactly one Responses function call arguments done event for "
                f"item {item_id!r}, found {len(done_events)}."
            )
        done_arguments = done_events[0].get("arguments")
        if "".join(argument_deltas) != done_arguments or done_arguments != arguments:
            raise ValueError("Responses function call argument deltas do not match arguments.")
        calls.append(
            _ResponseToolCall(
                call_id=call_id,
                name=name,
                argument_deltas=argument_deltas,
            )
        )

    argument_item_ids = {
        event.get("item_id")
        for event in events
        if event.get("type")
        in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }
    }
    if argument_item_ids != item_ids:
        raise ValueError(
            "Responses function call argument events do not match completed function calls."
        )
    return calls


def _replace_thinking_start_signature(units: list[_SSEEventUnit], signature: str | None) -> None:
    block = _content_block(units, "thinking")
    if block is None:
        raise ValueError("Messages cassette has no thinking block for Responses reasoning.")
    start_position, _, _ = block
    content_block = units[start_position].event.get("content_block")
    if not isinstance(content_block, dict):
        raise ValueError("Messages thinking start must contain a content block.")
    content_block["signature"] = signature or ""


def _content_blocks(units: list[_SSEEventUnit], block_type: str) -> list[tuple[int, int, int]]:
    starts = [
        (position, unit.event)
        for position, unit in enumerate(units)
        if unit.event.get("type") == "content_block_start"
        and isinstance(unit.event.get("content_block"), dict)
        and unit.event["content_block"].get("type") == block_type
    ]
    blocks: list[tuple[int, int, int]] = []
    for start_position, start_event in starts:
        block_index = start_event.get("index")
        if not isinstance(block_index, int):
            raise ValueError(f"Messages {block_type!r} content block index must be an integer.")

        stop_positions = [
            position
            for position, unit in enumerate(units)
            if position > start_position
            and unit.event.get("type") == "content_block_stop"
            and unit.event.get("index") == block_index
        ]
        if len(stop_positions) != 1:
            raise ValueError(
                f"Expected exactly one stop for Messages {block_type!r} content block, "
                f"found {len(stop_positions)}."
            )
        blocks.append((start_position, stop_positions[0], block_index))
    return blocks


def _content_block(units: list[_SSEEventUnit], block_type: str) -> tuple[int, int, int] | None:
    blocks = _content_blocks(units, block_type)
    if len(blocks) > 1:
        raise ValueError(
            f"Expected at most one Messages {block_type!r} content block, found {len(blocks)}."
        )
    return blocks[0] if blocks else None


def _replace_block_deltas(
    units: list[_SSEEventUnit],
    block: tuple[int, int, int],
    response_deltas: list[str],
    *,
    delta_type: str,
    delta_field: str,
) -> None:
    block_start, block_stop, block_index = block
    block_units = units[block_start + 1 : block_stop]
    delta_units = [
        (position, unit)
        for position, unit in enumerate(block_units)
        if unit.event.get("type") == "content_block_delta"
        and unit.event.get("index") == block_index
        and isinstance(unit.event.get("delta"), dict)
        and unit.event["delta"].get("type") == delta_type
        and isinstance(unit.event["delta"].get(delta_field), str)
    ]
    if not delta_units and response_deltas:
        raise ValueError(f"Cannot insert {delta_type} events without a Messages template.")

    delta_positions = {position for position, _ in delta_units}
    replacement_units: list[_SSEEventUnit] = []
    next_delta = 0
    for position, unit in enumerate(block_units):
        if position not in delta_positions:
            replacement_units.append(unit)
            continue
        if next_delta < len(response_deltas):
            replacement = deepcopy(unit)
            replacement.event["delta"] = {
                "type": delta_type,
                delta_field: response_deltas[next_delta],
            }
            replacement_units.append(replacement)
            next_delta += 1
        if position == delta_units[-1][0]:
            while next_delta < len(response_deltas):
                template = deepcopy(delta_units[-1][1])
                template.event["delta"] = {
                    "type": delta_type,
                    delta_field: response_deltas[next_delta],
                }
                replacement_units.append(template)
                next_delta += 1

    units[block_start + 1 : block_stop] = replacement_units


def _replace_content_deltas(
    units: list[_SSEEventUnit],
    response_deltas: list[str] | None,
    *,
    block_type: str,
    delta_type: str,
    delta_field: str,
) -> None:
    block = _content_block(units, block_type)
    if (block is None) != (response_deltas is None):
        raise ValueError(
            f"Responses and Messages streams disagree on the presence of {block_type!r}."
        )
    if block is None or response_deltas is None:
        return

    _replace_block_deltas(
        units,
        block,
        response_deltas,
        delta_type=delta_type,
        delta_field=delta_field,
    )


def _insert_empty_thinking_block(units: list[_SSEEventUnit]) -> None:
    if _content_block(units, "thinking") is not None:
        return

    message_start_position = next(
        (
            position
            for position, unit in enumerate(units)
            if unit.event.get("type") == "message_start"
        ),
        None,
    )
    if message_start_position is None:
        raise ValueError("Messages cassette has no message_start event.")

    for unit in units:
        event = unit.event
        if event.get("type") not in {
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        }:
            continue
        block_index = event.get("index")
        if not isinstance(block_index, int):
            raise ValueError("Messages content block index must be an integer.")
        event["index"] = block_index + 1

    template = units[message_start_position]

    def new_unit(event: dict[str, Any]) -> _SSEEventUnit:
        unit = deepcopy(template)
        unit.event = event
        return unit

    units[message_start_position + 1 : message_start_position + 1] = [
        new_unit(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
            }
        ),
        new_unit(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": ""},
            }
        ),
        new_unit({"type": "content_block_stop", "index": 0}),
    ]


def _replace_tool_calls(
    units: list[_SSEEventUnit], response_tool_calls: list[_ResponseToolCall]
) -> None:
    blocks = _content_blocks(units, "tool_use")
    if len(blocks) != len(response_tool_calls):
        raise ValueError(
            "Responses and Messages streams disagree on the number of tool calls: "
            f"{len(response_tool_calls)} Responses calls and {len(blocks)} Messages blocks."
        )

    for block, response_tool_call in reversed(list(zip(blocks, response_tool_calls, strict=True))):
        block_start, _, _ = block
        content_block = units[block_start].event.get("content_block")
        if not isinstance(content_block, dict):
            raise ValueError("Messages tool_use content block must be an object.")
        content_block["id"] = response_tool_call.call_id
        content_block["name"] = response_tool_call.name
        content_block["input"] = {}
        _replace_block_deltas(
            units,
            block,
            response_tool_call.argument_deltas,
            delta_type="input_json_delta",
            delta_field="partial_json",
        )


def _message_event(units: list[_SSEEventUnit], event_type: str) -> dict[str, Any]:
    matches = [unit.event for unit in units if unit.event.get("type") == event_type]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {event_type!r} event, found {len(matches)}.")
    return matches[0]


def _map_input_usage(
    response_usage: dict[str, Any], message_usage: dict[str, Any], cache_ttl: str
) -> None:
    input_details = response_usage.get("input_tokens_details") or {}
    if not isinstance(input_details, dict):
        raise ValueError("Responses input token details must be an object.")

    input_tokens = response_usage["input_tokens"]
    cache_creation_tokens = input_details.get("cache_write_tokens", 0)
    cache_read_tokens = input_details.get("cached_tokens", 0)
    if not all(
        isinstance(value, int)
        for value in (input_tokens, cache_creation_tokens, cache_read_tokens)
    ):
        raise ValueError("Responses input token counts must be integers.")

    message_input_tokens = input_tokens - cache_creation_tokens - cache_read_tokens
    if message_input_tokens < 0:
        raise ValueError("Responses cache tokens exceed total input tokens.")

    message_usage["input_tokens"] = message_input_tokens
    message_usage["cache_creation_input_tokens"] = cache_creation_tokens
    message_usage["cache_read_input_tokens"] = cache_read_tokens
    if isinstance(message_usage.get("cache_creation"), dict):
        # OpenAI has one shared lifetime, so every projected write has the same TTL.
        message_usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": cache_creation_tokens if cache_ttl == "5m" else 0,
            "ephemeral_1h_input_tokens": cache_creation_tokens if cache_ttl == "1h" else 0,
        }


def _map_usage(
    response_usage: dict[str, Any], message_usage: dict[str, Any], cache_ttl: str
) -> None:
    _map_input_usage(response_usage, message_usage, cache_ttl)

    output_tokens = response_usage["output_tokens"]
    total_tokens = response_usage["total_tokens"]
    if not isinstance(output_tokens, int) or not isinstance(total_tokens, int):
        raise ValueError("Responses output and total token counts must be integers.")
    if response_usage["input_tokens"] + output_tokens != total_tokens:
        raise ValueError("Responses total tokens do not equal input plus output tokens.")
    message_usage["output_tokens"] = output_tokens

    output_details = response_usage.get("output_tokens_details") or {}
    if not isinstance(output_details, dict):
        raise ValueError("Responses output token details must be an object.")
    if "reasoning_tokens" in output_details:
        message_output_details = message_usage.setdefault("output_tokens_details", {})
        if not isinstance(message_output_details, dict):
            raise ValueError("Messages output token details must be an object.")
        message_output_details["thinking_tokens"] = output_details["reasoning_tokens"]


def _write_message_event_units(
    cassette: dict[str, Any], prefix: list[str], units: list[_SSEEventUnit]
) -> None:
    sse_entries = prefix.copy()
    for unit in units:
        event_type = unit.event.get("type")
        if not isinstance(event_type, str):
            raise ValueError("Messages event type must be a string.")
        event_name_index = next(
            (
                index
                for index, entry in enumerate(unit.entries[: unit.data_entry_index])
                if entry.strip().startswith("event:")
            ),
            None,
        )
        if event_name_index is not None:
            event_name_template = unit.entries[event_name_index]
            unit.entries[event_name_index] = _serialize_event_name(event_type, event_name_template)
        data_template = unit.entries[unit.data_entry_index]
        unit.entries[unit.data_entry_index] = _serialize_data_event(unit.event, data_template)
        sse_entries.extend(unit.entries)
    cassette["response"]["sse"] = sse_entries


def _replace_search_content(
    units: list[_SSEEventUnit], events: list[dict[str, Any]], response: dict[str, Any]
) -> None:
    """Rebuild balanced blocks while retaining reconstructible native delta boundaries."""
    projected: list[_SSEEventUnit] = []

    def append(event: dict[str, Any]) -> None:
        projected.append(
            _SSEEventUnit(
                [f"event: {event['type']}\n", f"data: {json_dumps(event)}\n", "\n"], 1, event
            )
        )

    def emit(block: dict[str, Any], deltas: list[dict[str, Any]]) -> None:
        nonlocal index
        append({"type": "content_block_start", "index": index, "content_block": block})
        for delta in deltas:
            append({"type": "content_block_delta", "index": index, "delta": delta})
        append({"type": "content_block_stop", "index": index})
        index += 1

    index = 0
    for output_index, terminal_item in enumerate(response["output"]):
        item_events = [event for event in events if event.get("output_index") == output_index]
        # Item-done owns opaque reasoning state, rather than the later terminal snapshot.
        item = next(
            (
                event["item"]
                for event in item_events
                if event.get("type") == "response.output_item.done"
            ),
            terminal_item,
        )
        if item["type"] == "message":
            for content_index, part in enumerate(item["content"]):
                part_events = [
                    event for event in item_events if event.get("content_index") == content_index
                ]
                chunks = _response_text_deltas(part_events, "response.output_text", optional=True)
                if chunks is not None and "".join(chunks) != part["text"]:
                    raise ValueError(
                        "Responses text deltas disagree with the output item snapshot."
                    )
                if part.get("logprobs") not in (None, []):
                    raise ValueError("Responses output logprobs cannot map to Messages.")
                for block in _cited_text_content(part):
                    # Only annotated parts need buffering and citation-range splitting.
                    text_chunks = (
                        [block["text"]] if part.get("annotations") or chunks is None else chunks
                    )
                    deltas = [{"type": "text_delta", "text": chunk} for chunk in text_chunks]
                    deltas.extend(
                        {"type": "citations_delta", "citation": citation}
                        for citation in block.get("citations", [])
                    )
                    emit({"type": "text", "text": ""}, deltas)
            continue
        blocks, _ = _response_content({"output": [item]})
        for block in blocks:
            start = deepcopy(block)
            deltas = []
            if block["type"] in {"server_tool_use", "tool_use"}:
                start["input"] = {}
                if block["type"] == "tool_use":
                    chunks = [
                        event["delta"]
                        for event in item_events
                        if event.get("type") == "response.function_call_arguments.delta"
                    ]
                    added_arguments = next(
                        (
                            event["item"].get("arguments", "")
                            for event in item_events
                            if event.get("type") == "response.output_item.added"
                        ),
                        "",
                    )
                    if added_arguments:
                        chunks.insert(0, added_arguments)
                    if chunks and "".join(chunks) != item["arguments"]:
                        raise ValueError(
                            "Responses function argument deltas disagree with its item snapshot."
                        )
                else:
                    chunks = []
                deltas.extend(
                    {"type": "input_json_delta", "partial_json": chunk}
                    for chunk in chunks or [json_dumps(block["input"])]
                )
            elif block["type"] == "thinking":
                start["thinking"] = ""
                added = next(
                    (
                        event["item"]
                        for event in item_events
                        if event.get("type") == "response.output_item.added"
                    ),
                    {},
                )
                start["signature"] = added.get("encrypted_content", "")
                chunks = _response_reasoning_deltas(item_events) or []
                if chunks and "".join(chunks) != block["thinking"]:
                    raise ValueError("Responses reasoning deltas disagree with its item snapshot.")
                deltas.extend(
                    {"type": "thinking_delta", "thinking": chunk}
                    for chunk in chunks or ([block["thinking"]] if block["thinking"] else [])
                )
                if "signature" in block:
                    deltas.append({"type": "signature_delta", "signature": block["signature"]})
            emit(start, deltas)
    # Keep native envelope/ping events and replace only content lifecycle events.
    insertion = next(
        (
            i
            for i, unit in enumerate(units)
            if unit.event.get("type", "").startswith("content_block_")
        ),
        next(i for i, unit in enumerate(units) if unit.event.get("type") == "message_delta"),
    )
    units[:] = [
        unit for unit in units if not unit.event.get("type", "").startswith("content_block_")
    ]
    units[insertion:insertion] = projected


def match_messages_events(
    messages_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    """Rewrite Messages SSE events into the canonical input for Responses parity."""
    messages_request = _request_body(messages_cassette)
    responses_request = _request_body(responses_cassette)
    matched_request = _matched_request_body(messages_request, responses_request)
    cache_ttl = (
        "1h"
        if _has_cache_breakpoint(responses_request.get("input"))
        else matched_request.get("cache_control", {}).get("ttl", "5m")
    )

    prefix, message_units = _message_event_units(messages_cassette)
    response_events = [event for _, event in _data_events(responses_cassette)]

    created_event = _response_event(response_events, "response.created")
    created_response = created_event.get("response")
    terminal_response, stop_reason = _response_terminal(
        response_events,
        has_search=any(
            tool.get("type") == "web_search" for tool in responses_request.get("tools", [])
        ),
    )
    if not isinstance(created_response, dict):
        raise ValueError("Responses lifecycle events must contain response objects.")
    for field in ("id", "created_at", "model"):
        if created_response.get(field) != terminal_response.get(field):
            raise ValueError(f"Responses lifecycle events disagree on {field!r}.")

    response_usage = terminal_response.get("usage")
    if not isinstance(response_usage, dict):
        raise ValueError("Responses terminal event must contain usage for stream matching.")

    has_search = (
        any(item.get("type") == "web_search_call" for item in terminal_response.get("output", []))
        or any(tool.get("type") == "web_search" for tool in responses_request.get("tools", []))
        or any(
            unit.event.get("content_block", {}).get("type")
            in {"server_tool_use", "web_search_tool_result"}
            for unit in message_units
        )
    )
    has_citations = any(
        part.get("annotations")
        for item in terminal_response.get("output", [])
        for part in item.get("content", [])
    )
    if has_search or has_citations:
        _replace_search_content(message_units, response_events, terminal_response)
        if stop_reason == "end_turn" and any(
            item.get("type") == "function_call" for item in terminal_response["output"]
        ):
            stop_reason = "tool_use"
    else:
        response_text_deltas = _response_text_deltas(
            response_events, "response.output_text", optional=True
        )
        response_reasoning_deltas = _response_reasoning_deltas(response_events)
        response_tool_calls = _response_tool_calls(response_events)
        if response_tool_calls and any(
            event.get("type") == "response.completed" for event in response_events
        ):
            stop_reason = "tool_use"
        if response_reasoning_deltas == []:
            _insert_empty_thinking_block(message_units)
        _replace_content_deltas(
            message_units,
            response_reasoning_deltas,
            block_type="thinking",
            delta_type="thinking_delta",
            delta_field="thinking",
        )
        if response_reasoning_deltas is None:
            response_reasoning_done_encrypted_content = None
        else:
            _replace_thinking_start_signature(
                message_units,
                _response_reasoning_encrypted_content(
                    response_events, "response.output_item.added"
                ),
            )
            response_reasoning_done_encrypted_content = _response_reasoning_encrypted_content(
                response_events, "response.output_item.done"
            )
        _replace_content_deltas(
            message_units,
            None
            if response_reasoning_deltas is None
            else (
                [response_reasoning_done_encrypted_content]
                if response_reasoning_done_encrypted_content is not None
                else []
            ),
            block_type="thinking",
            delta_type="signature_delta",
            delta_field="signature",
        )
        _replace_content_deltas(
            message_units,
            response_text_deltas,
            block_type="text",
            delta_type="text_delta",
            delta_field="text",
        )
        _replace_tool_calls(message_units, response_tool_calls)

    message_start = _message_event(message_units, "message_start").get("message")
    if not isinstance(message_start, dict):
        raise ValueError("message_start must contain a message object.")
    message_start["id"] = created_response["id"]
    message_start["model"] = terminal_response["model"]
    message_start["stop_reason"] = None
    message_start["stop_sequence"] = None
    # Empty server-container metadata has no counterpart in Responses or Chat.
    if message_start.get("container") is None:
        message_start.pop("container", None)

    message_start_usage = message_start.get("usage")
    if not isinstance(message_start_usage, dict):
        raise ValueError("message_start.message must contain usage.")
    _map_input_usage(response_usage, message_start_usage, cache_ttl)
    if has_search:
        _map_search_usage(created_response, message_start_usage)
    message_start_usage["output_tokens"] = 0
    message_start_usage["service_tier"] = "standard"

    message_delta = _message_event(message_units, "message_delta")
    message_delta_value = message_delta.get("delta")
    if not isinstance(message_delta_value, dict):
        raise ValueError("message_delta must contain a delta object.")
    message_delta_value["stop_reason"] = stop_reason
    message_delta_value["stop_sequence"] = None
    if message_delta_value.get("container") is None:
        message_delta_value.pop("container", None)

    message_delta_usage = message_delta.get("usage")
    if not isinstance(message_delta_usage, dict):
        raise ValueError("message_delta must contain usage.")
    _map_usage(response_usage, message_delta_usage, cache_ttl)
    if has_search:
        _map_search_usage(terminal_response, message_delta_usage)

    messages_cassette["request"]["body"] = matched_request
    _write_message_event_units(messages_cassette, prefix, message_units)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an Anthropic Messages input cassette for "
            "protocol stream parity, including reduced hosted-search projection. The Messages cassette is updated in "
            "place unless --output is provided."
        )
    )
    parser.add_argument("messages_cassette", type=Path)
    parser.add_argument("responses_cassette", type=Path)
    parser.add_argument("--output", type=Path, help="Write to a different YAML file.")
    parser.add_argument(
        "--request-only",
        action="store_true",
        help="Match request fields and preserve the response.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    messages_cassette = read_yaml(str(args.messages_cassette))
    responses_cassette = read_yaml(str(args.responses_cassette))
    if not isinstance(messages_cassette, dict) or not isinstance(responses_cassette, dict):
        raise ValueError("Cassette roots must be YAML mappings.")

    if args.request_only:
        messages_cassette["request"]["body"] = _matched_request_body(
            _request_body(messages_cassette), _request_body(responses_cassette)
        )
    else:
        match_messages_events(messages_cassette, responses_cassette)
    output_path = args.output or args.messages_cassette
    dump_yaml(messages_cassette, str(output_path), sort_keys=True)
    print(f"Matched Anthropic Messages cassette and wrote {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
