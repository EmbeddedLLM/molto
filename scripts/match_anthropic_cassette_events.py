#!/usr/bin/env python3
"""Prepare a Messages cassette for Messages-to-Responses stream parity tests.

The Responses cassette is the canonical expected output. This script rewrites the
Anthropic Messages SSE stream so a converter receives one corresponding thinking
or text delta for every ``response.reasoning_text.delta`` or
``response.output_text.delta`` event, in the same order and with the same content.
It also synchronizes fields that can pass through the conversion, including the
request model, response ID and model, same-provider opaque reasoning state, and
terminal usage. Translated Messages responses use the target default service tier
``standard`` because OpenAI tier labels have no faithful Anthropic equivalent.
Responses normal completion maps to Messages ``end_turn``; an incomplete
output-token limit maps to ``max_tokens``. A Responses content-filter limit is
coerced to ``end_turn`` because Messages has no corresponding stop reason.
Matched reasoning requests use adaptive summarized thinking and carry normalized
reasoning effort in ``output_config.effort``; reasoning history defaults that
effort to ``high`` when the canonical request omits it.

The matched fixtures assume OpenAI serves both the Responses and Messages
endpoints. Under that same-provider contract, Responses ``encrypted_content``
and Messages thinking ``signature`` carry the same OpenAI-issued opaque state.
This mapping must not be generalized to an Anthropic-issued signature.

The result remains an Anthropic Messages wire-format cassette. Responses-only
envelope fields such as item IDs, sequence numbers, and timestamps cannot be
encoded in Messages events; the converter must generate those fields
deterministically for parity. Messages-only lifecycle and transport events remain
native, and request fields other than the model and all transport metadata are
intentionally left unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from molto.utils.io import dump_yaml, json_dumps, json_loads, read_yaml


@dataclass
class _SSEEventUnit:
    entries: list[str]
    data_entry_index: int
    event: dict[str, Any]


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
            blocks.append({"type": "text", "text": text})
        else:
            raise ValueError("Responses message content cannot map to Messages.")
    return blocks


def _reasoning_content(item: dict[str, Any]) -> dict[str, str]:
    content = item.get("content")
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
        "thinking": "".join(summary_text or reasoning_text),
    }
    encrypted_content = item.get("encrypted_content")
    if encrypted_content is not None:
        if not isinstance(encrypted_content, str):
            raise ValueError("Responses encrypted reasoning content must be a string.")
        thinking["signature"] = encrypted_content
    return thinking


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
        if item.get("type") == "reasoning":
            pending_assistant_content.append(_reasoning_content(item))
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


def _matched_request_body(
    messages_request: dict[str, Any], responses_request: dict[str, Any]
) -> dict[str, Any]:
    request_model = responses_request.get("model")
    if not isinstance(request_model, str):
        raise ValueError("Responses request must contain a string model.")
    stream = responses_request.get("stream")
    if not isinstance(stream, bool):
        raise ValueError("Responses request must contain a boolean stream field.")

    matched_request = deepcopy(messages_request)
    matched_request["messages"] = _messages_from_response_input(responses_request.get("input"))
    matched_request["model"] = request_model
    matched_request["stream"] = stream
    if responses_request.get("max_output_tokens") is not None:
        matched_request["max_tokens"] = responses_request["max_output_tokens"]

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
    if not isinstance(incomplete_details, dict):
        raise ValueError("response.incomplete must contain incomplete_details.")
    reason = incomplete_details.get("reason")
    if reason == "max_output_tokens":
        return response, "max_tokens"
    return response, "end_turn"


def _serialize_data_event(event: dict[str, Any], template: str) -> str:
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
    return summary_deltas if summary_deltas is not None else reasoning_deltas


def _response_reasoning_encrypted_content(events: list[dict[str, Any]]) -> str | None:
    reasoning_items = [
        event.get("item")
        for event in events
        if event.get("type") == "response.output_item.done"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "reasoning"
    ]
    if len(reasoning_items) != 1:
        raise ValueError(
            "Expected exactly one completed Responses reasoning item, "
            f"found {len(reasoning_items)}."
        )

    encrypted_content = reasoning_items[0].get("encrypted_content")
    if encrypted_content is not None and not isinstance(encrypted_content, str):
        raise ValueError("Responses encrypted reasoning content must be a string.")
    return encrypted_content


def _content_block(units: list[_SSEEventUnit], block_type: str) -> tuple[int, int, int] | None:
    starts = [
        (position, unit.event)
        for position, unit in enumerate(units)
        if unit.event.get("type") == "content_block_start"
        and isinstance(unit.event.get("content_block"), dict)
        and unit.event["content_block"].get("type") == block_type
    ]
    if not starts:
        return None
    if len(starts) != 1:
        raise ValueError(
            f"Expected at most one Messages {block_type!r} content block, found {len(starts)}."
        )

    start_position, start_event = starts[0]
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
    return start_position, stop_positions[0], block_index


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

    block_start, block_stop, block_index = block
    delta_units = [
        (position, unit)
        for position, unit in enumerate(units)
        if block_start < position < block_stop
        and unit.event.get("type") == "content_block_delta"
        and unit.event.get("index") == block_index
        and isinstance(unit.event.get("delta"), dict)
        and unit.event["delta"].get("type") == delta_type
        and isinstance(unit.event["delta"].get(delta_field), str)
    ]
    if not delta_units and response_deltas:
        raise ValueError(f"Cannot insert {delta_type} events without a Messages template.")

    delta_positions = {position for position, _ in delta_units}
    last_delta_position = delta_units[-1][0] if delta_units else None
    next_delta = 0
    replacement_units: list[_SSEEventUnit] = []

    for position, unit in enumerate(units):
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

        if position == last_delta_position:
            while next_delta < len(response_deltas):
                template = deepcopy(delta_units[-1][1])
                template.event["delta"] = {
                    "type": delta_type,
                    delta_field: response_deltas[next_delta],
                }
                replacement_units.append(template)
                next_delta += 1

    units[:] = replacement_units


def _message_event(units: list[_SSEEventUnit], event_type: str) -> dict[str, Any]:
    matches = [unit.event for unit in units if unit.event.get("type") == event_type]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {event_type!r} event, found {len(matches)}.")
    return matches[0]


def _map_input_usage(response_usage: dict[str, Any], message_usage: dict[str, Any]) -> None:
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


def _map_usage(response_usage: dict[str, Any], message_usage: dict[str, Any]) -> None:
    _map_input_usage(response_usage, message_usage)

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


def match_messages_events(
    messages_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    """Rewrite Messages SSE events into the canonical input for Responses parity."""
    messages_request = _request_body(messages_cassette)
    responses_request = _request_body(responses_cassette)
    matched_request = _matched_request_body(messages_request, responses_request)

    prefix, message_units = _message_event_units(messages_cassette)
    response_events = [event for _, event in _data_events(responses_cassette)]

    created_event = _response_event(response_events, "response.created")
    created_response = created_event.get("response")
    terminal_response, stop_reason = _response_terminal(response_events)
    if not isinstance(created_response, dict):
        raise ValueError("Responses lifecycle events must contain response objects.")
    for field in ("id", "created_at", "model"):
        if created_response.get(field) != terminal_response.get(field):
            raise ValueError(f"Responses lifecycle events disagree on {field!r}.")

    response_usage = terminal_response.get("usage")
    if not isinstance(response_usage, dict):
        raise ValueError("Responses terminal event must contain usage for stream matching.")

    response_text_deltas = _response_text_deltas(response_events, "response.output_text")
    response_reasoning_deltas = _response_reasoning_deltas(response_events)
    _replace_content_deltas(
        message_units,
        response_reasoning_deltas,
        block_type="thinking",
        delta_type="thinking_delta",
        delta_field="thinking",
    )
    response_reasoning_encrypted_content = (
        None
        if response_reasoning_deltas is None
        else _response_reasoning_encrypted_content(response_events)
    )
    _replace_content_deltas(
        message_units,
        None
        if response_reasoning_deltas is None
        else (
            [response_reasoning_encrypted_content]
            if response_reasoning_encrypted_content is not None
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

    message_start = _message_event(message_units, "message_start").get("message")
    if not isinstance(message_start, dict):
        raise ValueError("message_start must contain a message object.")
    message_start["id"] = created_response["id"]
    message_start["model"] = terminal_response["model"]
    message_start["stop_reason"] = None
    message_start["stop_sequence"] = None

    message_start_usage = message_start.get("usage")
    if not isinstance(message_start_usage, dict):
        raise ValueError("message_start.message must contain usage.")
    _map_input_usage(response_usage, message_start_usage)
    message_start_usage["output_tokens"] = 0
    message_start_usage["service_tier"] = "standard"

    message_delta = _message_event(message_units, "message_delta")
    message_delta_value = message_delta.get("delta")
    if not isinstance(message_delta_value, dict):
        raise ValueError("message_delta must contain a delta object.")
    message_delta_value["stop_reason"] = stop_reason
    message_delta_value["stop_sequence"] = None

    message_delta_usage = message_delta.get("usage")
    if not isinstance(message_delta_usage, dict):
        raise ValueError("message_delta must contain usage.")
    _map_usage(response_usage, message_delta_usage)

    messages_cassette["request"]["body"] = matched_request
    _write_message_event_units(messages_cassette, prefix, message_units)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an Anthropic Messages input cassette for exact "
            "Messages-to-Responses stream parity. The Messages cassette is updated in "
            "place unless --output is provided."
        )
    )
    parser.add_argument("messages_cassette", type=Path)
    parser.add_argument("responses_cassette", type=Path)
    parser.add_argument("--output", type=Path, help="Write to a different YAML file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    messages_cassette = read_yaml(str(args.messages_cassette))
    responses_cassette = read_yaml(str(args.responses_cassette))
    if not isinstance(messages_cassette, dict) or not isinstance(responses_cassette, dict):
        raise ValueError("Cassette roots must be YAML mappings.")

    match_messages_events(messages_cassette, responses_cassette)
    output_path = args.output or args.messages_cassette
    dump_yaml(messages_cassette, str(output_path), sort_keys=False)
    print(f"Matched Anthropic Messages events and wrote {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
