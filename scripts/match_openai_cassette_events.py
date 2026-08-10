#!/usr/bin/env python3
"""Prepare a Chat Completions cassette for CC-to-Responses stream parity tests.

The Responses cassette is the canonical expected output. This script rewrites the
Chat Completions SSE stream so a converter receives one corresponding CC delta for
every ``response.output_text.delta`` and ``response.reasoning_text.delta`` event,
in the same order and with the same content. It also synchronizes fields that can
pass through the conversion, including the response ID, creation time, model,
service tier, and usage.

The result remains a Chat Completions wire-format cassette. Responses-only envelope
fields such as message item IDs, sequence numbers, and a distinct completion time
cannot be encoded in CC chunks; the converter must generate those fields
deterministically for exact output parity. Request and transport metadata are
intentionally left unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from molto.utils.io import dump_yaml, json_dumps, json_loads, read_yaml


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


def _serialize_data_event(event: dict[str, Any], template: str) -> str:
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


def _map_usage(response_usage: dict[str, Any], chat_usage: dict[str, Any]) -> None:
    chat_usage["prompt_tokens"] = response_usage["input_tokens"]
    chat_usage["completion_tokens"] = response_usage["output_tokens"]
    chat_usage["total_tokens"] = response_usage["total_tokens"]

    input_details = response_usage.get("input_tokens_details") or {}
    prompt_details = chat_usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        for field in ("cache_write_tokens", "cached_tokens"):
            if field in input_details:
                prompt_details[field] = input_details[field]

    output_details = response_usage.get("output_tokens_details") or {}
    completion_details = chat_usage.get("completion_tokens_details")
    if isinstance(completion_details, dict) and "reasoning_tokens" in output_details:
        completion_details["reasoning_tokens"] = output_details["reasoning_tokens"]


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


def match_chat_completions_events(
    chat_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    """Rewrite CC SSE events into the canonical input for Responses parity."""
    indexed_chat_events = _data_events(chat_cassette)
    response_events = [event for _, event in _data_events(responses_cassette)]

    created_event = _response_event(response_events, "response.created")
    completed_event = _response_event(response_events, "response.completed")
    created_response = created_event.get("response")
    completed_response = completed_event.get("response")
    if not isinstance(created_response, dict) or not isinstance(completed_response, dict):
        raise ValueError("Responses lifecycle events must contain response objects.")
    for field in ("id", "created_at", "model"):
        if created_response.get(field) != completed_response.get(field):
            raise ValueError(f"Responses lifecycle events disagree on {field!r}.")

    response_id = created_response["id"]
    created_at = created_response["created_at"]
    model = completed_response["model"]
    service_tier = completed_response.get("service_tier")
    response_usage = completed_response.get("usage")
    if not isinstance(response_usage, dict):
        raise ValueError("response.completed must contain usage for stream matching.")

    response_content_deltas = _response_text_deltas(response_events, "response.output_text")
    response_reasoning_deltas = _response_text_deltas(
        response_events, "response.reasoning_text", optional=True
    )

    usage_events = [
        usage
        for _, event in indexed_chat_events
        if isinstance((usage := event.get("usage")), dict)
    ]

    if len(usage_events) != 1:
        raise ValueError(
            f"Expected exactly one Chat Completions usage event, found {len(usage_events)}."
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
    usage_events = [
        usage
        for _, event in indexed_chat_events
        if isinstance((usage := event.get("usage")), dict)
    ]
    for _, event in indexed_chat_events:
        event["id"] = response_id
        event["created"] = created_at
        event["model"] = model
        if service_tier is not None:
            event["service_tier"] = service_tier
    _map_usage(response_usage, usage_events[0])

    sse_entries = chat_cassette["response"]["sse"]
    for index, event in indexed_chat_events:
        sse_entries[index] = _serialize_data_event(event, sse_entries[index])


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
    dump_yaml(chat_cassette, str(output_path), sort_keys=False)
    print(f"Matched Chat Completions events and wrote {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
