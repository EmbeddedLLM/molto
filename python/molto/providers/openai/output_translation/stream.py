"""Preserve exact frames from OpenAI-compatible streaming endpoints."""

from __future__ import annotations

import json
from typing import Any, Literal

from molto.types.ir.events import (
    OutputEvent,
    PreservedOpenAIResponseEvent,
    PreservedOpenAIStreamEvent,
)
from molto.utils.exceptions import BadInputError


def openai_stream_frame(
    response_id: str,
    sequence_number: int,
    payload: dict[str, Any],
    endpoint: Literal["chat_completions", "responses"],
) -> PreservedOpenAIStreamEvent:
    """Create one preserved stream frame without waiting for the stream terminal."""

    event_type = payload.get("type")
    if not isinstance(event_type, str):
        if endpoint == "chat_completions" and payload.get("object") == "chat.completion.chunk":
            event_type = "chat.completion.chunk"
        else:
            raise BadInputError("OpenAI stream frame is missing its event type.")
    return PreservedOpenAIStreamEvent(
        response_id=response_id,
        sequence_number=sequence_number,
        endpoint=endpoint,
        event_type=event_type,
        payload_json=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )


def attach_openai_response_snapshot(
    semantic_events: list[OutputEvent],
    payload: dict[str, Any],
    endpoint: Literal["chat_completions", "responses"],
) -> list[OutputEvent]:
    """Insert one exact OpenAI response snapshot after response start."""

    if not semantic_events:
        raise BadInputError("OpenAI response produced no semantic events.")
    response_id = semantic_events[0].response_id
    events: list[OutputEvent] = [
        semantic_events[0].model_copy(update={"sequence_number": 0}),
        PreservedOpenAIResponseEvent(
            response_id=response_id,
            sequence_number=1,
            endpoint=endpoint,
            payload_json=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        ),
    ]
    events.extend(
        event.model_copy(update={"sequence_number": index + 2})
        for index, event in enumerate(semantic_events[1:])
    )
    return events
