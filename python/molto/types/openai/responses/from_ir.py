"""Compose OpenAI Responses models from portable output IR."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator

from molto.providers.openai.usage import usage_to_openai
from molto.types.ir.events import (
    OutputEvent,
    OutputEventSequence,
    PreservedOpenAIResponseEvent,
    PreservedOpenAIStreamEvent,
)
from molto.types.ir.output_items import AssistantMessageItem, FunctionCallItem, ReasoningItem
from molto.types.ir.portable import (
    EncryptedReasoningPart,
    OutputIR,
    OutputTextPart,
    ReasoningContentPart,
    ReasoningSummaryPart,
    RefusalPart,
)

from .models_response import (
    OpenAIResponsesResponse,
    OpenAIResponsesStreamEvent,
    parse_stream_event,
)


def _output_item(item) -> dict:
    if isinstance(item, AssistantMessageItem):
        content = []
        for part in item.content:
            if isinstance(part, OutputTextPart):
                content.append(
                    {
                        "type": "output_text",
                        "text": part.text,
                        "annotations": [
                            annotation.model_dump(mode="python")
                            for annotation in (
                                part.annotations
                                if part.annotations is not None
                                else item.annotations or ()
                            )
                        ],
                        "logprobs": [],
                    }
                )
            elif isinstance(part, RefusalPart):
                content.append({"type": "refusal", "refusal": part.refusal})
        return {
            "type": "message",
            "id": item.id,
            "status": item.status.value,
            "role": "assistant",
            "content": content,
            "phase": item.phase,
        }
    if isinstance(item, FunctionCallItem):
        return {
            "type": "function_call",
            "id": item.id,
            "status": item.status.value,
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
        }
    if isinstance(item, ReasoningItem):
        summary = []
        content = []
        encrypted = None
        for part in item.parts:
            if isinstance(part, ReasoningSummaryPart):
                summary.append({"type": "summary_text", "text": part.text})
            elif isinstance(part, ReasoningContentPart):
                content.append({"type": "reasoning_text", "text": part.text})
            elif isinstance(part, EncryptedReasoningPart):
                encrypted = part.data
        return {
            "type": "reasoning",
            "id": item.id,
            "status": item.status.value,
            "summary": summary,
            "content": content,
            "encrypted_content": encrypted,
        }
    raise ValueError(f"Unsupported OpenAI output item {type(item).__name__}.")


def _response_values(output: OutputIR) -> dict:
    metadata = {}
    if output.metadata.system_fingerprint is not None:
        metadata["system_fingerprint"] = output.metadata.system_fingerprint
    values = {
        "id": output.id,
        "object": "response",
        "created_at": output.metadata.created_at or 0,
        "model": output.metadata.model,
        "status": output.status.value,
        "output": [_output_item(item) for item in output.output.values],
        "usage": usage_to_openai(output.usage, "responses"),
        "service_tier": output.metadata.service_tier or "default",
        "metadata": metadata,
    }
    if output.error is not None:
        values["error"] = {
            "code": output.error.code,
            "message": output.error.message,
        }
    if output.incomplete_reason is not None:
        values["incomplete_details"] = {"reason": output.incomplete_reason}
    return values


def from_output_ir(output: OutputIR) -> OpenAIResponsesResponse:
    """Compose one OpenAI Responses response from semantic output."""

    snapshot = next(
        (
            event
            for event in output.event_sequence.events
            if isinstance(event, PreservedOpenAIResponseEvent) and event.endpoint == "responses"
        ),
        None,
    )
    if snapshot is not None:
        return OpenAIResponsesResponse.model_validate(json.loads(snapshot.payload_json))
    return OpenAIResponsesResponse.model_validate(_response_values(output))


async def from_output_event_stream(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OpenAIResponsesStreamEvent]:
    """Compose a Responses stream, forwarding matching native frames immediately."""

    event_values: list[OutputEvent] = []
    matching_frames = False
    async for event in events:
        if isinstance(event, PreservedOpenAIStreamEvent) and event.endpoint == "responses":
            matching_frames = True
            yield parse_stream_event(json.loads(event.payload_json))
            continue
        event_values.append(event)
    if matching_frames:
        return

    sequence = OutputEventSequence(events=tuple(event_values))
    output = OutputIR.from_output_events(sequence)
    response = _response_values(output)
    event_type = {
        "completed": "response.completed",
        "incomplete": "response.incomplete",
        "failed": "response.failed",
    }[output.status.value]
    yield parse_stream_event(
        {
            "type": event_type,
            "sequence_number": sequence.terminal.sequence_number,
            "response": response,
        }
    )
