"""Translate OpenAI Responses payloads into semantic output events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from molto.types.ir.base import ItemStatus
from molto.types.ir.content import (
    EncryptedReasoningPart,
    OutputTextPart,
    ReasoningContentPart,
    ReasoningSummaryPart,
    RefusalPart,
    UrlCitation,
)
from molto.types.ir.emitter import OutputEventEmitter
from molto.types.ir.errors import ErrorDetail
from molto.types.ir.events import OutputEvent, ResponseMetadata
from molto.types.ir.output_items import (
    AssistantMessageItem,
    FunctionCallItem,
    OpenAIHostedToolItem,
    ReasoningItem,
)
from molto.utils.exceptions import BadInputError

from ..usage import usage_from_openai
from .stream import attach_openai_response_snapshot, openai_stream_frame


def _metadata(payload: dict[str, Any]) -> ResponseMetadata:
    return ResponseMetadata(
        model=str(payload.get("model") or ""),
        created_at=payload.get("created_at"),
        object=payload.get("object", "response"),
        service_tier=payload.get("service_tier"),
        service_tier_present="service_tier" in payload,
    )


def _status(value: str | None, *, default: ItemStatus = ItemStatus.COMPLETED) -> ItemStatus:
    if value is None:
        return default
    try:
        return ItemStatus(value)
    except ValueError as exc:
        raise BadInputError(f"Unsupported OpenAI Responses item status {value!r}.") from exc


def _message_parts(raw_item: dict[str, Any]):
    parts = []
    for raw_part in raw_item.get("content") or []:
        part_type = raw_part.get("type")
        if part_type == "output_text":
            if raw_part.get("logprobs"):
                raise BadInputError("Responses logprobs are outside the IR.")
            annotations = None
            if "annotations" in raw_part:
                raw_annotations = raw_part["annotations"] or []
                if not isinstance(raw_annotations, list):
                    raise BadInputError("OpenAI Responses annotations must be a list.")
                annotations = tuple(
                    UrlCitation.model_validate(annotation) for annotation in raw_annotations
                )
            parts.append(
                OutputTextPart(text=str(raw_part.get("text") or ""), annotations=annotations)
            )
        elif part_type == "refusal":
            parts.append(RefusalPart(refusal=str(raw_part.get("refusal") or "")))
        else:
            raise BadInputError(f"Unsupported OpenAI Responses message part {part_type!r}.")
    return tuple(parts)


def _reasoning_parts(raw_item: dict[str, Any]):
    parts = []
    for raw_part in raw_item.get("summary") or []:
        parts.append(ReasoningSummaryPart(text=str(raw_part.get("text") or "")))
    for raw_part in raw_item.get("content") or []:
        parts.append(ReasoningContentPart(text=str(raw_part.get("text") or ""), provider="openai"))
    encrypted = raw_item.get("encrypted_content")
    if encrypted is not None:
        parts.append(EncryptedReasoningPart(data=str(encrypted), provider="openai"))
    return tuple(parts)


def _response_events(payload: dict[str, Any]) -> list[OutputEvent]:
    response_id = str(payload.get("id") or "responses_response")
    metadata = _metadata(payload)
    emitter = OutputEventEmitter(response_id, metadata=metadata)
    events: list[OutputEvent] = [emitter.started()]
    item_index = 0

    for raw_item in payload.get("output") or []:
        item_type = raw_item.get("type")
        item_id = str(raw_item.get("id") or f"{response_id}_item_{item_index}")
        hosted_tool_item = item_type in {
            "code_interpreter_call",
            "image_generation_call",
            "mcp_approval_request",
            "mcp_call",
            "web_search_call",
        }
        item_status = (
            ItemStatus.COMPLETED
            if hosted_tool_item
            else _status(raw_item.get("status"), default=ItemStatus.COMPLETED)
        )
        if item_type == "message":
            parts = _message_parts(raw_item)
            item = AssistantMessageItem(
                id=item_id,
                status=ItemStatus.IN_PROGRESS,
                content=parts,
                phase=raw_item.get("phase"),
            )
            events.append(emitter.item_added(item_index, item))
            for content_index, part in enumerate(parts):
                empty_part = (
                    OutputTextPart(text="", annotations=part.annotations)
                    if isinstance(part, OutputTextPart)
                    else RefusalPart(refusal="")
                )
                events.extend(
                    emitter.replay_part(
                        item_id,
                        item_index,
                        content_index,
                        part,
                        part.text if isinstance(part, OutputTextPart) else part.refusal,
                        empty_part=empty_part,
                    )
                )
            events.append(
                emitter.item_done(
                    item_index,
                    item.model_copy(update={"status": item_status}),
                )
            )
        elif item_type == "function_call":
            arguments = str(raw_item.get("arguments") or "")
            item = FunctionCallItem(
                id=item_id,
                status=ItemStatus.IN_PROGRESS,
                call_id=str(raw_item.get("call_id") or item_id),
                name=str(raw_item.get("name") or ""),
                arguments="",
            )
            events.append(emitter.item_added(item_index, item))
            if arguments:
                events.append(emitter.function_arguments_delta(item_id, item_index, arguments))
            events.append(emitter.function_arguments_done(item_id, item_index, arguments))
            events.append(
                emitter.item_done(
                    item_index,
                    item.model_copy(update={"status": item_status, "arguments": arguments}),
                )
            )
        elif item_type == "reasoning":
            parts = _reasoning_parts(raw_item)
            item = ReasoningItem(id=item_id, status=ItemStatus.IN_PROGRESS, parts=parts)
            events.append(emitter.item_added(item_index, item))
            for content_index, part in enumerate(parts):
                if isinstance(part, ReasoningSummaryPart):
                    empty_part = ReasoningSummaryPart(text="")
                elif isinstance(part, ReasoningContentPart):
                    empty_part = ReasoningContentPart(text="", provider="openai")
                else:
                    empty_part = EncryptedReasoningPart(data="", provider="openai")
                events.extend(
                    emitter.replay_part(
                        item_id,
                        item_index,
                        content_index,
                        part,
                        getattr(part, part.value_field),
                        empty_part=empty_part,
                    )
                )
            events.append(
                emitter.item_done(item_index, item.model_copy(update={"status": item_status}))
            )
        elif hosted_tool_item:
            raw_status = raw_item.get("status")
            payload_json = json.dumps(raw_item, separators=(",", ":"), ensure_ascii=False)
            item = OpenAIHostedToolItem(
                id=item_id,
                status=ItemStatus.IN_PROGRESS,
                item_type=str(item_type),
                openai_status=str(raw_status) if raw_status is not None else None,
                payload_json=payload_json,
            )
            events.append(emitter.item_added(item_index, item))
            events.append(
                emitter.item_done(
                    item_index,
                    item.model_copy(update={"status": ItemStatus.COMPLETED}),
                )
            )
        else:
            raise BadInputError(f"Unsupported OpenAI Responses output item {item_type!r}.")
        item_index += 1

    usage = usage_from_openai(payload.get("usage"), "responses")
    status = payload.get("status", "completed")
    if status == "failed":
        raw_error = payload.get("error") or {}
        events.append(
            emitter.failed(
                ErrorDetail(
                    code=str(raw_error.get("code") or "provider_error"),
                    message=str(raw_error.get("message") or "OpenAI Responses request failed."),
                    provider="openai",
                ),
                usage,
            )
        )
    elif status == "incomplete":
        reason = str((payload.get("incomplete_details") or {}).get("reason") or "unknown")
        events.append(emitter.incomplete(reason, usage))
    else:
        events.append(emitter.completed(usage))
    return events


async def responses_response_to_output_events(
    payload: dict[str, Any],
) -> AsyncIterator[OutputEvent]:
    """Translate one non-stream Responses response."""

    semantic_events = _response_events(payload)
    for event in attach_openai_response_snapshot(semantic_events, payload, "responses"):
        yield event


async def responses_stream_to_output_events(
    payloads: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[OutputEvent]:
    """Translate a Responses stream using its terminal OpenAI response snapshot."""

    payload_values: list[dict[str, Any]] = []
    response: dict[str, Any] | None = None
    response_id: str | None = None
    async for payload in payloads:
        payload_values.append(payload)
        if response_id is None:
            candidate = payload.get("response")
            if isinstance(candidate, dict):
                response_id = str(candidate.get("id") or "responses_response")
                yield OutputEventEmitter(
                    response_id,
                    metadata=_metadata(candidate),
                ).started()
            else:
                response_id = str(payload.get("id") or "responses_response")
                yield OutputEventEmitter(response_id).started()
        yield openai_stream_frame(
            response_id,
            len(payload_values),
            payload,
            "responses",
        )
        if isinstance(payload.get("response"), dict):
            response = payload["response"]
    if response is None:
        raise BadInputError("OpenAI Responses stream did not contain a terminal response.")
    semantic_events = _response_events(response)
    for index, event in enumerate(semantic_events[1:], start=len(payload_values) + 1):
        yield event.model_copy(update={"sequence_number": index})
