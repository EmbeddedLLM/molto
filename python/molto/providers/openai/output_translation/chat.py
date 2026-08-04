"""Translate OpenAI Chat Completions payloads into semantic output events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from molto.types.ir.base import ItemStatus
from molto.types.ir.content import OutputTextPart, RefusalPart, UrlCitation
from molto.types.ir.emitter import OutputEventEmitter
from molto.types.ir.events import OutputEvent, ResponseMetadata
from molto.types.ir.output_items import AssistantMessageItem, FunctionCallItem, GeneratedAudio
from molto.utils.exceptions import BadInputError

from ..usage import usage_from_openai
from .stream import openai_stream_frame


def _metadata(payload: dict[str, Any], choice: dict[str, Any] | None = None) -> ResponseMetadata:
    return ResponseMetadata(
        model=str(payload.get("model") or ""),
        created_at=payload.get("created"),
        object=payload.get("object", "chat.completion"),
        choice_index=int((choice or {}).get("index", 0)),
        finish_reason=(choice or {}).get("finish_reason"),
        service_tier=payload.get("service_tier"),
        service_tier_present="service_tier" in payload,
        system_fingerprint=payload.get("system_fingerprint"),
        logprobs_present="logprobs" in (choice or {}),
    )


def _message_annotations(message: dict[str, Any]) -> tuple[UrlCitation, ...] | None:
    if "annotations" not in message:
        return None
    raw_annotations = message["annotations"]
    if raw_annotations is None:
        return None
    if not isinstance(raw_annotations, list):
        raise BadInputError("OpenAI Chat annotations must be a list.")
    annotations = []
    for raw_annotation in raw_annotations:
        if not isinstance(raw_annotation, dict) or raw_annotation.get("type") != "url_citation":
            raise BadInputError("OpenAI Chat supports only URL-citation annotations.")
        citation = raw_annotation.get("url_citation")
        if not isinstance(citation, dict):
            raise BadInputError("OpenAI Chat URL citations require url_citation data.")
        annotations.append(UrlCitation.model_validate({"type": "url_citation", **citation}))
    return tuple(annotations)


def _message_part(
    message: dict[str, Any], annotations: tuple[UrlCitation, ...] | None
) -> OutputTextPart | RefusalPart | None:
    content = message.get("content")
    refusal = message.get("refusal")
    if content is not None and refusal is not None:
        raise BadInputError("A Chat message cannot contain both content and refusal.")
    if content is not None:
        return OutputTextPart(text=str(content), annotations=annotations)
    if annotations:
        raise BadInputError("OpenAI Chat citations require assistant text.")
    if refusal is not None:
        return RefusalPart(refusal=str(refusal))
    return None


def _message_audio(message: dict[str, Any]) -> GeneratedAudio | None:
    payload = message.get("audio")
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "id",
        "data",
        "expires_at",
        "transcript",
    }:
        raise BadInputError(
            "OpenAI Chat generated audio requires id, data, expires_at, and transcript."
        )
    return GeneratedAudio.model_validate(payload)


def _response_events(payload: dict[str, Any]) -> list[OutputEvent]:
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    response_id = str(payload.get("id") or "chat_response")
    metadata = _metadata(payload, choice)
    if choice.get("logprobs") is not None:
        raise BadInputError("Chat logprobs require an output field outside the IR.")
    emitter = OutputEventEmitter(response_id, metadata=metadata)
    events: list[OutputEvent] = [emitter.started()]

    annotations = _message_annotations(message)
    part = _message_part(message, annotations)
    audio = _message_audio(message)
    item_index = 0
    if (
        part is not None
        or audio is not None
        or "annotations" in message
        or not message.get("tool_calls")
    ):
        item = AssistantMessageItem(
            id=f"{response_id}_message",
            status=ItemStatus.IN_PROGRESS,
            content=(part,) if part is not None else (),
            audio=audio,
            annotations=annotations,
        )
        events.append(emitter.item_added(item_index, item))
        if part is not None:
            empty_part = (
                OutputTextPart(text="", annotations=part.annotations)
                if isinstance(part, OutputTextPart)
                else RefusalPart(refusal="")
            )
            events.extend(
                emitter.replay_part(
                    item.id,
                    item_index,
                    0,
                    part,
                    part.text if isinstance(part, OutputTextPart) else part.refusal,
                    empty_part=empty_part,
                )
            )
        events.append(
            emitter.item_done(
                item_index,
                item.model_copy(update={"status": ItemStatus.COMPLETED}),
            )
        )
        item_index += 1

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        call_id = str(tool_call.get("id") or f"call_{item_index}")
        arguments = str(function.get("arguments") or "")
        item = FunctionCallItem(
            id=f"{response_id}_{call_id}",
            status=ItemStatus.IN_PROGRESS,
            call_id=call_id,
            name=str(function.get("name") or ""),
            arguments="",
        )
        events.append(emitter.item_added(item_index, item))
        if arguments:
            events.append(emitter.function_arguments_delta(item.id, item_index, arguments))
        events.append(emitter.function_arguments_done(item.id, item_index, arguments))
        events.append(
            emitter.item_done(
                item_index,
                item.model_copy(update={"status": ItemStatus.COMPLETED, "arguments": arguments}),
            )
        )
        item_index += 1

    usage = usage_from_openai(payload.get("usage"), "chat_completions")
    finish_reason = choice.get("finish_reason")
    if finish_reason in {"length", "content_filter"}:
        incomplete_reason = "max_output_tokens" if finish_reason == "length" else "content_filter"
        events.append(emitter.incomplete(incomplete_reason, usage))
    else:
        events.append(emitter.completed(usage))
    return events


async def chat_response_to_output_events(payload: dict[str, Any]) -> AsyncIterator[OutputEvent]:
    """Translate one non-stream Chat response."""

    for event in _response_events(payload):
        yield event


def _combine_stream(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        raise BadInputError("OpenAI Chat stream returned no events.")
    first = chunks[0]
    choice: dict[str, Any] = {"index": 0, "message": {}}
    content: list[str] = []
    refusal: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    audio: dict[str, Any] = {}
    finish_reason = None
    usage = None
    for chunk in chunks:
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for raw_choice in chunk.get("choices") or []:
            finish_reason = raw_choice.get("finish_reason") or finish_reason
            delta = raw_choice.get("delta") or {}
            if delta.get("content") is not None:
                content.append(str(delta["content"]))
            if delta.get("refusal") is not None:
                refusal.append(str(delta["refusal"]))
            if delta.get("audio") is not None:
                raw_audio = delta["audio"]
                for key, value in raw_audio.items():
                    if value is None:
                        continue
                    if key in {"data", "transcript"}:
                        audio[key] = audio.get(key, "") + str(value)
                    else:
                        audio[key] = value
            for raw_call in delta.get("tool_calls") or []:
                index = int(raw_call.get("index", 0))
                call = tool_calls.setdefault(
                    index, {"id": None, "type": "function", "function": {}}
                )
                call["id"] = raw_call.get("id") or call["id"]
                function = raw_call.get("function") or {}
                call["function"]["name"] = function.get("name") or call["function"].get("name")
                call["function"]["arguments"] = call["function"].get("arguments", "") + str(
                    function.get("arguments") or ""
                )
    choice["message"] = {
        "role": "assistant",
        "content": "".join(content) if content else None,
        "refusal": "".join(refusal) if refusal else None,
        "tool_calls": list(tool_calls.values()),
        "audio": audio or None,
    }
    choice["finish_reason"] = finish_reason
    return {
        "id": first.get("id"),
        "model": first.get("model"),
        "created": first.get("created"),
        "object": "chat.completion",
        "choices": [choice],
        "usage": usage,
        "service_tier": first.get("service_tier"),
        "system_fingerprint": first.get("system_fingerprint"),
    }


async def chat_stream_to_output_events(
    payloads: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[OutputEvent]:
    """Translate a Chat stream into the same semantic event sequence."""

    chunks: list[dict[str, Any]] = []
    response_id: str | None = None
    async for payload in payloads:
        chunks.append(payload)
        if response_id is None:
            response_id = str(payload.get("id") or "chat_response")
            choices = payload.get("choices") or []
            choice = choices[0] if choices else None
            yield OutputEventEmitter(
                response_id,
                metadata=_metadata(payload, choice),
            ).started()
        yield openai_stream_frame(
            response_id,
            len(chunks),
            payload,
            "chat_completions",
        )
    if not chunks:
        raise BadInputError("OpenAI Chat stream returned no events.")
    semantic_events = _response_events(_combine_stream(chunks))
    for index, event in enumerate(semantic_events[1:], start=len(chunks) + 1):
        yield event.model_copy(update={"sequence_number": index})
