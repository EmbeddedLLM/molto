"""OpenAI-compatible cross-endpoint output adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

from molto.providers.openai.usage import usage_from_openai, usage_to_openai
from molto.types.ir.base import ResponseStatus
from molto.types.ir.events import (
    OutputEvent,
    OutputItemDoneEvent,
    PreservedOpenAIStreamEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
)
from molto.types.ir.output_items import AssistantMessageItem, OpenAIHostedToolItem
from molto.types.ir.portable import OutputIR
from molto.utils.exceptions import BadInputError

from .chat_completions.from_ir import (
    from_output_event_stream as chat_from_output_event_stream,
)
from .chat_completions.from_ir import (
    from_output_ir as chat_from_output_ir,
)
from .chat_completions.models_response import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionResponse,
)
from .responses.from_ir import (
    from_output_event_stream as responses_from_output_event_stream,
)
from .responses.from_ir import (
    from_output_ir as responses_from_output_ir,
)
from .responses.models_response import (
    OpenAIResponsesResponse,
    OpenAIResponsesStreamEvent,
    parse_stream_event,
)


@dataclass
class _ResponsesToChatState:
    response_id: str = "responses_response"
    model: str = ""
    created: int = 0
    service_tier: str | None = None
    role_sent: bool = False
    tool_calls: dict[int, dict[str, str | None]] | None = None

    def __post_init__(self) -> None:
        self.tool_calls = {}


@dataclass
class _ChatToResponsesState:
    response_id: str = "chat_response"
    model: str = ""
    created_at: int = 0
    service_tier: str = "default"
    system_fingerprint: str | None = None
    sequence_number: int = 0
    response_started: bool = False
    message_started: bool = False
    message_text: str = ""
    refusal: str = ""
    function_calls: dict[int, dict[str, Any]] = field(default_factory=dict)


def _chat_responses_response(
    state: _ChatToResponsesState,
    *,
    status: str,
    usage: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
) -> OpenAIResponsesResponse:
    output: list[dict[str, Any]] = []
    if state.message_started:
        content: list[dict[str, Any]] = []
        if state.message_text:
            content.append(
                {
                    "type": "output_text",
                    "text": state.message_text,
                    "annotations": [],
                    "logprobs": [],
                }
            )
        elif state.refusal:
            content.append({"type": "refusal", "refusal": state.refusal})
        output.append(
            {
                "type": "message",
                "id": f"{state.response_id}_message",
                "status": "completed" if status != "in_progress" else "in_progress",
                "role": "assistant",
                "content": content,
            }
        )
    for call in sorted(state.function_calls.values(), key=lambda call: call["output_index"]):
        output.append(
            {
                "type": "function_call",
                "id": call["item_id"],
                "status": "completed" if status != "in_progress" else "in_progress",
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": call["arguments"],
            }
        )
    values: dict[str, Any] = {
        "id": state.response_id,
        "object": "response",
        "created_at": state.created_at,
        "model": state.model,
        "status": status,
        "output": output,
        "service_tier": state.service_tier,
        "metadata": (
            {"system_fingerprint": state.system_fingerprint}
            if state.system_fingerprint is not None
            else {}
        ),
        "usage": usage,
    }
    if incomplete_reason is not None:
        values["incomplete_details"] = {"reason": incomplete_reason}
    return OpenAIResponsesResponse.model_validate(values)


def _chat_responses_event(
    state: _ChatToResponsesState,
    event_type: str,
    **values: Any,
) -> OpenAIResponsesStreamEvent:
    event = parse_stream_event(
        {"type": event_type, "sequence_number": state.sequence_number, **values}
    )
    state.sequence_number += 1
    return event


def _update_chat_responses_state(
    state: _ChatToResponsesState,
    payload: dict[str, Any],
) -> None:
    state.response_id = str(payload.get("id") or state.response_id)
    state.model = str(payload.get("model") or state.model)
    state.created_at = int(payload.get("created") or state.created_at)
    state.service_tier = str(payload.get("service_tier") or state.service_tier)
    fingerprint = payload.get("system_fingerprint")
    if fingerprint is not None:
        state.system_fingerprint = str(fingerprint)


def _chat_frame_to_responses(
    state: _ChatToResponsesState,
    event: PreservedOpenAIStreamEvent,
) -> list[OpenAIResponsesStreamEvent]:
    payload = json.loads(event.payload_json)
    _update_chat_responses_state(state, payload)
    events: list[OpenAIResponsesStreamEvent] = []
    if not state.response_started:
        state.response_started = True
        response = _chat_responses_response(state, status="in_progress")
        events.extend(
            [
                _chat_responses_event(state, "response.created", response=response),
                _chat_responses_event(state, "response.in_progress", response=response),
            ]
        )

    for choice in payload.get("choices") or []:
        if int(choice.get("index", 0)) != 0:
            raise BadInputError("Responses output cannot represent multiple Chat choices.")
        delta = choice.get("delta") or {}
        if delta.get("audio") is not None:
            raise BadInputError("Responses output cannot represent generated Chat audio.")
        content = delta.get("content")
        refusal = delta.get("refusal")
        if content is not None or refusal is not None:
            if not state.message_started:
                state.message_started = True
                message = _chat_responses_response(state, status="in_progress").output[0]
                events.extend(
                    [
                        _chat_responses_event(
                            state,
                            "response.output_item.added",
                            output_index=0,
                            item=message,
                        ),
                        _chat_responses_event(
                            state,
                            "response.content_part.added",
                            item_id=message.id,
                            output_index=0,
                            content_index=0,
                            part={
                                "type": "output_text" if content is not None else "refusal",
                                "text": "",
                                "annotations": [],
                                "logprobs": [],
                            }
                            if content is not None
                            else {"type": "refusal", "refusal": ""},
                        ),
                    ]
                )
            if content is not None:
                text = str(content)
                state.message_text += text
                events.append(
                    _chat_responses_event(
                        state,
                        "response.output_text.delta",
                        item_id=f"{state.response_id}_message",
                        output_index=0,
                        content_index=0,
                        delta=text,
                    )
                )

        for raw_call in delta.get("tool_calls") or []:
            call_index = int(raw_call.get("index", 0))
            call = state.function_calls.get(call_index)
            function = raw_call.get("function") or {}
            if call is None:
                output_index = call_index + (1 if state.message_started else 0)
                call_id = str(raw_call.get("id") or f"call_{call_index}")
                call = {
                    "item_id": f"{state.response_id}_{call_id}",
                    "output_index": output_index,
                    "call_id": call_id,
                    "name": str(function.get("name") or ""),
                    "arguments": "",
                }
                state.function_calls[call_index] = call
                events.append(
                    _chat_responses_event(
                        state,
                        "response.output_item.added",
                        output_index=output_index,
                        item={
                            "type": "function_call",
                            "id": call["item_id"],
                            "status": "in_progress",
                            "call_id": call["call_id"],
                            "name": call["name"],
                            "arguments": "",
                        },
                    )
                )
            else:
                if raw_call.get("id") is not None:
                    call["call_id"] = str(raw_call["id"])
                if function.get("name") is not None:
                    call["name"] = str(function["name"])
            arguments = function.get("arguments")
            if arguments is not None:
                delta_arguments = str(arguments)
                call["arguments"] += delta_arguments
                events.append(
                    _chat_responses_event(
                        state,
                        "response.function_call_arguments.delta",
                        item_id=call["item_id"],
                        output_index=call["output_index"],
                        delta=delta_arguments,
                    )
                )
            if refusal is not None:
                text = str(refusal)
                state.refusal += text
                events.append(
                    _chat_responses_event(
                        state,
                        "response.refusal.delta",
                        item_id=f"{state.response_id}_message",
                        output_index=0,
                        content_index=0,
                        delta=text,
                    )
                )

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            if not state.message_started and finish_reason != "tool_calls":
                state.message_started = True
                message = _chat_responses_response(state, status="in_progress").output[0]
                events.append(
                    _chat_responses_event(
                        state,
                        "response.output_item.added",
                        output_index=0,
                        item=message,
                    )
                )
            usage = payload.get("usage")
            if usage is not None:
                usage = usage_to_openai(usage_from_openai(usage, "chat_completions"), "responses")
            incomplete_reason = {
                "length": "max_output_tokens",
                "content_filter": "content_filter",
            }.get(finish_reason)
            status = "incomplete" if incomplete_reason is not None else "completed"
            response = _chat_responses_response(
                state,
                status=status,
                usage=usage,
                incomplete_reason=incomplete_reason,
            )
            if state.message_started:
                message = response.output[0]
                if state.message_text:
                    events.append(
                        _chat_responses_event(
                            state,
                            "response.output_text.done",
                            item_id=message.id,
                            output_index=0,
                            content_index=0,
                            text=state.message_text,
                        )
                    )
                    events.append(
                        _chat_responses_event(
                            state,
                            "response.content_part.done",
                            item_id=message.id,
                            output_index=0,
                            content_index=0,
                            part=message.content[0],
                        )
                    )
                elif state.refusal:
                    events.append(
                        _chat_responses_event(
                            state,
                            "response.refusal.done",
                            item_id=message.id,
                            output_index=0,
                            content_index=0,
                            refusal=state.refusal,
                        )
                    )
                    events.append(
                        _chat_responses_event(
                            state,
                            "response.content_part.done",
                            item_id=message.id,
                            output_index=0,
                            content_index=0,
                            part=message.content[0],
                        )
                    )
                events.append(
                    _chat_responses_event(
                        state,
                        "response.output_item.done",
                        output_index=0,
                        item=message,
                    )
                )
            for call in state.function_calls.values():
                function_item = next(
                    item for item in response.output if item.id == call["item_id"]
                )
                events.extend(
                    [
                        _chat_responses_event(
                            state,
                            "response.function_call_arguments.done",
                            item_id=call["item_id"],
                            output_index=call["output_index"],
                            arguments=call["arguments"],
                        ),
                        _chat_responses_event(
                            state,
                            "response.output_item.done",
                            output_index=call["output_index"],
                            item=function_item,
                        ),
                    ]
                )
            events.append(
                _chat_responses_event(
                    state,
                    "response.incomplete"
                    if incomplete_reason is not None
                    else "response.completed",
                    response=response,
                )
            )
    return events


def _responses_chat_chunk(
    state: _ResponsesToChatState,
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> OpenAIChatCompletionChunk:
    values: dict[str, Any] = {
        "id": state.response_id,
        "created": state.created,
        "model": state.model,
        "object": "chat.completion.chunk",
        "service_tier": state.service_tier,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        values["usage"] = usage
    return OpenAIChatCompletionChunk.model_validate(values)


def _chat_chunk_without_terminal_usage(
    chunk: OpenAIChatCompletionChunk,
) -> OpenAIChatCompletionChunk:
    if not chunk.choices or chunk.choices[0].finish_reason is None:
        return chunk
    return chunk.model_copy(update={"usage": None})


def _update_responses_chat_state(
    state: _ResponsesToChatState,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    state.response_id = str(response.get("id") or state.response_id)
    state.model = str(response.get("model") or state.model)
    state.created = int(response.get("created_at") or state.created)
    state.service_tier = response.get("service_tier") or state.service_tier
    return response


def _responses_frame_to_chat(
    state: _ResponsesToChatState,
    event: PreservedOpenAIStreamEvent,
) -> OpenAIChatCompletionChunk | None:
    payload = json.loads(event.payload_json)
    response = _update_responses_chat_state(state, payload)
    event_type = payload.get("type")
    if event_type == "response.output_item.added":
        item = payload.get("item") or {}
        item_type = item.get("type")
        output_index = int(payload.get("output_index", 0))
        if item_type == "function_call":
            call = {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "name": str(item.get("name") or ""),
            }
            state.tool_calls[output_index] = call
            return _responses_chat_chunk(
                state,
                delta={
                    "tool_calls": [
                        {
                            "index": output_index,
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"]},
                        }
                    ]
                },
            )
        if item_type not in {"message", "reasoning"}:
            raise BadInputError(
                f"Chat Completions output cannot represent Responses item {item_type!r}."
            )
        return None
    if event_type == "response.output_text.delta":
        delta = {"content": str(payload.get("delta") or "")}
        if not state.role_sent:
            delta["role"] = "assistant"
            state.role_sent = True
        return _responses_chat_chunk(state, delta=delta)
    if event_type == "response.refusal.delta":
        return _responses_chat_chunk(state, delta={"refusal": str(payload.get("delta") or "")})
    if event_type == "response.function_call_arguments.delta":
        output_index = int(payload.get("output_index", 0))
        call = state.tool_calls.get(output_index, {})
        return _responses_chat_chunk(
            state,
            delta={
                "tool_calls": [
                    {
                        "index": output_index,
                        "id": call.get("id"),
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": str(payload.get("delta") or ""),
                        },
                    }
                ]
            },
        )
    if event_type in {"response.completed", "response.incomplete"}:
        response = response or {}
        reason = (response.get("incomplete_details") or {}).get("reason")
        if event_type == "response.incomplete":
            finish_reason = {
                "max_output_tokens": "length",
                "content_filter": "content_filter",
            }.get(
                reason,
                "length",
            )
        else:
            finish_reason = "tool_calls" if state.tool_calls else "stop"
        usage = usage_to_openai(
            usage_from_openai(response.get("usage"), "responses"), "chat_completions"
        )
        return _responses_chat_chunk(
            state,
            delta={},
            finish_reason=finish_reason,
            usage=usage,
        )
    if event_type in {"response.failed", "error"}:
        raise BadInputError("Chat Completions output cannot represent a failed Response.")
    return None


def responses_from_chat_output_ir(output: OutputIR) -> OpenAIResponsesResponse:
    """Render Chat-native semantic output as an OpenAI Responses response."""

    if any(
        isinstance(item, AssistantMessageItem) and item.audio is not None
        for item in output.output.values
    ):
        raise BadInputError("Responses output cannot represent generated Chat audio.")
    return responses_from_output_ir(output)


def chat_from_responses_output_ir(output: OutputIR) -> OpenAIChatCompletionResponse:
    """Render Responses-native semantic output as a Chat response."""

    if output.status is ResponseStatus.FAILED:
        raise BadInputError("Chat Completions output cannot represent a failed Response.")
    if output.status is ResponseStatus.INCOMPLETE and output.incomplete_reason not in {
        "max_output_tokens",
        "content_filter",
    }:
        raise BadInputError(
            "Chat Completions output cannot represent Responses incomplete reason "
            f"{output.incomplete_reason!r}."
        )
    _validate_responses_output_for_chat(output.output.values)
    # Chat has no reasoning-item shape. The Chat composer emits only assistant
    # messages and function calls, intentionally discarding Responses reasoning.
    return chat_from_output_ir(output)


async def responses_from_chat_output_event_stream(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OpenAIResponsesStreamEvent]:
    """Render Chat-native semantic events as an OpenAI Responses stream."""

    semantic_events: list[OutputEvent] = []
    state = _ChatToResponsesState()
    matching_frames = False
    async for event in events:
        if isinstance(event, PreservedOpenAIStreamEvent) and event.endpoint == "chat_completions":
            matching_frames = True
            for response_event in _chat_frame_to_responses(state, event):
                yield response_event
            continue
        if not matching_frames:
            semantic_events.append(event)
    if matching_frames:
        return

    async for event in responses_from_output_event_stream(
        _reject_chat_audio(_as_async_events(semantic_events))
    ):
        yield event


async def chat_from_responses_output_event_stream(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Render Responses-native events as Chat chunks without buffering native frames."""

    semantic_events: list[OutputEvent] = []
    state = _ResponsesToChatState()
    matching_frames = False
    async for event in events:
        if isinstance(event, PreservedOpenAIStreamEvent) and event.endpoint == "responses":
            matching_frames = True
            chunk = _responses_frame_to_chat(state, event)
            if chunk is not None:
                yield _chat_chunk_without_terminal_usage(chunk)
            continue
        if not matching_frames:
            semantic_events.append(event)
    if matching_frames:
        return

    # The Chat stream is synthesized from the complete event sequence when a
    # provider does not expose native Responses frames.
    async for chunk in chat_from_output_event_stream(
        _reject_unsupported_chat_output(_as_async_events(semantic_events))
    ):
        yield _chat_chunk_without_terminal_usage(chunk)


async def _as_async_events(events: Iterable[OutputEvent]) -> AsyncIterator[OutputEvent]:
    for event in events:
        yield event


async def _reject_chat_audio(events: AsyncIterable[OutputEvent]) -> AsyncIterator[OutputEvent]:
    async for event in events:
        if (
            isinstance(event, OutputItemDoneEvent)
            and isinstance(event.item, AssistantMessageItem)
            and event.item.audio is not None
        ):
            raise BadInputError("Responses output cannot represent generated Chat audio.")
        yield event


def _validate_responses_output_for_chat(items: tuple[object, ...]) -> None:
    messages = [item for item in items if isinstance(item, AssistantMessageItem)]
    if len(messages) > 1:
        raise BadInputError("Chat Completions output cannot represent multiple messages.")
    if any(isinstance(item, OpenAIHostedToolItem) for item in items):
        raise BadInputError(
            "Chat Completions output cannot represent Responses hosted-tool items."
        )
    for message in messages:
        if message.phase is not None:
            raise BadInputError("Chat Completions output cannot represent Responses phase.")
        part_types = [type(part) for part in message.content]
        if any(part_types.count(part_type) > 1 for part_type in set(part_types)):
            raise BadInputError(
                "Chat Completions output cannot represent repeated content-part types."
            )


async def _reject_unsupported_chat_output(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OutputEvent]:
    assistant_messages = 0
    async for event in events:
        if isinstance(event, ResponseFailedEvent):
            raise BadInputError("Chat Completions output cannot represent a failed Response.")
        if isinstance(event, ResponseIncompleteEvent) and event.reason not in {
            "max_output_tokens",
            "content_filter",
        }:
            raise BadInputError(
                "Chat Completions output cannot represent Responses incomplete reason "
                f"{event.reason!r}."
            )
        if isinstance(event, OutputItemDoneEvent):
            _validate_responses_output_for_chat((event.item,))
            if isinstance(event.item, AssistantMessageItem):
                assistant_messages += 1
                if assistant_messages > 1:
                    raise BadInputError(
                        "Chat Completions output cannot represent multiple messages."
                    )
        yield event
