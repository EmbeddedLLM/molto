"""Translate OpenAI Responses and Chat Completions protocol boundaries.

See ``docs/openai-responses-cc-parity.md`` and
``docs/openai-responses-anthropic-messages-parity.md`` for supported semantics,
compatibility mappings, and testing policy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from molto.adapters.base import BaseAdapter
from molto.types.normalized import (
    ImageUrlInput,
    ModelResponseStreamEvent,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    Reasoning,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TerminationReason,
    TextDelta,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    UserInput,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesStreamEvent,
    OpenAIResponsesStreamEventAdapter,
)

_CHAT_FINISH_REASONS: dict[TerminationReason, str] = {
    "content_filter": "content_filter",
    "max_output_tokens": "length",
    "stop": "stop",
    "stop_sequence": "stop",
    "tool_call": "tool_calls",
}


class OpenAIResponsesAdapter(
    BaseAdapter[OpenAIResponsesCreateRequest, OpenAIResponsesStreamEvent]
):
    """Translate the downstream OpenAI Responses protocol to and from normalized IR."""

    def _normalize_message(
        self,
        message: dict[str, Any],
    ) -> list[SystemPrompt | UserInput | Thinking | TextOutput]:
        message_type = message.get("type", "message")
        if message_type == "reasoning":
            content = message.get("content") or []
            summary = message.get("summary") or []
            if not all(part.get("type") == "reasoning_text" for part in content):
                raise ValueError("Reasoning content should only contain `reasoning_text`.")
            if not all(part.get("type") == "summary_text" for part in summary):
                raise ValueError("Reasoning summary should only contain `summary_text`.")
            return [
                Thinking(
                    content="".join(part["text"] for part in content),
                    summary="".join(part["text"] for part in summary),
                    id=message.get("id"),
                    signature=message.get("encrypted_content"),
                )
            ]
        if message_type != "message":
            raise ValueError(f"Unsupported Responses input type: {message_type}")

        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            text_parts = [content]
        elif isinstance(content, list):
            if role == "user":
                parts: list[TextInput | ImageUrlInput] = []
                for part in content:
                    if isinstance(part, BaseModel):
                        part = part.model_dump()
                    part_type = part.get("type") if isinstance(part, dict) else None
                    if isinstance(part, dict) and "prompt_cache_breakpoint" in part:
                        raise ValueError(
                            f"Responses {part_type} prompt_cache_breakpoint is not "
                            "supported for translation."
                        )
                    if part_type == "input_text":
                        parts.append(TextInput(content=part["text"]))
                    elif part_type == "input_image":
                        image_url = part.get("image_url")
                        if image_url is None:
                            raise ValueError("Responses input_image requires image_url.")
                        parts.append(ImageUrlInput(url=image_url, detail=part.get("detail")))
                    else:
                        raise ValueError(f"Unsupported Responses user content type: {part_type}")
                return [UserInput(content=parts)]
            if role not in {"system", "developer", "assistant"}:
                raise ValueError(f"Unsupported Responses message role: {role}")

            expected_part_type = "output_text" if role == "assistant" else "input_text"
            if any(
                not isinstance(part, dict) or part.get("type") != expected_part_type
                for part in content
            ):
                raise ValueError(f"Responses {role} message content cannot be translated.")
            text_parts = [part["text"] for part in content]
        else:
            raise ValueError("Responses message content must be text.")

        if role in {"system", "developer"}:
            return [SystemPrompt(content=text) for text in text_parts]
        if role == "user":
            return [UserInput(content=[TextInput(content=text) for text in text_parts])]
        if role == "assistant":
            return [TextOutput(content=text) for text in text_parts]
        raise ValueError(f"Unsupported Responses message role: {role}")

    def _normalize_request(self, request: OpenAIResponsesCreateRequest) -> NormalizedRequest:
        normalized_input: list[SystemPrompt | UserInput | Thinking | TextOutput] | None
        if isinstance(request.input, str):
            normalized_input = [UserInput(content=[TextInput(content=request.input)])]
        elif request.input is None:
            normalized_input = None
        else:
            normalized_input = []
            for item in request.input:
                if isinstance(item, BaseModel):
                    item = item.model_dump()
                if not isinstance(item, dict):
                    raise ValueError(f"Unsupported Responses input value: {type(item)}")
                normalized_input.extend(self._normalize_message(item))

        if request.instructions is not None:
            normalized_input = [
                SystemPrompt(content=request.instructions),
                *(normalized_input or []),
            ]

        request_data = request.model_dump()
        request_data.pop("instructions", None)
        request_data["input"] = normalized_input
        return NormalizedRequest.model_validate(request_data)

    async def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        sequence_number = 0
        response_start: ResponseStartEvent | None = None
        item_id: str | None = None
        thinking_representation: str | None = None
        output_items: list[dict[str, Any]] = []

        def event(data: dict[str, Any]) -> BaseModel:
            nonlocal sequence_number
            data["sequence_number"] = sequence_number
            sequence_number += 1
            event = OpenAIResponsesStreamEventAdapter.validate_python(data)
            return event

        def response_data(
            metadata: ResponseStartEvent | ResponseEndEvent,
            *,
            status: str,
            output: list[dict[str, Any]],
        ) -> dict[str, Any]:
            data: dict[str, Any] = {
                "id": metadata.id,
                "object": "response",
                "created_at": metadata.created_at,
                "status": status,
                "model": metadata.model,
                "output": output,
                "max_output_tokens": request.max_output_tokens,
                "reasoning": self.native_request.reasoning,
                "parallel_tool_calls": request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True,
                "service_tier": "default",
                "tool_choice": self.native_request.tool_choice or "auto",
                "tools": self.native_request.tools or [],
                "usage": None,
            }
            if isinstance(metadata, ResponseEndEvent) and metadata.usage is not None:
                usage = metadata.usage
                data["usage"] = {
                    "input_tokens": usage.input_tokens,
                    "input_tokens_details": {
                        "cached_tokens": usage.input_tokens_details.cached_tokens,
                        "cache_write_tokens": usage.input_tokens_details.cache_write_tokens,
                    },
                    "output_tokens": usage.output_tokens,
                    "output_tokens_details": {
                        "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
                    },
                    "total_tokens": usage.total_tokens,
                }
            return data

        async for normalized_event in events:
            if isinstance(normalized_event, ResponseStartEvent):
                response_start = normalized_event
                initial_response = response_data(
                    normalized_event,
                    status="in_progress",
                    output=[],
                )
                yield event({"type": "response.created", "response": initial_response})
                yield event({"type": "response.in_progress", "response": initial_response})

            elif isinstance(normalized_event, PartStartEvent):
                if not isinstance(normalized_event.part, (TextOutput, Thinking)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                is_thinking = isinstance(normalized_event.part, Thinking)
                thinking_representation = None
                item_id = f"{'' if is_thinking else 'msg_'}{uuid4().hex}"
                output_index = normalized_event.index
                item = (
                    {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "content": [],
                        "summary": [],
                        **(
                            {"encrypted_content": normalized_event.part.signature}
                            if normalized_event.part.signature
                            else {}
                        ),
                    }
                    if is_thinking
                    else {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "content": [],
                        "role": "assistant",
                    }
                )
                yield event(
                    {
                        "type": "response.output_item.added",
                        "item": item,
                        "output_index": output_index,
                    }
                )
                if not is_thinking:
                    yield event(
                        {
                            "type": "response.content_part.added",
                            "content_index": 0,
                            "item_id": item_id,
                            "output_index": output_index,
                            "part": {
                                "type": "output_text",
                                "annotations": [],
                                "logprobs": [],
                                "text": "",
                            },
                        }
                    )

            elif isinstance(normalized_event, PartDeltaEvent):
                if not isinstance(normalized_event.delta, (TextDelta, ThinkingDelta)):
                    raise ValueError(f"Unsupported output delta: {type(normalized_event.delta)}")
                if item_id is None:
                    raise ValueError("Received a delta before its output item started.")
                if isinstance(normalized_event.delta, ThinkingDelta):
                    if normalized_event.delta.content_delta is not None:
                        representation = "raw"
                        delta = normalized_event.delta.content_delta
                    elif normalized_event.delta.summary_delta is not None:
                        representation = "summary"
                        delta = normalized_event.delta.summary_delta
                    else:
                        continue
                    if thinking_representation is None:
                        thinking_representation = representation
                        if representation == "raw":
                            yield event(
                                {
                                    "type": "response.content_part.added",
                                    "content_index": 0,
                                    "item_id": item_id,
                                    "output_index": normalized_event.index,
                                    "part": {"type": "reasoning_text", "text": ""},
                                }
                            )
                        else:
                            yield event(
                                {
                                    "type": "response.reasoning_summary_part.added",
                                    "item_id": item_id,
                                    "output_index": normalized_event.index,
                                    "part": {"type": "summary_text", "text": ""},
                                    "summary_index": 0,
                                }
                            )
                    elif thinking_representation != representation:
                        raise ValueError(
                            "A translated reasoning part cannot contain both raw reasoning "
                            "and a summary."
                        )
                    yield event(
                        {
                            "type": (
                                "response.reasoning_text.delta"
                                if representation == "raw"
                                else "response.reasoning_summary_text.delta"
                            ),
                            "delta": delta,
                            "item_id": item_id,
                            "output_index": normalized_event.index,
                            **(
                                {"content_index": 0}
                                if representation == "raw"
                                else {"summary_index": 0}
                            ),
                        }
                    )
                else:
                    yield event(
                        {
                            "type": "response.output_text.delta",
                            "content_index": 0,
                            "delta": normalized_event.delta.content_delta,
                            "item_id": item_id,
                            "output_index": normalized_event.index,
                            "logprobs": [],
                        }
                    )

            elif isinstance(normalized_event, PartEndEvent):
                if not isinstance(normalized_event.part, (TextOutput, Thinking)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                if item_id is None:
                    raise ValueError("Received a part end before its output item started.")
                output_index = normalized_event.index
                is_thinking = isinstance(normalized_event.part, Thinking)
                if is_thinking:
                    if normalized_event.part.content and normalized_event.part.summary:
                        raise ValueError(
                            "A translated reasoning part cannot contain both raw reasoning "
                            "and a summary."
                        )
                    representation = thinking_representation or (
                        "summary" if normalized_event.part.summary else self._capabilities.thinking
                    )
                    reasoning_text = (
                        normalized_event.part.content
                        if representation == "raw"
                        else normalized_event.part.summary
                    )
                    completed_part = {
                        "type": "reasoning_text" if representation == "raw" else "summary_text",
                        "text": reasoning_text,
                    }
                    if thinking_representation is None:
                        if representation == "raw":
                            yield event(
                                {
                                    "type": "response.content_part.added",
                                    "content_index": 0,
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "part": {"type": "reasoning_text", "text": ""},
                                }
                            )
                        else:
                            yield event(
                                {
                                    "type": "response.reasoning_summary_part.added",
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "part": {"type": "summary_text", "text": ""},
                                    "summary_index": 0,
                                }
                            )
                else:
                    completed_part = {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": normalized_event.part.content,
                    }
                completed_item = (
                    {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "completed",
                        "content": [completed_part] if representation == "raw" else [],
                        "summary": [completed_part] if representation == "summary" else [],
                        **(
                            {"encrypted_content": normalized_event.part.signature}
                            if normalized_event.part.signature
                            else {}
                        ),
                    }
                    if is_thinking
                    else {
                        "id": item_id,
                        "type": "message",
                        "status": "completed",
                        "content": [completed_part],
                        "role": "assistant",
                    }
                )
                output_items.append(completed_item)
                if is_thinking and representation == "summary":
                    yield event(
                        {
                            "type": "response.reasoning_summary_text.done",
                            "item_id": item_id,
                            "output_index": output_index,
                            "summary_index": 0,
                            "text": reasoning_text,
                        }
                    )
                    yield event(
                        {
                            "type": "response.reasoning_summary_part.done",
                            "item_id": item_id,
                            "output_index": output_index,
                            "part": completed_part,
                            "summary_index": 0,
                        }
                    )
                else:
                    yield event(
                        {
                            "type": (
                                "response.reasoning_text.done"
                                if is_thinking
                                else "response.output_text.done"
                            ),
                            "content_index": 0,
                            "item_id": item_id,
                            "output_index": output_index,
                            "text": normalized_event.part.content,
                            **({} if is_thinking else {"logprobs": []}),
                        }
                    )
                    yield event(
                        {
                            "type": "response.content_part.done",
                            "content_index": 0,
                            "item_id": item_id,
                            "output_index": output_index,
                            "part": completed_part,
                        }
                    )
                yield event(
                    {
                        "type": "response.output_item.done",
                        "item": completed_item,
                        "output_index": output_index,
                    }
                )

            elif isinstance(normalized_event, ResponseEndEvent):
                if response_start is None:
                    raise ValueError("Received a response end before its response start.")
                termination_reason = normalized_event.termination_reason
                if termination_reason in {"content_filter", "max_output_tokens"}:
                    event_type = "response.incomplete"
                    status = "incomplete"
                else:
                    event_type = "response.completed"
                    status = "completed"
                final_response = response_data(
                    normalized_event,
                    status=status,
                    output=output_items,
                )
                if status == "incomplete":
                    final_response["incomplete_details"] = {"reason": termination_reason}
                yield event(
                    {
                        "type": event_type,
                        "response": final_response,
                    }
                )


class OpenAIChatCompletionsAdapter(
    BaseAdapter[OpenAIChatCompletionsCreateRequest, OpenAIChatCompletionsStreamChunk]
):
    """Translate Chat Completions requests and normalized response events."""

    def _normalize_message(
        self,
        message: dict[str, Any],
    ) -> list[SystemPrompt | UserInput | Thinking | TextOutput]:
        role = message.get("role")
        content = message.get("content")

        if role == "user" and isinstance(content, list):
            parts: list[TextInput | ImageUrlInput] = []
            for part in content:
                if isinstance(part, BaseModel):
                    part = part.model_dump()
                part_type = part.get("type") if isinstance(part, dict) else None
                if part_type == "text":
                    parts.append(TextInput(content=part["text"]))
                elif part_type == "image_url":
                    image = part["image_url"]
                    parts.append(ImageUrlInput(url=image["url"], detail=image.get("detail")))
                else:
                    raise ValueError(f"Unsupported Chat user content type: {part_type}")
            return [UserInput(content=parts)]

        if not isinstance(content, str):
            raise ValueError("Only string non-user Chat Completions content is supported.")

        if role in {"system", "developer"}:
            return [SystemPrompt(content=content)]
        if role == "user":
            return [UserInput(content=[TextInput(content=content)])]
        if role == "assistant":
            items: list[Thinking | TextOutput] = []
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is not None:
                if not isinstance(reasoning_content, str):
                    raise ValueError("Chat Completions reasoning_content must be a string.")
                items.append(
                    Thinking(
                        content=(
                            reasoning_content if self._capabilities.thinking == "raw" else ""
                        ),
                        summary=(
                            reasoning_content if self._capabilities.thinking == "summary" else ""
                        ),
                    )
                )
            items.append(TextOutput(content=content))
            return items
        raise ValueError(f"Unsupported Chat Completions message role: {role}")

    def _normalize_request(self, request: OpenAIChatCompletionsCreateRequest) -> NormalizedRequest:
        normalized_input: list[SystemPrompt | UserInput | Thinking | TextOutput] = []
        for message in request.messages:
            normalized_input.extend(self._normalize_message(dict(message)))

        stream_options = request.stream_options or {}
        include_obfuscation = stream_options.get("include_obfuscation")
        data: dict[str, Any] = {
            "input": normalized_input,
            "model": request.model,
            "stream": request.stream,
        }
        if include_obfuscation is not None:
            data["stream_options"] = {"include_obfuscation": include_obfuscation}
        if request.max_completion_tokens is not None:
            data["max_output_tokens"] = request.max_completion_tokens
        if request.parallel_tool_calls is not None:
            data["parallel_tool_calls"] = request.parallel_tool_calls
        if request.reasoning_effort is not None:
            # Chat does not say whether its reasoning field is raw or summarized.
            # Request a detailed Responses summary; the provider profile classifies
            # ambiguous Chat output independently.
            data["reasoning"] = Reasoning(
                effort=request.reasoning_effort,
                # in case of effort is "none" summary "detailed" does not make sense.
                summary="detailed" if request.reasoning_effort != "none" else None,
            )
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        return NormalizedRequest.model_validate(data)

    async def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        del request
        response_start: ResponseStartEvent | None = None
        initial_chunk_sent = False

        def chunk(
            metadata: ResponseStartEvent | ResponseEndEvent,
            *,
            choices: list[dict[str, Any]],
            usage: dict[str, Any] | None = None,
        ) -> BaseModel:
            return OpenAIChatCompletionsStreamChunk.model_validate(
                {
                    "id": metadata.id,
                    "object": "chat.completion.chunk",
                    "created": metadata.created_at,
                    "model": metadata.model,
                    "service_tier": "default",
                    "choices": choices,
                    "usage": usage,
                }
            )

        async for normalized_event in events:
            if isinstance(normalized_event, ResponseStartEvent):
                response_start = normalized_event

            elif isinstance(normalized_event, PartStartEvent):
                if not isinstance(normalized_event.part, (Thinking, TextOutput)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                if initial_chunk_sent:
                    continue
                if response_start is None:
                    raise ValueError("Received a part before its response started.")
                initial_chunk_sent = True
                delta = (
                    {"role": "assistant", "content": None, "reasoning_content": ""}
                    if isinstance(normalized_event.part, Thinking)
                    else {"role": "assistant", "content": "", "refusal": None}
                )
                yield chunk(
                    response_start,
                    choices=[
                        {
                            "index": 0,
                            "delta": delta,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                )

            elif isinstance(normalized_event, PartDeltaEvent):
                if response_start is None:
                    raise ValueError("Received a delta before its response started.")
                if isinstance(normalized_event.delta, ThinkingDelta):
                    reasoning_delta = (
                        normalized_event.delta.content_delta
                        if normalized_event.delta.content_delta is not None
                        else normalized_event.delta.summary_delta
                    )
                    if reasoning_delta is None:
                        continue
                    delta = {"reasoning_content": reasoning_delta}
                elif isinstance(normalized_event.delta, TextDelta):
                    delta = {"content": normalized_event.delta.content_delta}
                else:
                    raise ValueError(f"Unsupported output delta: {type(normalized_event.delta)}")
                yield chunk(
                    response_start,
                    choices=[
                        {
                            "index": 0,
                            "delta": delta,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                )

            elif isinstance(normalized_event, PartEndEvent):
                if not isinstance(normalized_event.part, (Thinking, TextOutput)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")

            elif isinstance(normalized_event, ResponseEndEvent):
                finish_reason = _CHAT_FINISH_REASONS.get(
                    normalized_event.termination_reason, "stop"
                )
                usage = None
                if normalized_event.usage is not None:
                    normalized_usage = normalized_event.usage
                    usage = {
                        "prompt_tokens": normalized_usage.input_tokens,
                        "completion_tokens": normalized_usage.output_tokens,
                        "total_tokens": normalized_usage.total_tokens,
                        "prompt_tokens_details": {
                            "cached_tokens": normalized_usage.input_tokens_details.cached_tokens,
                            "cache_write_tokens": (
                                normalized_usage.input_tokens_details.cache_write_tokens
                            ),
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": (
                                normalized_usage.output_tokens_details.reasoning_tokens
                            ),
                        },
                    }
                yield chunk(
                    normalized_event,
                    choices=[
                        {
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                    usage=usage,
                )
