from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from molto.types.normalized import (
    ModelResponseStreamEvent,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    Reasoning,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TextDelta,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesStreamEventAdapter,
)


class OpenAIResponsesAdapter:
    """Translate the downstream OpenAI Responses protocol to and from normalized IR."""

    @staticmethod
    def _normalize_message(
        message: dict[str, Any],
    ) -> list[SystemPrompt | TextInput | Thinking | TextOutput]:
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
                )
            ]
        if message_type != "message":
            raise ValueError(f"Unsupported Responses input type: {message_type}")

        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            text_parts = [content]
        elif isinstance(content, list):
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, dict) and part.get("type") in {"input_text", "output_text"}
            ]
            if len(text_parts) != len(content):
                raise ValueError("Only text message content is currently supported.")
        else:
            raise ValueError("Responses message content must be text.")

        item_type: type[SystemPrompt] | type[TextInput] | type[TextOutput]
        if role in {"system", "developer"}:
            item_type = SystemPrompt
        elif role == "user":
            item_type = TextInput
        elif role == "assistant":
            item_type = TextOutput
        else:
            raise ValueError(f"Unsupported Responses message role: {role}")
        return [item_type(content=text) for text in text_parts]

    def normalize_request(self, request: OpenAIResponsesCreateRequest) -> NormalizedRequest:
        normalized_input: str | list[SystemPrompt | TextInput | Thinking | TextOutput] | None
        if isinstance(request.input, str) or request.input is None:
            normalized_input = request.input
        else:
            normalized_input = []
            for item in request.input:
                if isinstance(item, BaseModel):
                    item = item.model_dump()
                if not isinstance(item, dict):
                    raise ValueError(f"Unsupported Responses input value: {type(item)}")
                normalized_input.extend(self._normalize_message(item))

        if request.instructions is not None:
            if isinstance(normalized_input, str):
                normalized_input = [
                    SystemPrompt(content=request.instructions),
                    TextInput(content=normalized_input),
                ]
            else:
                normalized_input = [
                    SystemPrompt(content=request.instructions),
                    *(normalized_input or []),
                ]

        request_data = request.model_dump()
        request_data.pop("instructions", None)
        request_data["input"] = normalized_input
        return NormalizedRequest.model_validate(request_data)

    async def project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[BaseModel]:
        sequence_number = 0
        response_start: ResponseStartEvent | None = None
        item_id: str | None = None
        output_items: list[dict[str, Any]] = []
        had_thinking = False

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
                "usage": None,
                "parallel_tool_calls": request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True,
                "tool_choice": request.tool_choice or "auto",
                "tools": request.tools or [],
            }
            if metadata.service_tier is not None:
                data["service_tier"] = metadata.service_tier
            if request.reasoning is not None:
                data["reasoning"] = request.reasoning.model_dump(exclude={"extra_fields"})
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
                had_thinking = had_thinking or is_thinking
                item_id = f"{'' if is_thinking else 'msg_'}{uuid4().hex}"
                output_index = normalized_event.index
                item = (
                    {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "content": [],
                        "summary": [],
                    }
                    if is_thinking
                    else {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "content": [],
                        "role": "assistant",
                        **({"phase": "final_answer"} if had_thinking else {}),
                    }
                )
                part = (
                    {"type": "reasoning_text", "text": ""}
                    if is_thinking
                    else {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "",
                    }
                )
                yield event(
                    {
                        "type": "response.output_item.added",
                        "item": item,
                        "output_index": output_index,
                    }
                )
                yield event(
                    {
                        "type": "response.content_part.added",
                        "content_index": 0,
                        "item_id": item_id,
                        "output_index": output_index,
                        "part": part,
                    }
                )

            elif isinstance(normalized_event, PartDeltaEvent):
                if not isinstance(normalized_event.delta, (TextDelta, ThinkingDelta)):
                    raise ValueError(f"Unsupported output delta: {type(normalized_event.delta)}")
                if item_id is None:
                    raise ValueError("Received a delta before its output item started.")
                is_thinking = isinstance(normalized_event.delta, ThinkingDelta)
                delta = normalized_event.delta.content_delta
                if delta is None:
                    continue
                yield event(
                    {
                        "type": (
                            "response.reasoning_text.delta"
                            if is_thinking
                            else "response.output_text.delta"
                        ),
                        "content_index": 0,
                        "delta": delta,
                        "item_id": item_id,
                        "output_index": normalized_event.index,
                        **({} if is_thinking else {"logprobs": []}),
                    }
                )

            elif isinstance(normalized_event, PartEndEvent):
                if not isinstance(normalized_event.part, (TextOutput, Thinking)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                if item_id is None:
                    raise ValueError("Received a part end before its output item started.")
                output_index = normalized_event.index
                is_thinking = isinstance(normalized_event.part, Thinking)
                completed_part = (
                    {"type": "reasoning_text", "text": normalized_event.part.content}
                    if is_thinking
                    else {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": normalized_event.part.content,
                    }
                )
                completed_item = (
                    {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "completed",
                        "content": [completed_part],
                        "summary": [],
                    }
                    if is_thinking
                    else {
                        "id": item_id,
                        "type": "message",
                        "status": "completed",
                        "content": [completed_part],
                        "role": "assistant",
                        **({"phase": "final_answer"} if had_thinking else {}),
                    }
                )
                output_items.append(completed_item)
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
                yield event(
                    {
                        "type": "response.completed",
                        "response": response_data(
                            normalized_event,
                            status="completed",
                            output=output_items,
                        ),
                    }
                )


class OpenAIChatCompletionsAdapter:
    """Translate Chat Completions requests and normalized response events."""

    @staticmethod
    def _normalize_message(
        message: dict[str, Any],
    ) -> list[SystemPrompt | TextInput | Thinking | TextOutput]:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Only string Chat Completions message content is supported.")

        if role in {"system", "developer"}:
            return [SystemPrompt(content=content)]
        if role == "user":
            return [TextInput(content=content)]
        if role == "assistant":
            items: list[Thinking | TextOutput] = []
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is not None:
                if not isinstance(reasoning_content, str):
                    raise ValueError("Chat Completions reasoning_content must be a string.")
                items.append(Thinking(content=reasoning_content, summary=""))
            items.append(TextOutput(content=content))
            return items
        raise ValueError(f"Unsupported Chat Completions message role: {role}")

    def normalize_request(self, request: OpenAIChatCompletionsCreateRequest) -> NormalizedRequest:
        normalized_input: list[SystemPrompt | TextInput | Thinking | TextOutput] = []
        for message in request.messages:
            normalized_input.extend(self._normalize_message(dict(message)))

        request_input: str | list[SystemPrompt | TextInput | Thinking | TextOutput]
        if len(normalized_input) == 1 and isinstance(normalized_input[0], TextInput):
            request_input = normalized_input[0].content
        else:
            request_input = normalized_input

        stream_options = request.stream_options or {}
        include_obfuscation = stream_options.get("include_obfuscation")
        data: dict[str, Any] = {
            "input": request_input,
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
            # Chat providers expose raw reasoning for this mode. Responses needs an
            # explicit detailed summary request to provide the equivalent stream.
            data["reasoning"] = Reasoning(
                effort=request.reasoning_effort,
                summary="detailed",
            )
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        return NormalizedRequest.model_validate(data)

    async def project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[BaseModel]:
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
                    "service_tier": metadata.service_tier,
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
                    if normalized_event.delta.content_delta is None:
                        continue
                    delta = {"reasoning_content": normalized_event.delta.content_delta}
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
                            "finish_reason": "stop",
                        }
                    ],
                    usage=usage,
                )
