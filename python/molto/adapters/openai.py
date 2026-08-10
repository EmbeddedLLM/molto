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
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TextDelta,
    TextInput,
    TextOutput,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    openai_responses_event_from_data,
)


class OpenAIResponsesAdapter:
    """Translate the downstream OpenAI Responses protocol to and from normalized IR."""

    @staticmethod
    def _normalize_message(message: dict[str, Any]) -> list[SystemPrompt | TextInput | TextOutput]:
        if message.get("type", "message") != "message":
            raise ValueError(f"Unsupported Responses input type: {message.get('type')}")

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
        normalized_input: str | list[SystemPrompt | TextInput | TextOutput] | None
        if isinstance(request.input, str) or request.input is None:
            normalized_input = request.input
        else:
            normalized_input = []
            for item in request.input:
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

        def event(data: dict[str, Any]) -> BaseModel:
            nonlocal sequence_number
            data["sequence_number"] = sequence_number
            sequence_number += 1
            return openai_responses_event_from_data(data)

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
                "parallel_tool_calls": request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True,
                "tool_choice": request.tool_choice or "auto",
                "tools": request.tools or [],
            }
            if metadata.service_tier is not None:
                data["service_tier"] = metadata.service_tier
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
                if not isinstance(normalized_event.part, TextOutput):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                item_id = f"msg_{uuid4().hex}"
                output_index = normalized_event.index
                yield event(
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "status": "in_progress",
                            "content": [],
                            "role": "assistant",
                        },
                        "output_index": output_index,
                    }
                )
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
                if not isinstance(normalized_event.delta, TextDelta):
                    raise ValueError(f"Unsupported output delta: {type(normalized_event.delta)}")
                if item_id is None:
                    raise ValueError("Received a text delta before its output item started.")
                yield event(
                    {
                        "type": "response.output_text.delta",
                        "content_index": 0,
                        "delta": normalized_event.delta.content_delta,
                        "item_id": item_id,
                        "logprobs": [],
                        "output_index": normalized_event.index,
                    }
                )

            elif isinstance(normalized_event, PartEndEvent):
                if not isinstance(normalized_event.part, TextOutput):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                if item_id is None:
                    raise ValueError("Received a text end before its output item started.")
                output_index = normalized_event.index
                output_text = {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": normalized_event.part.content,
                }
                completed_item = {
                    "id": item_id,
                    "type": "message",
                    "status": "completed",
                    "content": [output_text],
                    "role": "assistant",
                }
                output_items.append(completed_item)
                yield event(
                    {
                        "type": "response.output_text.done",
                        "content_index": 0,
                        "item_id": item_id,
                        "logprobs": [],
                        "output_index": output_index,
                        "text": normalized_event.part.content,
                    }
                )
                yield event(
                    {
                        "type": "response.content_part.done",
                        "content_index": 0,
                        "item_id": item_id,
                        "output_index": output_index,
                        "part": output_text,
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
