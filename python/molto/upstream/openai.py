from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

from molto.types.normalized import (
    InputTokenDetails,
    ModelResponseStreamEvent,
    NormalizedRequest,
    OutputTokenDetails,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TextDelta,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    ToolReturn,
    Usage,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesStreamEvent,
    validate_openai_responses_stream_event,
)
from molto.upstream.base import BaseClient
from molto.utils.io import json_loads, load_sse


class OpenAIResponsesClient(BaseClient):
    """OpenAI Responses transport for native and normalized streaming paths."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    async def native_events(
        self, request: OpenAIResponsesCreateRequest
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        async with self._http_client.stream(
            "POST",
            self._endpoint_url("responses"),
            headers=self._authorization_headers(),
            json=request.model_dump(),
        ) as response:
            response.raise_for_status()
            async for record in self._iter_sse_records(response):
                yield validate_openai_responses_stream_event(load_sse(record))

    @staticmethod
    def _input(request: NormalizedRequest) -> str | list[dict[str, Any]] | None:
        if isinstance(request.input, str) or request.input is None:
            return request.input

        items: list[dict[str, Any]] = []
        previous_was_thinking = False
        for item in request.input:
            if isinstance(item, SystemPrompt):
                items.append({"role": "developer", "content": item.content})
            elif isinstance(item, TextInput):
                items.append({"role": "user", "content": item.content})
            elif isinstance(item, Thinking):
                items.append(
                    {
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": item.content}],
                    }
                )
            elif isinstance(item, TextOutput):
                if previous_was_thinking:
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "phase": "final_answer",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": item.content,
                                    "annotations": [],
                                    "logprobs": [],
                                }
                            ],
                        }
                    )
                else:
                    items.append({"role": "assistant", "content": item.content})
            else:
                raise ValueError(f"Unsupported Responses input item: {type(item)}")
            previous_was_thinking = isinstance(item, Thinking)
        return items

    @classmethod
    def _request_body(cls, request: NormalizedRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "input": cls._input(request),
            "model": request.model,
            "stream": request.stream,
        }
        for field in (
            "max_output_tokens",
            "max_tool_calls",
            "parallel_tool_calls",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ):
            value = getattr(request, field)
            if value is not None:
                body[field] = value
        if request.reasoning is not None:
            body["reasoning"] = request.reasoning.model_dump(exclude={"extra_fields"})
        if request.stream_options is not None:
            body["stream_options"] = request.stream_options.model_dump(exclude={"extra_fields"})
        return OpenAIResponsesCreateRequest.model_validate(body).model_dump()

    async def events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        async with self._http_client.stream(
            "POST",
            self._endpoint_url("responses"),
            headers=self._authorization_headers(),
            json=self._request_body(request),
        ) as response:
            response.raise_for_status()
            async for record in self._iter_sse_records(response):
                payload = self._sse_data(record)
                if not payload:
                    continue
                data = json_loads(payload)
                event_type = data["type"]

                if event_type == "response.created":
                    metadata = data["response"]
                    yield ResponseStartEvent(
                        id=metadata["id"],
                        created_at=int(metadata["created_at"]),
                        model=metadata["model"],
                        service_tier=metadata.get("service_tier"),
                    )
                elif event_type == "response.output_item.added":
                    item = data["item"]
                    if item["type"] == "reasoning":
                        part = Thinking(content="", summary="", id=item.get("id"))
                    elif item["type"] == "message":
                        part = TextOutput(content="", id=item.get("id"))
                    else:
                        raise ValueError(f"Unsupported Responses output item: {item['type']}")
                    yield PartStartEvent(index=data["output_index"], part=part)
                elif event_type == "response.reasoning_text.delta":
                    yield PartDeltaEvent(
                        index=data["output_index"],
                        delta=ThinkingDelta(content_delta=data["delta"]),
                    )
                elif event_type == "response.output_text.delta":
                    yield PartDeltaEvent(
                        index=data["output_index"],
                        delta=TextDelta(content_delta=data["delta"]),
                    )
                elif event_type == "response.output_item.done":
                    item = data["item"]
                    if item["type"] == "reasoning":
                        content = "".join(
                            part["text"]
                            for part in item.get("content", [])
                            if part.get("type") == "reasoning_text"
                        )
                        summary = "".join(
                            part["text"]
                            for part in item.get("summary", [])
                            if part.get("type") == "summary_text"
                        )
                        part = Thinking(
                            content=content,
                            summary=summary,
                            id=item.get("id"),
                        )
                    elif item["type"] == "message":
                        content = "".join(
                            part["text"]
                            for part in item.get("content", [])
                            if part.get("type") == "output_text"
                        )
                        part = TextOutput(content=content, id=item.get("id"))
                    else:
                        raise ValueError(f"Unsupported Responses output item: {item['type']}")
                    yield PartEndEvent(index=data["output_index"], part=part)
                elif event_type == "response.completed":
                    metadata = data["response"]
                    raw_usage = metadata.get("usage")
                    usage = None
                    if raw_usage is not None:
                        input_details = raw_usage.get("input_tokens_details") or {}
                        output_details = raw_usage.get("output_tokens_details") or {}
                        usage = Usage(
                            input_tokens=raw_usage["input_tokens"],
                            input_tokens_details=InputTokenDetails(
                                cached_tokens=input_details.get("cached_tokens", 0),
                                cache_write_tokens=input_details.get("cache_write_tokens", 0),
                            ),
                            output_tokens=raw_usage["output_tokens"],
                            output_tokens_details=OutputTokenDetails(
                                reasoning_tokens=output_details.get("reasoning_tokens", 0)
                            ),
                            total_tokens=raw_usage["total_tokens"],
                        )
                    yield ResponseEndEvent(
                        id=metadata["id"],
                        created_at=int(metadata["created_at"]),
                        model=metadata["model"],
                        service_tier=metadata.get("service_tier"),
                        usage=usage,
                    )
                elif event_type in {
                    "response.in_progress",
                    "response.content_part.added",
                    "response.content_part.done",
                    "response.reasoning_text.done",
                    "response.output_text.done",
                }:
                    continue
                else:
                    raise ValueError(f"Unsupported Responses stream event: {event_type}")


class OpenAIChatCompletionsClient(BaseClient):
    """OpenAI Chat Completions transport for native and normalized streaming paths."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    async def native_events(
        self, request: OpenAIChatCompletionsCreateRequest
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        async with self._http_client.stream(
            "POST",
            self._endpoint_url("chat/completions"),
            headers=self._authorization_headers(),
            json=request.model_dump(),
        ) as response:
            response.raise_for_status()
            async for record in self._iter_sse_records(response):
                if self._sse_data(record) != "[DONE]":
                    yield OpenAIChatCompletionsStreamChunk.model_validate(load_sse(record))

    @staticmethod
    def _messages(request: NormalizedRequest) -> list[dict[str, Any]]:
        if isinstance(request.input, str):
            return [{"role": "user", "content": request.input}]

        messages: list[dict[str, Any]] = []
        reasoning_content: str | None = None
        for item in request.input or []:
            if isinstance(item, SystemPrompt):
                messages.append({"role": "developer", "content": item.content})
            elif isinstance(item, TextInput):
                messages.append({"role": "user", "content": item.content})
            elif isinstance(item, Thinking):
                reasoning_content = item.content
            elif isinstance(item, TextOutput):
                message = {"role": "assistant", "content": item.content}
                if reasoning_content is not None:
                    message["reasoning_content"] = reasoning_content
                    reasoning_content = None
                messages.append(message)
            elif isinstance(item, ToolReturn):
                messages.append(
                    {
                        "role": "tool",
                        "content": item.content,
                        "tool_call_id": item.tool_call_id,
                    }
                )
        if reasoning_content is not None:
            raise ValueError("Chat Completions requires reasoning history with assistant text.")
        return messages

    @classmethod
    def _request_body(cls, request: NormalizedRequest) -> dict[str, Any]:
        stream_options: dict[str, Any] = {}
        if request.stream:
            stream_options["include_usage"] = True
        if (
            request.stream_options is not None
            and request.stream_options.include_obfuscation is not None
        ):
            stream_options["include_obfuscation"] = request.stream_options.include_obfuscation

        body: dict[str, Any] = {
            "messages": cls._messages(request),
            "model": request.model,
            "stream": request.stream,
        }
        if stream_options:
            body["stream_options"] = stream_options
        if request.max_output_tokens is not None:
            body["max_completion_tokens"] = request.max_output_tokens
        if request.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.reasoning is not None and request.reasoning.effort is not None:
            body["reasoning_effort"] = request.reasoning.effort
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        return OpenAIChatCompletionsCreateRequest.model_validate(body).model_dump()

    async def events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        response_metadata: ResponseStartEvent | None = None
        thinking_content = ""
        text_content = ""
        usage: Usage | None = None
        active_part: Literal["thinking", "text"] | None = None
        next_part_index = 0

        def end_active_part() -> PartEndEvent | None:
            nonlocal active_part
            if active_part == "thinking":
                part = Thinking(content=thinking_content, summary="")
            elif active_part == "text":
                part = TextOutput(content=text_content)
            else:
                return None
            active_part = None
            return PartEndEvent(index=next_part_index - 1, part=part)

        async with self._http_client.stream(
            "POST",
            self._endpoint_url("chat/completions"),
            headers=self._authorization_headers(),
            json=self._request_body(request),
        ) as response:
            response.raise_for_status()
            async for record in self._iter_sse_records(response):
                data = self._sse_data(record)
                if not data or data == "[DONE]":
                    continue
                chunk = OpenAIChatCompletionsStreamChunk.model_validate_json(data)

                if response_metadata is None:
                    response_metadata = ResponseStartEvent(
                        id=chunk.id,
                        created_at=chunk.created,
                        model=chunk.model,
                        service_tier=chunk.service_tier,
                    )
                    yield response_metadata

                if chunk.choices:
                    choice = chunk.choices[0]
                    reasoning_delta = getattr(choice.delta, "reasoning_content", None)
                    if reasoning_delta is not None:
                        if active_part is None:
                            active_part = "thinking"
                            yield PartStartEvent(
                                index=next_part_index,
                                part=Thinking(content="", summary=""),
                            )
                            next_part_index += 1
                        elif active_part != "thinking":
                            raise ValueError("Received reasoning after text output started.")
                        if reasoning_delta:
                            thinking_content += reasoning_delta
                            yield PartDeltaEvent(
                                index=next_part_index - 1,
                                delta=ThinkingDelta(content_delta=reasoning_delta),
                            )

                    if choice.delta.content is not None:
                        if active_part == "thinking":
                            ended_part = end_active_part()
                            if ended_part is not None:
                                yield ended_part
                        if active_part is None:
                            active_part = "text"
                            yield PartStartEvent(
                                index=next_part_index,
                                part=TextOutput(content=""),
                            )
                            next_part_index += 1
                        if choice.delta.content:
                            text_content += choice.delta.content
                            yield PartDeltaEvent(
                                index=next_part_index - 1,
                                delta=TextDelta(content_delta=choice.delta.content),
                            )

                    if choice.finish_reason is not None:
                        ended_part = end_active_part()
                        if ended_part is not None:
                            yield ended_part

                if chunk.usage is not None:
                    prompt_details = chunk.usage.prompt_tokens_details
                    completion_details = chunk.usage.completion_tokens_details
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        input_tokens_details=InputTokenDetails(
                            cached_tokens=(prompt_details.cached_tokens or 0)
                            if prompt_details is not None
                            else 0,
                            cache_write_tokens=(prompt_details.cache_write_tokens or 0)
                            if prompt_details is not None
                            else 0,
                        ),
                        output_tokens=chunk.usage.completion_tokens,
                        output_tokens_details=OutputTokenDetails(
                            reasoning_tokens=(completion_details.reasoning_tokens or 0)
                            if completion_details is not None
                            else 0,
                        ),
                        total_tokens=chunk.usage.total_tokens,
                    )

        if response_metadata is None:
            raise ValueError("Chat Completions stream did not contain a response chunk.")
        ended_part = end_active_part()
        if ended_part is not None:
            yield ended_part
        yield ResponseEndEvent(
            id=response_metadata.id,
            created_at=response_metadata.created_at,
            model=response_metadata.model,
            service_tier=response_metadata.service_tier,
            usage=usage,
        )
