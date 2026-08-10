from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletionChunk

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
    ToolReturn,
    Usage,
)
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesStreamEvent,
    openai_responses_event_from_sse,
)
from molto.upstream.base import BaseClient


class OpenAIResponsesClient(BaseClient):
    """OpenAI Responses transport used for lossless same-protocol passthrough."""

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
                yield openai_responses_event_from_sse(record)


class OpenAIChatCompletionsClient(BaseClient):
    """Translate normalized requests to Chat Completions and normalize its stream."""

    @staticmethod
    def _messages(request: NormalizedRequest) -> list[dict[str, Any]]:
        if isinstance(request.input, str):
            return [{"role": "user", "content": request.input}]

        messages: list[dict[str, Any]] = []
        for item in request.input or []:
            if isinstance(item, SystemPrompt):
                messages.append({"role": "developer", "content": item.content})
            elif isinstance(item, TextInput):
                messages.append({"role": "user", "content": item.content})
            elif isinstance(item, TextOutput):
                messages.append({"role": "assistant", "content": item.content})
            elif isinstance(item, ToolReturn):
                messages.append(
                    {
                        "role": "tool",
                        "content": item.content,
                        "tool_call_id": item.tool_call_id,
                    }
                )
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
        content = ""
        usage: Usage | None = None
        part_started = False
        part_ended = False

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
                chunk = ChatCompletionChunk.model_validate_json(data)

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
                    if not part_started:
                        part_started = True
                        yield PartStartEvent(index=choice.index, part=TextOutput(content=""))

                    if choice.delta.content:
                        content += choice.delta.content
                        yield PartDeltaEvent(
                            index=choice.index,
                            delta=TextDelta(content_delta=choice.delta.content),
                        )

                    if choice.finish_reason is not None and not part_ended:
                        part_ended = True
                        yield PartEndEvent(index=choice.index, part=TextOutput(content=content))

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
        if part_started and not part_ended:
            yield PartEndEvent(index=0, part=TextOutput(content=content))
        yield ResponseEndEvent(
            id=response_metadata.id,
            created_at=response_metadata.created_at,
            model=response_metadata.model,
            service_tier=response_metadata.service_tier,
            usage=usage,
        )
