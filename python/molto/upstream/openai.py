"""OpenAI Responses and Chat Completions upstream transport and normalization."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncStream,
)
from openai.types.responses import ResponseCompletedEvent, ResponseStreamEvent

from molto.types.normalized import (
    FunctionTool,
    FunctionToolChoice,
    ImageUrlInput,
    InputTokenDetails,
    ModelResponseStreamEvent,
    NormalizedAPIConnectionError,
    NormalizedAPIError,
    NormalizedAPIResponseValidationError,
    NormalizedAPIStatusError,
    NormalizedAPITimeoutError,
    NormalizedRequest,
    OutputTokenDetails,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TerminationReason,
    TextDelta,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolReturn,
    Usage,
    UserInput,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.base import BaseClient

_CHAT_TERMINATION_REASONS: dict[str, TerminationReason] = {
    "content_filter": "content_filter",
    "function_call": "tool_call",
    "length": "max_output_tokens",
    "stop": "stop",
    "tool_calls": "tool_call",
}


def _normalized_api_status_error(error: APIStatusError) -> NormalizedAPIStatusError:
    body = error.body
    message = body.get("message") if isinstance(body, Mapping) else None
    source_type = body.get("type") if isinstance(body, Mapping) else None
    code = body.get("code") if isinstance(body, Mapping) else None
    param = body.get("param") if isinstance(body, Mapping) else None
    request = error.response.request
    return NormalizedAPIStatusError(
        message if isinstance(message, str) else error.message,
        status_code=error.status_code,
        source_protocol="openai",
        source_type=source_type if isinstance(source_type, str) else None,
        code=code if isinstance(code, str) else None,
        param=param if isinstance(param, str) else None,
        request_id=error.request_id,
        response_headers=dict(error.response.headers),
        request_method=request.method,
        request_url=str(request.url),
        body=error.body,
    )


def _normalized_api_error(error: APIError) -> NormalizedAPIError:
    if isinstance(error, APIStatusError):
        return _normalized_api_status_error(error)

    request = error.request
    common = {
        "message": error.message,
        "source_protocol": "openai",
        "request_method": request.method,
        "request_url": str(request.url),
        "body": error.body,
    }
    if isinstance(error, APIResponseValidationError):
        return NormalizedAPIResponseValidationError(
            **common,
            status_code=error.status_code,
            response_headers=dict(error.response.headers),
        )
    if isinstance(error, APITimeoutError):
        return NormalizedAPITimeoutError(**common)
    if isinstance(error, APIConnectionError):
        return NormalizedAPIConnectionError(**common)
    return NormalizedAPIError(**common)


async def _normalized_api_events[EventT](
    events: AsyncIterator[EventT],
) -> AsyncIterator[EventT]:
    try:
        async for event in events:
            yield event
    except APIError as error:
        raise _normalized_api_error(error) from error


async def _open_normalized_api_events[EventT](
    events: Awaitable[AsyncIterator[EventT]],
) -> AsyncIterator[EventT]:
    try:
        native_events = await events
    except APIError as error:
        raise _normalized_api_error(error) from error
    return _normalized_api_events(native_events)


def _function_tool_payload(tool: FunctionTool) -> dict[str, Any]:
    """Project one function tool definition into the shared OpenAI field shape."""
    payload: dict[str, Any] = {
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": tool.strict,
    }
    if tool.description is not None:
        payload["description"] = tool.description
    return payload


def _responses_tools(request: NormalizedRequest) -> list[dict[str, Any]] | None:
    if request.tools is None:
        return None
    return [{"type": "function", **_function_tool_payload(tool)} for tool in request.tools]


def _responses_tool_choice(request: NormalizedRequest) -> Any:
    if isinstance(request.tool_choice, FunctionToolChoice):
        return {"type": "function", "name": request.tool_choice.name}
    return request.tool_choice


def _chat_tools(request: NormalizedRequest) -> list[dict[str, Any]] | None:
    if request.tools is None:
        return None
    return [
        {"type": "function", "function": _function_tool_payload(tool)} for tool in request.tools
    ]


def _chat_tool_choice(request: NormalizedRequest) -> Any:
    if isinstance(request.tool_choice, FunctionToolChoice):
        return {
            "type": "function",
            "function": {"name": request.tool_choice.name},
        }
    return request.tool_choice


def _validate_responses_function_call(
    item: Any,
    *,
    statuses: set[str | None],
) -> None:
    caller = item.caller
    if caller is not None and caller.type != "direct":
        raise ValueError("Only direct Responses function calls are supported.")
    if item.namespace is not None:
        raise ValueError("Namespaced Responses function calls are not supported.")
    if item.status not in statuses:
        raise ValueError(f"Unsupported Responses function call status: {item.status!r}.")


def _function_call_part(item: Any) -> ToolCall:
    """Project a Responses function call item into a normalized tool call."""
    return ToolCall(
        id=item.id,
        tool_call_id=item.call_id,
        tool_name=item.name,
        args=item.arguments,
    )


def _reasoning_part(item: Any) -> Thinking:
    """Project a Responses reasoning item snapshot into a normalized part."""
    return Thinking(
        content="".join(part.text for part in item.content or [] if part.type == "reasoning_text"),
        summary="".join(part.text for part in item.summary if part.type == "summary_text"),
        id=item.id,
        signature=item.encrypted_content,
    )


def _validate_completed_responses_reasoning(part: Thinking) -> Thinking:
    if part.summary and not part.signature:
        raise ValueError("A Responses reasoning summary requires a signature.")
    return part


def _validate_responses_reasoning_capability(
    part: Thinking, representation: Literal["raw", "summary"]
) -> None:
    if representation == "raw" and (part.summary or part.signature):
        raise ValueError("Raw Responses reasoning cannot contain a summary or signature.")
    if representation == "summary" and part.content:
        raise ValueError("Summarized Responses reasoning cannot contain raw text.")


def _started_output_item(item: Any) -> Thinking | TextOutput | ToolCall:
    """Project a just-added Responses output item into its initial normalized part."""
    if item.type == "reasoning":
        return _reasoning_part(item)
    if item.type == "message":
        return TextOutput(content="", id=item.id)
    if item.type == "function_call":
        _validate_responses_function_call(item, statuses={None, "in_progress"})
        return _function_call_part(item)
    raise ValueError(f"Unsupported Responses output item: {item.type}")


def _completed_output_item(item: Any) -> Thinking | TextOutput | ToolCall:
    """Project a completed Responses output item into its normalized part."""
    if item.type == "reasoning":
        return _validate_completed_responses_reasoning(_reasoning_part(item))
    if item.type == "message":
        return TextOutput(
            content="".join(part.text for part in item.content if part.type == "output_text"),
            id=item.id,
        )
    if item.type == "function_call":
        _validate_responses_function_call(item, statuses={None, "completed"})
        return _function_call_part(item)
    raise ValueError(f"Unsupported Responses output item: {item.type}")


def _responses_usage(raw_usage: Any) -> Usage | None:
    """Project Responses usage into normalized usage, preserving absent usage."""
    if raw_usage is None:
        return None
    input_details = raw_usage.input_tokens_details
    output_details = raw_usage.output_tokens_details
    return Usage(
        input_tokens=raw_usage.input_tokens,
        input_tokens_details=InputTokenDetails(
            cached_tokens=input_details.cached_tokens,
            cache_write_tokens=input_details.cache_write_tokens,
        ),
        output_tokens=raw_usage.output_tokens,
        output_tokens_details=OutputTokenDetails(reasoning_tokens=output_details.reasoning_tokens),
        total_tokens=raw_usage.total_tokens,
    )


def _chat_usage(raw_usage: Any) -> Usage:
    """Project Chat Completions usage into normalized usage."""
    prompt_details = raw_usage.prompt_tokens_details
    completion_details = raw_usage.completion_tokens_details
    return Usage(
        input_tokens=raw_usage.prompt_tokens,
        input_tokens_details=InputTokenDetails(
            cached_tokens=prompt_details.cached_tokens if prompt_details is not None else 0,
            cache_write_tokens=(
                prompt_details.cache_write_tokens if prompt_details is not None else 0
            ),
        ),
        output_tokens=raw_usage.completion_tokens,
        output_tokens_details=OutputTokenDetails(
            reasoning_tokens=(
                completion_details.reasoning_tokens if completion_details is not None else 0
            ),
        ),
        total_tokens=raw_usage.total_tokens,
    )


# Lifecycle echoes that carry no content beyond the events already translated.
_IGNORED_RESPONSES_EVENTS = frozenset(
    {
        "response.in_progress",
        "response.content_part.done",
        "response.reasoning_text.done",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.done",
        "response.function_call_arguments.done",
        "response.output_text.done",
    }
)


@dataclass
class _ChatToolCallState:
    """Accumulated wire state for one streamed Chat Completions tool call."""

    part_index: int
    tool_call_id: str
    tool_name: str
    args: str


class OpenAIResponsesClient(BaseClient[OpenAIResponsesCreateRequest, ResponseStreamEvent]):
    """OpenAI Responses transport for native and normalized streaming paths."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    async def native_events(
        self, request: OpenAIResponsesCreateRequest
    ) -> AsyncIterator[ResponseStreamEvent]:
        sdk_client = self._make_sdk_client()
        stream = await sdk_client.responses.create(**request.model_dump())
        if not isinstance(stream, AsyncStream):
            raise TypeError("Responses request did not return a stream.")

        async def iter_events() -> AsyncIterator[ResponseStreamEvent]:
            async with stream:
                async for event in stream:
                    if isinstance(event, ResponseCompletedEvent):
                        usage = event.response.usage
                        if (
                            usage is not None
                            and usage.input_tokens_details.cache_write_tokens is None
                        ):
                            usage.input_tokens_details.cache_write_tokens = 0
                    # The SDK already constructs ResponseStreamEvent instances. Apart
                    # from the provider omission above, avoid validating every event twice.
                    yield event

        return iter_events()

    @staticmethod
    def _process_input(request: NormalizedRequest) -> list[dict[str, Any]] | None:
        if request.input is None:
            return None

        items: list[dict[str, Any]] = []
        for item in request.input:
            if isinstance(item, SystemPrompt):
                items.append({"role": "developer", "content": item.content})
            elif isinstance(item, UserInput):
                content: list[dict[str, Any]] = []
                for part in item.content:
                    if isinstance(part, TextInput):
                        content.append({"type": "input_text", "text": part.content})
                    elif isinstance(part, ImageUrlInput):
                        image = {"type": "input_image", "image_url": part.url}
                        if part.detail is not None:
                            image["detail"] = part.detail
                        content.append(image)
                items.append({"role": "user", "content": content})
            elif isinstance(item, Thinking):
                _validate_completed_responses_reasoning(item)
                reasoning_item = {
                    "type": "reasoning",
                    "status": "completed",
                    "summary": (
                        [{"type": "summary_text", "text": item.summary}] if item.summary else []
                    ),
                    "content": (
                        [{"type": "reasoning_text", "text": item.content}] if item.content else []
                    ),
                }
                if item.signature is not None:
                    reasoning_item["encrypted_content"] = item.signature
                items.append(reasoning_item)
            elif isinstance(item, TextOutput):
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
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
            elif isinstance(item, ToolCall):
                function_call = {
                    "type": "function_call",
                    "status": "completed",
                    "arguments": item.args or "",
                    "call_id": item.tool_call_id,
                    "name": item.tool_name,
                }
                if item.id is not None:
                    function_call["id"] = item.id
                items.append(function_call)
            elif isinstance(item, ToolReturn):
                if item.outcome != "success":
                    raise ValueError(
                        "OpenAI protocols cannot preserve non-successful tool outcomes."
                    )
                output: list[dict[str, Any]] = []
                for part in item.content:
                    if isinstance(part, TextInput):
                        output.append({"type": "input_text", "text": part.content})
                    elif isinstance(part, ImageUrlInput):
                        image = {"type": "input_image", "image_url": part.url}
                        if part.detail is not None:
                            image["detail"] = part.detail
                        output.append(image)
                function_output = {
                    "type": "function_call_output",
                    "call_id": item.tool_call_id,
                    "output": output,
                }
                if item.id is not None:
                    function_output["id"] = item.id
                if item.tool_name is not None:
                    function_output["name"] = item.tool_name
                items.append(function_output)
            else:
                raise ValueError(f"Unsupported Responses input item: {type(item)}")
        return items

    @classmethod
    def _request_body(cls, request: NormalizedRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "input": cls._process_input(request),
            "model": request.model,
            "stream": request.stream,
        }
        for field in (
            "max_output_tokens",
            "max_tool_calls",
            "parallel_tool_calls",
            "temperature",
            "top_p",
        ):
            value = getattr(request, field)
            if value is not None:
                body[field] = value
        tools = _responses_tools(request)
        if tools is not None:
            body["tools"] = tools
        tool_choice = _responses_tool_choice(request)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if request.reasoning is not None:
            body["reasoning"] = request.reasoning.model_dump(exclude={"extra_fields"})
        if request.stream_options is not None:
            body["stream_options"] = request.stream_options.model_dump(exclude={"extra_fields"})
        return OpenAIResponsesCreateRequest.model_validate(body).model_dump()

    async def events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        native_request = OpenAIResponsesCreateRequest.model_validate(self._request_body(request))
        native_events = await _open_normalized_api_events(self.native_events(native_request))
        return self._events(native_events)

    async def _events(
        self,
        native_events: AsyncIterator[ResponseStreamEvent],
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        # Only a reasoning item whose `added` snapshot is completely empty (no
        # content, summary, or encrypted state) is deferred: if nothing ever
        # arrives, its part lifecycle is omitted from the IR entirely. An item
        # that already carries content or encrypted state at `added` starts
        # immediately, so a stalled or failed stream cannot lose it.
        pending_reasoning: dict[int, Thinking] = {}
        # Source output indices map to dense normalized indices so an omitted
        # reasoning item cannot leave a gap for downstream projections.
        part_indices: dict[int, int] = {}

        def start_part(
            source_index: int, part: Thinking | TextOutput | ToolCall
        ) -> PartStartEvent:
            part_indices[source_index] = len(part_indices)
            return PartStartEvent(index=part_indices[source_index], part=part)

        def part_index(source_index: int) -> int:
            # A delta or completion can reference an item whose start was never
            # seen; keep its source index rather than inventing a dense one.
            return part_indices.get(source_index, source_index)

        async for event in native_events:
            event_type = event.type

            if event_type == "response.created":
                metadata = event.response
                yield ResponseStartEvent(
                    id=metadata.id,
                    created_at=int(metadata.created_at),
                    model=metadata.model,
                    service_tier=metadata.service_tier,
                )
            elif event_type == "response.output_item.added":
                part = _started_output_item(event.item)
                if isinstance(part, Thinking):
                    _validate_responses_reasoning_capability(part, self._capabilities.thinking)
                if (
                    isinstance(part, Thinking)
                    and not part.content
                    and not part.summary
                    and part.signature is None
                ):
                    pending_reasoning[event.output_index] = part
                else:
                    yield start_part(event.output_index, part)
            elif event_type == "response.reasoning_text.delta":
                if self._capabilities.thinking != "raw" and event.delta:
                    raise ValueError("Summarized Responses reasoning cannot contain raw text.")
                # An empty delta must not materialize a deferred reasoning part.
                if not event.delta and event.output_index in pending_reasoning:
                    continue
                if (start := pending_reasoning.pop(event.output_index, None)) is not None:
                    yield start_part(event.output_index, start)
                yield PartDeltaEvent(
                    index=part_index(event.output_index),
                    delta=ThinkingDelta(content_delta=event.delta),
                )
            elif event_type == "response.reasoning_summary_text.delta":
                if self._capabilities.thinking != "summary" and event.delta:
                    raise ValueError("Raw Responses reasoning cannot contain a summary.")
                # An empty delta must not materialize a deferred reasoning part.
                if not event.delta and event.output_index in pending_reasoning:
                    continue
                if (start := pending_reasoning.pop(event.output_index, None)) is not None:
                    yield start_part(event.output_index, start)
                yield PartDeltaEvent(
                    index=part_index(event.output_index),
                    delta=ThinkingDelta(summary_delta=event.delta),
                )
            elif event_type == "response.content_part.added":
                # A part-added snapshot can carry initial text; empty ones are
                # lifecycle echoes and must not materialize a deferred part.
                added_part = event.part
                if added_part.type == "reasoning_text" and added_part.text:
                    if self._capabilities.thinking != "raw":
                        raise ValueError("Summarized Responses reasoning cannot contain raw text.")
                    if (start := pending_reasoning.pop(event.output_index, None)) is not None:
                        yield start_part(event.output_index, start)
                    yield PartDeltaEvent(
                        index=part_index(event.output_index),
                        delta=ThinkingDelta(content_delta=added_part.text),
                    )
                elif added_part.type == "output_text" and added_part.text:
                    yield PartDeltaEvent(
                        index=part_index(event.output_index),
                        delta=TextDelta(content_delta=added_part.text),
                    )
            elif event_type == "response.reasoning_summary_part.added":
                added_part = event.part
                if added_part.type == "summary_text" and added_part.text:
                    if self._capabilities.thinking != "summary":
                        raise ValueError("Raw Responses reasoning cannot contain a summary.")
                    if (start := pending_reasoning.pop(event.output_index, None)) is not None:
                        yield start_part(event.output_index, start)
                    yield PartDeltaEvent(
                        index=part_index(event.output_index),
                        delta=ThinkingDelta(summary_delta=added_part.text),
                    )
            elif event_type == "response.output_text.delta":
                yield PartDeltaEvent(
                    index=part_index(event.output_index),
                    delta=TextDelta(content_delta=event.delta),
                )
            elif event_type == "response.function_call_arguments.delta":
                yield PartDeltaEvent(
                    index=part_index(event.output_index),
                    delta=ToolCallDelta(args_delta=event.delta),
                )
            elif event_type == "response.output_item.done":
                start = pending_reasoning.pop(event.output_index, None)
                part = _completed_output_item(event.item)
                if isinstance(part, Thinking):
                    _validate_responses_reasoning_capability(part, self._capabilities.thinking)
                    # Deferred items always start empty, so the part is omitted
                    # when it also completes without content or state.
                    if (
                        start is not None
                        and not part.content
                        and not part.summary
                        and part.signature is None
                    ):
                        continue
                    if start is not None:
                        yield start_part(event.output_index, start)
                    if part.signature is not None:
                        yield PartDeltaEvent(
                            index=part_index(event.output_index),
                            delta=ThinkingDelta(signature_delta=part.signature),
                        )
                yield PartEndEvent(index=part_index(event.output_index), part=part)
            elif event_type in {"response.completed", "response.incomplete"}:
                metadata = event.response
                if event_type == "response.completed":
                    termination_reason: TerminationReason = (
                        "tool_call"
                        if any(item.type == "function_call" for item in metadata.output)
                        else "stop"
                    )
                else:
                    incomplete_details = metadata.incomplete_details
                    if incomplete_details is None or incomplete_details.reason is None:
                        raise ValueError("Incomplete Responses stream omitted its reason.")
                    termination_reason = incomplete_details.reason
                yield ResponseEndEvent(
                    id=metadata.id,
                    created_at=int(metadata.created_at),
                    model=metadata.model,
                    service_tier=metadata.service_tier,
                    termination_reason=termination_reason,
                    usage=_responses_usage(metadata.usage),
                )
            elif event_type in _IGNORED_RESPONSES_EVENTS:
                continue
            else:
                raise ValueError(f"Unsupported Responses stream event: {event_type}")


class OpenAIChatCompletionsClient(
    BaseClient[OpenAIChatCompletionsCreateRequest, OpenAIChatCompletionsStreamChunk]
):
    """OpenAI Chat Completions transport for native and normalized streaming paths."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    async def native_events(
        self, request: OpenAIChatCompletionsCreateRequest
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        sdk_client = self._make_sdk_client()
        stream = await sdk_client.chat.completions.create(**request.model_dump())
        if not isinstance(stream, AsyncStream):
            raise TypeError("Chat Completions request did not return a stream.")

        async def iter_events() -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
            async with stream:
                async for chunk in stream:
                    # The SDK already constructs ChatCompletionChunk instances. Trust that
                    # boundary to avoid serializing and validating every streaming chunk twice.
                    yield chunk

        return iter_events()

    @staticmethod
    def _messages(request: NormalizedRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        assistant_message: dict[str, Any] | None = None

        def pending_assistant() -> dict[str, Any]:
            nonlocal assistant_message
            if assistant_message is None:
                assistant_message = {"role": "assistant"}
            return assistant_message

        def flush_assistant() -> None:
            nonlocal assistant_message
            if assistant_message is None:
                return
            if "content" not in assistant_message:
                if "tool_calls" not in assistant_message:
                    raise ValueError(
                        "Chat Completions requires reasoning history with assistant output."
                    )
                assistant_message["content"] = None
            messages.append(assistant_message)
            assistant_message = None

        for item in request.input or []:
            if isinstance(item, SystemPrompt):
                flush_assistant()
                messages.append({"role": "developer", "content": item.content})
            elif isinstance(item, UserInput):
                flush_assistant()
                content: list[dict[str, Any]] = []
                for part in item.content:
                    if isinstance(part, TextInput):
                        content.append({"type": "text", "text": part.content})
                    elif isinstance(part, ImageUrlInput):
                        image_url = {"url": part.url}
                        if part.detail is not None:
                            image_url["detail"] = part.detail
                        content.append({"type": "image_url", "image_url": image_url})
                messages.append({"role": "user", "content": content})
            elif isinstance(item, Thinking):
                pending_assistant()["reasoning_content"] = item.content or item.summary
            elif isinstance(item, TextOutput):
                message = pending_assistant()
                if "content" in message:
                    flush_assistant()
                    message = pending_assistant()
                message["content"] = item.content
            elif isinstance(item, ToolCall):
                message = pending_assistant()
                message.setdefault("tool_calls", []).append(
                    {
                        "id": item.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": item.tool_name,
                            "arguments": item.args or "",
                        },
                    }
                )
            elif isinstance(item, ToolReturn):
                flush_assistant()
                if item.outcome != "success":
                    raise ValueError(
                        "OpenAI protocols cannot preserve non-successful tool outcomes."
                    )
                content: list[dict[str, Any]] = []
                for part in item.content:
                    if isinstance(part, TextInput):
                        content.append({"type": "text", "text": part.content})
                    elif isinstance(part, ImageUrlInput):
                        raise ValueError(
                            "Chat Completions does not support image tool result content."
                        )
                messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": item.tool_call_id,
                    }
                )
            else:
                raise ValueError(f"Unsupported Chat Completions input item: {type(item)}")
        flush_assistant()
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
        tools = _chat_tools(request)
        if tools is not None:
            body["tools"] = tools
        tool_choice = _chat_tool_choice(request)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if request.reasoning is not None and request.reasoning.effort is not None:
            body["reasoning_effort"] = request.reasoning.effort
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        return OpenAIChatCompletionsCreateRequest.model_validate(body).model_dump()

    async def events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        native_request = OpenAIChatCompletionsCreateRequest.model_validate(
            self._request_body(request)
        )
        native_events = await _open_normalized_api_events(self.native_events(native_request))
        return self._events(native_events)

    async def _events(
        self,
        native_events: AsyncIterator[OpenAIChatCompletionsStreamChunk],
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        response_metadata: ResponseStartEvent | None = None
        part_text = ""
        usage: Usage | None = None
        termination_reason: TerminationReason | None = None
        active_part: Literal["thinking", "text"] | None = None
        active_part_index = 0
        next_part_index = 0
        tool_calls: dict[int, _ChatToolCallState] = {}
        # Chat has no per-call stop event. Calls are treated as contiguous, so a
        # new index closes the active call and a later return to that index is invalid.
        active_tool_call_index: int | None = None

        def start_part(part: Thinking | TextOutput) -> PartStartEvent:
            nonlocal active_part, active_part_index, next_part_index, part_text
            active_part = "thinking" if isinstance(part, Thinking) else "text"
            active_part_index = next_part_index
            next_part_index += 1
            part_text = ""
            return PartStartEvent(index=active_part_index, part=part)

        def end_active_part() -> PartEndEvent | None:
            nonlocal active_part
            if active_part == "thinking":
                part: Thinking | TextOutput = self._classify_thinking(part_text)
            elif active_part == "text":
                part = TextOutput(content=part_text)
            else:
                return None
            active_part = None
            return PartEndEvent(index=active_part_index, part=part)

        def end_tool_call(tool_call_index: int) -> PartEndEvent:
            state = tool_calls[tool_call_index]
            return PartEndEvent(
                index=state.part_index,
                part=ToolCall(
                    tool_call_id=state.tool_call_id,
                    tool_name=state.tool_name,
                    args=state.args,
                ),
            )

        async for chunk in native_events:
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
                # Reasoning and text may interleave: a non-empty reasoning chunk
                # ends an active text part and starts a new thinking part. Empty
                # reasoning chunks carry nothing and never start or switch parts.
                if reasoning_delta:
                    if active_part == "text":
                        ended_part = end_active_part()
                        if ended_part is not None:
                            yield ended_part
                    if active_part is None:
                        yield start_part(Thinking(content="", summary=""))
                    part_text += reasoning_delta
                    yield PartDeltaEvent(
                        index=active_part_index,
                        delta=self._classify_thinking_delta(reasoning_delta),
                    )

                if choice.delta.content is not None:
                    is_empty_role_chunk = (
                        choice.delta.role is not None and choice.delta.content == ""
                    )
                    if not is_empty_role_chunk:
                        if active_part == "thinking":
                            ended_part = end_active_part()
                            if ended_part is not None:
                                yield ended_part
                        if active_part is None:
                            yield start_part(TextOutput(content=""))
                        if choice.delta.content:
                            part_text += choice.delta.content
                            yield PartDeltaEvent(
                                index=active_part_index,
                                delta=TextDelta(content_delta=choice.delta.content),
                            )

                if choice.delta.tool_calls is not None:
                    ended_part = end_active_part()
                    if ended_part is not None:
                        yield ended_part
                    for tool_call_delta in choice.delta.tool_calls:
                        function_delta = tool_call_delta.function
                        state = tool_calls.get(tool_call_delta.index)
                        if state is None:
                            if (
                                tool_call_delta.id is None
                                or function_delta is None
                                or function_delta.name is None
                            ):
                                raise ValueError(
                                    "The first Chat Completions chunk for a tool call must "
                                    "include its ID and function name."
                                )
                            if active_tool_call_index is not None:
                                yield end_tool_call(active_tool_call_index)
                            state = _ChatToolCallState(
                                part_index=next_part_index,
                                tool_call_id=tool_call_delta.id,
                                tool_name=function_delta.name,
                                args=function_delta.arguments or "",
                            )
                            tool_calls[tool_call_delta.index] = state
                            active_tool_call_index = tool_call_delta.index
                            yield PartStartEvent(
                                index=next_part_index,
                                part=ToolCall(
                                    tool_call_id=state.tool_call_id,
                                    tool_name=state.tool_name,
                                    args=state.args,
                                ),
                            )
                            next_part_index += 1
                            continue

                        if tool_call_delta.index != active_tool_call_index:
                            raise ValueError(
                                "Chat Completions tool-call deltas must be contiguous by index."
                            )

                        name_delta = function_delta.name if function_delta is not None else None
                        args_delta = (
                            function_delta.arguments if function_delta is not None else None
                        )
                        if name_delta == state.tool_name:
                            name_delta = None
                        tool_call_id_delta = tool_call_delta.id
                        if tool_call_id_delta == state.tool_call_id:
                            tool_call_id_delta = None
                        if name_delta is not None:
                            state.tool_name += name_delta
                        if args_delta is not None:
                            state.args += args_delta
                        if tool_call_id_delta is not None:
                            state.tool_call_id = tool_call_id_delta
                        yield PartDeltaEvent(
                            index=state.part_index,
                            delta=ToolCallDelta(
                                tool_name_delta=name_delta,
                                args_delta=args_delta,
                                tool_call_id=tool_call_id_delta,
                            ),
                        )

                if choice.finish_reason is not None:
                    try:
                        termination_reason = _CHAT_TERMINATION_REASONS[choice.finish_reason]
                    except KeyError as exc:
                        raise ValueError(
                            f"Unsupported Chat Completions finish reason: {choice.finish_reason}"
                        ) from exc
                    ended_part = end_active_part()
                    if ended_part is not None:
                        yield ended_part
                    if active_tool_call_index is not None:
                        yield end_tool_call(active_tool_call_index)
                    tool_calls.clear()
                    active_tool_call_index = None

            if chunk.usage is not None:
                usage = _chat_usage(chunk.usage)

        if response_metadata is None:
            raise ValueError("Chat Completions stream did not contain a response chunk.")
        if termination_reason is None:
            raise ValueError("Chat Completions stream ended without a finish reason.")
        ended_part = end_active_part()
        if ended_part is not None:
            yield ended_part
        yield ResponseEndEvent(
            id=response_metadata.id,
            created_at=response_metadata.created_at,
            model=response_metadata.model,
            service_tier=response_metadata.service_tier,
            termination_reason=termination_reason,
            usage=usage,
        )
