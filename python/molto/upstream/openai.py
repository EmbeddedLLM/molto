from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

from openai import AsyncStream
from openai.types.responses import ResponseCompletedEvent, ResponseStreamEvent

from molto.types.normalized import (
    FunctionToolChoice,
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
    TerminationReason,
    TextDelta,
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


def _responses_tools(request: NormalizedRequest) -> list[dict[str, Any]] | None:
    if request.tools is None:
        return None
    tools: list[dict[str, Any]] = []
    for tool in request.tools:
        data: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }
        if tool.description is not None:
            data["description"] = tool.description
        tools.append(data)
    return tools


def _responses_tool_choice(request: NormalizedRequest) -> Any:
    if isinstance(request.tool_choice, FunctionToolChoice):
        return {"type": "function", "name": request.tool_choice.name}
    return request.tool_choice


def _chat_tools(request: NormalizedRequest) -> list[dict[str, Any]] | None:
    if request.tools is None:
        return None
    tools: list[dict[str, Any]] = []
    for tool in request.tools:
        function: dict[str, Any] = {
            "name": tool.name,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }
        if tool.description is not None:
            function["description"] = tool.description
        tools.append({"type": "function", "function": function})
    return tools


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
        async with stream:
            async for event in stream:
                if isinstance(event, ResponseCompletedEvent):
                    usage = event.response.usage
                    if usage is not None and usage.input_tokens_details.cache_write_tokens is None:
                        usage.input_tokens_details.cache_write_tokens = 0
                # The SDK already constructs ResponseStreamEvent instances. Apart from
                # the provider omission above, avoid validating every event twice.
                yield event

    @staticmethod
    def _process_input(request: NormalizedRequest) -> list[dict[str, Any]] | None:
        if isinstance(request.input, str):
            return [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request.input}],
                }
            ]
        if request.input is None:
            return None

        items: list[dict[str, Any]] = []
        for item in request.input:
            if isinstance(item, SystemPrompt):
                items.append({"role": "developer", "content": item.content})
            elif isinstance(item, UserInput):
                items.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": part.content} for part in item.content
                        ],
                    }
                )
            elif isinstance(item, Thinking):
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
                output: str | list[dict[str, Any]] = item.content
                if isinstance(item.content, list):
                    output = [
                        {"type": "input_text", "text": part.content} for part in item.content
                    ]
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
        async for event in self.native_events(native_request):
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
                item = event.item
                if item.type == "reasoning":
                    part = Thinking(
                        content="",
                        summary="",
                        id=item.id,
                        signature=item.encrypted_content,
                    )
                elif item.type == "message":
                    part = TextOutput(content="", id=item.id)
                elif item.type == "function_call":
                    _validate_responses_function_call(
                        item,
                        statuses={None, "in_progress"},
                    )
                    part = ToolCall(
                        id=item.id,
                        tool_call_id=item.call_id,
                        tool_name=item.name,
                        args=item.arguments,
                    )
                else:
                    raise ValueError(f"Unsupported Responses output item: {item.type}")
                yield PartStartEvent(index=event.output_index, part=part)
            elif event_type == "response.reasoning_text.delta":
                yield PartDeltaEvent(
                    index=event.output_index,
                    delta=ThinkingDelta(content_delta=event.delta),
                )
            elif event_type == "response.reasoning_summary_text.delta":
                yield PartDeltaEvent(
                    index=event.output_index,
                    delta=ThinkingDelta(summary_delta=event.delta),
                )
            elif event_type == "response.output_text.delta":
                yield PartDeltaEvent(
                    index=event.output_index,
                    delta=TextDelta(content_delta=event.delta),
                )
            elif event_type == "response.function_call_arguments.delta":
                yield PartDeltaEvent(
                    index=event.output_index,
                    delta=ToolCallDelta(args_delta=event.delta),
                )
            elif event_type == "response.output_item.done":
                item = event.item
                if item.type == "reasoning":
                    content = "".join(
                        part.text for part in item.content or [] if part.type == "reasoning_text"
                    )
                    summary = "".join(
                        part.text for part in item.summary if part.type == "summary_text"
                    )
                    if item.encrypted_content is not None:
                        yield PartDeltaEvent(
                            index=event.output_index,
                            delta=ThinkingDelta(signature_delta=item.encrypted_content),
                        )
                    part = Thinking(
                        content=content,
                        summary=summary,
                        id=item.id,
                        signature=item.encrypted_content,
                    )
                elif item.type == "message":
                    content = "".join(
                        part.text for part in item.content if part.type == "output_text"
                    )
                    part = TextOutput(content=content, id=item.id)
                elif item.type == "function_call":
                    _validate_responses_function_call(
                        item,
                        statuses={None, "completed"},
                    )
                    part = ToolCall(
                        id=item.id,
                        tool_call_id=item.call_id,
                        tool_name=item.name,
                        args=item.arguments,
                    )
                else:
                    raise ValueError(f"Unsupported Responses output item: {item.type}")
                yield PartEndEvent(index=event.output_index, part=part)
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
                raw_usage = metadata.usage
                usage = None
                if raw_usage is not None:
                    input_details = raw_usage.input_tokens_details
                    output_details = raw_usage.output_tokens_details
                    usage = Usage(
                        input_tokens=raw_usage.input_tokens,
                        input_tokens_details=InputTokenDetails(
                            cached_tokens=input_details.cached_tokens,
                            cache_write_tokens=input_details.cache_write_tokens,
                        ),
                        output_tokens=raw_usage.output_tokens,
                        output_tokens_details=OutputTokenDetails(
                            reasoning_tokens=output_details.reasoning_tokens
                        ),
                        total_tokens=raw_usage.total_tokens,
                    )
                yield ResponseEndEvent(
                    id=metadata.id,
                    created_at=int(metadata.created_at),
                    model=metadata.model,
                    service_tier=metadata.service_tier,
                    termination_reason=termination_reason,
                    usage=usage,
                )
            elif event_type in {
                "response.in_progress",
                "response.content_part.added",
                "response.content_part.done",
                "response.reasoning_text.done",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_part.done",
                "response.reasoning_summary_text.done",
                "response.function_call_arguments.done",
                "response.output_text.done",
            }:
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
        async with stream:
            async for chunk in stream:
                # The SDK already constructs ChatCompletionChunk instances. Trust that
                # boundary to avoid serializing and validating every streaming chunk twice.
                yield chunk

    @staticmethod
    def _messages(request: NormalizedRequest) -> list[dict[str, Any]]:
        if isinstance(request.input, str):
            return [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": request.input}],
                }
            ]

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
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": part.content} for part in item.content
                        ],
                    }
                )
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
                content: str | list[dict[str, str]] = item.content
                if isinstance(item.content, list):
                    content = [{"type": "text", "text": part.content} for part in item.content]
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
        response_metadata: ResponseStartEvent | None = None
        thinking_text = ""
        text_content = ""
        usage: Usage | None = None
        termination_reason: TerminationReason | None = None
        active_part: Literal["thinking", "text"] | None = None
        next_part_index = 0
        tool_calls: dict[int, dict[str, Any]] = {}

        def end_active_part() -> PartEndEvent | None:
            nonlocal active_part
            if active_part == "thinking":
                part = Thinking(
                    content=thinking_text if self._capabilities.thinking == "raw" else "",
                    summary=thinking_text if self._capabilities.thinking == "summary" else "",
                )
            elif active_part == "text":
                part = TextOutput(content=text_content)
            else:
                return None
            active_part = None
            return PartEndEvent(index=next_part_index - 1, part=part)

        native_request = OpenAIChatCompletionsCreateRequest.model_validate(
            self._request_body(request)
        )
        async for chunk in self.native_events(native_request):
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
                        thinking_text += reasoning_delta
                        yield PartDeltaEvent(
                            index=next_part_index - 1,
                            delta=ThinkingDelta(
                                content_delta=(
                                    reasoning_delta
                                    if self._capabilities.thinking == "raw"
                                    else None
                                ),
                                summary_delta=(
                                    reasoning_delta
                                    if self._capabilities.thinking == "summary"
                                    else None
                                ),
                            ),
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
                            arguments = function_delta.arguments or ""
                            state = {
                                "part_index": next_part_index,
                                "tool_call_id": tool_call_delta.id,
                                "tool_name": function_delta.name,
                                "args": arguments,
                            }
                            tool_calls[tool_call_delta.index] = state
                            yield PartStartEvent(
                                index=next_part_index,
                                part=ToolCall(
                                    tool_call_id=tool_call_delta.id,
                                    tool_name=function_delta.name,
                                    args=arguments,
                                ),
                            )
                            next_part_index += 1
                            continue

                        name_delta = function_delta.name if function_delta is not None else None
                        args_delta = (
                            function_delta.arguments if function_delta is not None else None
                        )
                        if name_delta == state["tool_name"]:
                            name_delta = None
                        tool_call_id_delta = tool_call_delta.id
                        if tool_call_id_delta == state["tool_call_id"]:
                            tool_call_id_delta = None
                        if name_delta is not None:
                            state["tool_name"] += name_delta
                        if args_delta is not None:
                            state["args"] += args_delta
                        if tool_call_id_delta is not None:
                            state["tool_call_id"] = tool_call_id_delta
                        yield PartDeltaEvent(
                            index=state["part_index"],
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
                    for tool_call_index in sorted(tool_calls):
                        state = tool_calls[tool_call_index]
                        yield PartEndEvent(
                            index=state["part_index"],
                            part=ToolCall(
                                tool_call_id=state["tool_call_id"],
                                tool_name=state["tool_name"],
                                args=state["args"],
                            ),
                        )
                    tool_calls.clear()

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
