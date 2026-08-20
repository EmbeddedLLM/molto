"""Translate OpenAI Responses and Chat Completions protocol boundaries.

See ``docs/openai-responses-cc-parity.md`` and
``docs/openai-responses-anthropic-messages-parity.md`` for supported semantics,
compatibility mappings, and testing policy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx
import openai
from pydantic import BaseModel

from molto.adapters.base import BaseAdapter
from molto.types.normalized import (
    FunctionTool,
    FunctionToolChoice,
    ImageUrlInput,
    ModelResponseStreamEvent,
    NormalizedAPIStatusError,
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
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolReturn,
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


def _openai_error_type(error: NormalizedAPIStatusError) -> str:
    if error.source_protocol == "openai" and error.source_type is not None:
        return error.source_type
    if error.category == "rate_limit":
        return "rate_limit_error"
    if error.category in {"overloaded", "server"}:
        return "server_error"
    return "invalid_request_error"


def _openai_api_status_error(error: NormalizedAPIStatusError) -> openai.APIStatusError:
    error_body = {
        "message": error.message,
        "type": _openai_error_type(error),
        "param": error.param,
        "code": error.code,
    }
    envelope = {"error": error_body}
    headers = httpx.Headers(error.response_headers)
    for header in ("content-encoding", "content-length", "transfer-encoding"):
        if header in headers:
            del headers[header]
    if "request-id" in headers:
        del headers["request-id"]
    if error.request_id is not None:
        headers["x-request-id"] = error.request_id
    request = httpx.Request(error.request_method, error.request_url)
    response = httpx.Response(
        error.status_code,
        headers=headers,
        json=envelope,
        request=request,
    )
    exception_type: type[openai.APIStatusError]
    if error.status_code == 400:
        exception_type = openai.BadRequestError
    elif error.status_code == 401:
        exception_type = openai.AuthenticationError
    elif error.status_code == 403:
        exception_type = openai.PermissionDeniedError
    elif error.status_code == 404:
        exception_type = openai.NotFoundError
    elif error.status_code == 409:
        exception_type = openai.ConflictError
    elif error.status_code == 422:
        exception_type = openai.UnprocessableEntityError
    elif error.status_code == 429:
        exception_type = openai.RateLimitError
    elif error.status_code >= 500:
        exception_type = openai.InternalServerError
    else:
        exception_type = openai.APIStatusError
    return exception_type(
        f"Error code: {error.status_code} - {envelope}",
        response=response,
        body=error_body,
    )


def _responses_tools(tools: list[Any] | None) -> list[FunctionTool] | None:
    if tools is None:
        return None
    normalized: list[FunctionTool] = []
    for tool in tools:
        tool = dict(tool)
        if tool.get("type") != "function":
            raise ValueError(f"Unsupported Responses tool type: {tool.get('type')}")
        if tool.get("allowed_callers") not in (None, ["direct"]):
            raise ValueError("Only directly callable Responses function tools are supported.")
        if tool.get("defer_loading"):
            raise ValueError("Deferred Responses function tools are not supported.")
        if tool.get("output_schema") is not None:
            raise ValueError("Responses function output schemas are not supported.")
        normalized.append(
            FunctionTool(
                name=tool["name"],
                description=tool.get("description"),
                parameters=tool.get("parameters") or {},
                strict=tool.get("strict") or False,
            )
        )
    return normalized


def _responses_tool_choice(choice: Any) -> ToolChoice | None:
    if choice is None or isinstance(choice, str):
        return choice
    choice = dict(choice)
    if choice.get("type") != "function":
        raise ValueError(f"Unsupported Responses tool choice: {choice.get('type')}")
    return FunctionToolChoice(name=choice["name"])


def _chat_tools(tools: list[Any] | None) -> list[FunctionTool] | None:
    if tools is None:
        return None
    normalized: list[FunctionTool] = []
    for tool in tools:
        tool = dict(tool)
        if tool.get("type") != "function":
            raise ValueError(f"Unsupported Chat Completions tool type: {tool.get('type')}")
        function = dict(tool["function"])
        normalized.append(
            FunctionTool(
                name=function["name"],
                description=function.get("description"),
                parameters=function.get("parameters") or {},
                strict=function.get("strict") or False,
            )
        )
    return normalized


def _chat_tool_choice(choice: Any) -> ToolChoice | None:
    if choice is None or isinstance(choice, str):
        return choice
    choice = dict(choice)
    if choice.get("type") != "function":
        raise ValueError(f"Unsupported Chat Completions tool choice: {choice.get('type')}")
    return FunctionToolChoice(name=choice["function"]["name"])


def _validate_responses_function_metadata(
    item: dict[str, Any],
    *,
    statuses: set[str | None],
) -> None:
    caller = item.get("caller")
    if caller is not None and caller.get("type") != "direct":
        raise ValueError("Only direct Responses function calls are supported.")
    if item.get("namespace") is not None:
        raise ValueError("Namespaced Responses function calls are not supported.")
    if item.get("status") not in statuses:
        raise ValueError(f"Unsupported Responses function call status: {item.get('status')!r}.")


class OpenAIResponsesAdapter(
    BaseAdapter[OpenAIResponsesCreateRequest, OpenAIResponsesStreamEvent]
):
    """Translate the downstream OpenAI Responses protocol to and from normalized IR."""

    def _project_api_status_error(self, error: NormalizedAPIStatusError) -> openai.APIStatusError:
        return _openai_api_status_error(error)

    def _normalize_message(
        self,
        message: dict[str, Any],
    ) -> list[SystemPrompt | UserInput | Thinking | TextOutput | ToolCall | ToolReturn]:
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
        if message_type == "function_call":
            _validate_responses_function_metadata(message, statuses={None, "completed"})
            return [
                ToolCall(
                    id=message.get("id"),
                    tool_call_id=message["call_id"],
                    tool_name=message["name"],
                    args=message["arguments"],
                )
            ]
        if message_type == "function_call_output":
            _validate_responses_function_metadata(message, statuses={None, "completed"})
            output = message["output"]
            if isinstance(output, list):
                unsupported = [
                    part.get("type") for part in output if part.get("type") != "input_text"
                ]
                if unsupported:
                    raise ValueError(
                        f"Unsupported Responses function output content: {unsupported}."
                    )
                content = [TextInput(content=part["text"]) for part in output]
            else:
                content = [TextInput(content=output)]
            return [
                ToolReturn(
                    id=message.get("id"),
                    tool_call_id=message["call_id"],
                    tool_name=message.get("name"),
                    content=content,
                    outcome="success",
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
        normalized_input: (
            list[SystemPrompt | UserInput | Thinking | TextOutput | ToolCall | ToolReturn] | None
        )
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
        request_data["tools"] = _responses_tools(request.tools)
        request_data["tool_choice"] = _responses_tool_choice(request.tool_choice)
        return NormalizedRequest.model_validate(request_data)

    async def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        sequence_number = 0
        response_start: ResponseStartEvent | None = None
        item_ids: dict[int, str] = {}
        thinking_representations: dict[int, str] = {}
        output_items: dict[int, dict[str, Any]] = {}

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
                if not isinstance(normalized_event.part, (TextOutput, Thinking, ToolCall)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                is_thinking = isinstance(normalized_event.part, Thinking)
                output_index = normalized_event.index
                item_id = normalized_event.part.id or (
                    f"{'' if is_thinking else 'fc_' if isinstance(normalized_event.part, ToolCall) else 'msg_'}{uuid4().hex}"
                )
                item_ids[output_index] = item_id
                thinking_representations.pop(output_index, None)
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
                        "type": "function_call",
                        "status": "in_progress",
                        "arguments": normalized_event.part.args or "",
                        "call_id": normalized_event.part.tool_call_id,
                        "name": normalized_event.part.tool_name,
                    }
                    if isinstance(normalized_event.part, ToolCall)
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
                if isinstance(normalized_event.part, TextOutput):
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
                if not isinstance(
                    normalized_event.delta, (TextDelta, ThinkingDelta, ToolCallDelta)
                ):
                    raise ValueError(f"Unsupported output delta: {type(normalized_event.delta)}")
                item_id = item_ids.get(normalized_event.index)
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
                    thinking_representation = thinking_representations.get(normalized_event.index)
                    if thinking_representation is None:
                        thinking_representations[normalized_event.index] = representation
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
                elif isinstance(normalized_event.delta, TextDelta):
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
                else:
                    if normalized_event.delta.tool_name_delta is not None:
                        raise ValueError(
                            "Responses cannot update a function name after call start."
                        )
                    if normalized_event.delta.tool_call_id is not None:
                        raise ValueError("Responses cannot update a call ID after call start.")
                    args_delta = normalized_event.delta.args_delta
                    if args_delta is None:
                        continue
                    if not isinstance(args_delta, str):
                        raise ValueError("Responses function argument deltas must be strings.")
                    yield event(
                        {
                            "type": "response.function_call_arguments.delta",
                            "delta": args_delta,
                            "item_id": item_id,
                            "output_index": normalized_event.index,
                        }
                    )

            elif isinstance(normalized_event, PartEndEvent):
                part = normalized_event.part
                if not isinstance(part, (TextOutput, Thinking, ToolCall)):
                    raise ValueError(f"Unsupported output part: {type(part)}")
                item_id = item_ids.get(normalized_event.index)
                if item_id is None:
                    raise ValueError("Received a part end before its output item started.")
                output_index = normalized_event.index
                part_events: list[dict[str, Any]] = []

                if isinstance(part, Thinking):
                    if part.content and part.summary:
                        raise ValueError(
                            "A translated reasoning part cannot contain both raw reasoning "
                            "and a summary."
                        )
                    representation = thinking_representations.get(output_index)
                    is_opaque_only = (
                        bool(part.signature)
                        and not part.content
                        and not part.summary
                        and representation is None
                    )
                    completed_part = None
                    reasoning_text = ""
                    if not is_opaque_only:
                        representation = representation or (
                            "summary" if part.summary else self._capabilities.thinking
                        )
                        reasoning_text = part.content if representation == "raw" else part.summary
                        completed_part = {
                            "type": (
                                "reasoning_text" if representation == "raw" else "summary_text"
                            ),
                            "text": reasoning_text,
                        }
                        if output_index not in thinking_representations:
                            if representation == "raw":
                                part_events.append(
                                    {
                                        "type": "response.content_part.added",
                                        "content_index": 0,
                                        "item_id": item_id,
                                        "output_index": output_index,
                                        "part": {"type": "reasoning_text", "text": ""},
                                    }
                                )
                            else:
                                part_events.append(
                                    {
                                        "type": "response.reasoning_summary_part.added",
                                        "item_id": item_id,
                                        "output_index": output_index,
                                        "part": {"type": "summary_text", "text": ""},
                                        "summary_index": 0,
                                    }
                                )
                    completed_item = {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "completed",
                        "content": (
                            [completed_part]
                            if representation == "raw" and completed_part is not None
                            else []
                        ),
                        "summary": (
                            [completed_part]
                            if representation == "summary" and completed_part is not None
                            else []
                        ),
                        **({"encrypted_content": part.signature} if part.signature else {}),
                    }
                    if representation == "summary":
                        part_events.extend(
                            [
                                {
                                    "type": "response.reasoning_summary_text.done",
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "summary_index": 0,
                                    "text": reasoning_text,
                                },
                                {
                                    "type": "response.reasoning_summary_part.done",
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "part": completed_part,
                                    "summary_index": 0,
                                },
                            ]
                        )
                    elif representation == "raw":
                        part_events.extend(
                            [
                                {
                                    "type": "response.reasoning_text.done",
                                    "content_index": 0,
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "text": reasoning_text,
                                },
                                {
                                    "type": "response.content_part.done",
                                    "content_index": 0,
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "part": completed_part,
                                },
                            ]
                        )

                elif isinstance(part, TextOutput):
                    completed_part = {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": part.content,
                    }
                    completed_item = {
                        "id": item_id,
                        "type": "message",
                        "status": "completed",
                        "content": [completed_part],
                        "role": "assistant",
                    }
                    part_events.extend(
                        [
                            {
                                "type": "response.output_text.done",
                                "content_index": 0,
                                "item_id": item_id,
                                "output_index": output_index,
                                "text": part.content,
                                "logprobs": [],
                            },
                            {
                                "type": "response.content_part.done",
                                "content_index": 0,
                                "item_id": item_id,
                                "output_index": output_index,
                                "part": completed_part,
                            },
                        ]
                    )

                elif isinstance(part, ToolCall):
                    arguments = part.args or ""
                    completed_item = {
                        "id": item_id,
                        "type": "function_call",
                        "status": "completed",
                        "arguments": arguments,
                        "call_id": part.tool_call_id,
                        "name": part.tool_name,
                    }
                    part_events.append(
                        {
                            "type": "response.function_call_arguments.done",
                            "arguments": arguments,
                            "item_id": item_id,
                            "output_index": output_index,
                        }
                    )

                for part_event in part_events:
                    yield event(part_event)
                output_items[output_index] = completed_item
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
                    output=[output_items[index] for index in sorted(output_items)],
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

    def _project_api_status_error(self, error: NormalizedAPIStatusError) -> openai.APIStatusError:
        return _openai_api_status_error(error)

    def _normalize_message(
        self,
        message: dict[str, Any],
    ) -> list[SystemPrompt | UserInput | Thinking | TextOutput | ToolCall | ToolReturn]:
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

        if role == "tool":
            if isinstance(content, str):
                tool_content = [TextInput(content=content)]
            elif isinstance(content, list):
                text_parts = [
                    part.get("text")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if len(text_parts) != len(content) or not all(
                    isinstance(text, str) for text in text_parts
                ):
                    raise ValueError("Only text Chat Completions tool content is supported.")
                tool_content = [TextInput(content=text) for text in text_parts]
            else:
                raise ValueError("Chat Completions tool content must be text.")
            return [
                ToolReturn(
                    tool_call_id=message["tool_call_id"],
                    tool_name=None,
                    content=tool_content,
                )
            ]

        if role != "assistant" and not isinstance(content, str):
            raise ValueError("Only string non-user Chat Completions content is supported.")

        if role in {"system", "developer"}:
            return [SystemPrompt(content=content)]
        if role == "user":
            return [UserInput(content=[TextInput(content=content)])]
        if role == "assistant":
            if content is not None and not isinstance(content, str):
                raise ValueError("Only string Chat Completions assistant content is supported.")
            items: list[Thinking | TextOutput | ToolCall] = []
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
            if content is not None:
                items.append(TextOutput(content=content))
            for tool_call in message.get("tool_calls") or []:
                tool_call = dict(tool_call)
                if tool_call.get("type") != "function":
                    raise ValueError(
                        f"Unsupported Chat Completions tool call type: {tool_call.get('type')}"
                    )
                function = dict(tool_call["function"])
                items.append(
                    ToolCall(
                        tool_name=function["name"],
                        args=function["arguments"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return items
        raise ValueError(f"Unsupported Chat Completions message role: {role}")

    def _normalize_request(self, request: OpenAIChatCompletionsCreateRequest) -> NormalizedRequest:
        normalized_input: list[
            SystemPrompt | UserInput | Thinking | TextOutput | ToolCall | ToolReturn
        ] = []
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
        if request.tools is not None:
            data["tools"] = _chat_tools(request.tools)
        if request.tool_choice is not None:
            data["tool_choice"] = _chat_tool_choice(request.tool_choice)
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
        tool_indices: dict[int, int] = {}

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
                if not isinstance(normalized_event.part, (Thinking, TextOutput, ToolCall)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                if response_start is None:
                    raise ValueError("Received a part before its response started.")
                if not initial_chunk_sent:
                    initial_chunk_sent = True
                    if isinstance(normalized_event.part, Thinking):
                        delta = {"role": "assistant", "content": None, "reasoning_content": ""}
                    elif isinstance(normalized_event.part, TextOutput):
                        delta = {"role": "assistant", "content": "", "refusal": None}
                    else:
                        delta = {"role": "assistant", "content": None, "refusal": None}
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
                if isinstance(normalized_event.part, ToolCall):
                    tool_index = len(tool_indices)
                    tool_indices[normalized_event.index] = tool_index
                    yield chunk(
                        response_start,
                        choices=[
                            {
                                "index": 0,
                                "delta": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "index": tool_index,
                                            "id": normalized_event.part.tool_call_id,
                                            "type": "function",
                                            "function": {
                                                "name": normalized_event.part.tool_name,
                                                "arguments": normalized_event.part.args or "",
                                            },
                                        }
                                    ],
                                },
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
                elif isinstance(normalized_event.delta, ToolCallDelta):
                    try:
                        tool_index = tool_indices[normalized_event.index]
                    except KeyError as exc:
                        raise ValueError("Received a tool delta before its call started.") from exc
                    function: dict[str, Any] = {}
                    if normalized_event.delta.tool_name_delta is not None:
                        function["name"] = normalized_event.delta.tool_name_delta
                    if normalized_event.delta.args_delta is not None:
                        if not isinstance(normalized_event.delta.args_delta, str):
                            raise ValueError(
                                "Chat Completions function argument deltas must be strings."
                            )
                        function["arguments"] = normalized_event.delta.args_delta
                    tool_call: dict[str, Any] = {"index": tool_index}
                    if normalized_event.delta.tool_call_id is not None:
                        tool_call["id"] = normalized_event.delta.tool_call_id
                    if function:
                        tool_call["function"] = function
                    delta = {"tool_calls": [tool_call]}
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
                if not isinstance(normalized_event.part, (Thinking, TextOutput, ToolCall)):
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
