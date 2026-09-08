"""Translate OpenAI Responses and Chat Completions protocol boundaries.

See `docs/protocols/openai-responses.md` and
`docs/protocols/openai-chat-completions.md` for supported semantics and
compatibility mappings.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, NoReturn
from uuid import uuid4

from openai import APIStatusError as OpenAIAPIStatusError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import BadRequestError as OpenAIBadRequestError
from openai import NotFoundError as OpenAINotFoundError
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import BaseModel

from molto.adapters.base import BaseAdapter
from molto.providers import ReasoningRepresentation
from molto.types.exceptions import (
    APIException,
    UnsupportedInput,
)
from molto.types.normalized import (
    FunctionTool,
    FunctionToolChoice,
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
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolReturn,
    UserInput,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsResponse,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesResponse,
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


def _raise_openai_error(error: Exception) -> NoReturn:
    cause = error.__cause__
    if (not isinstance(error, APIException)) or cause is None:
        raise error
    error_map = {
        400: OpenAIBadRequestError,
        401: OpenAIAuthenticationError,
        404: OpenAINotFoundError,
    }
    error_type = error_map.get(getattr(cause, "status_code", None), OpenAIAPIStatusError)
    raise error_type(
        error.message,
        response=error.response,
        body=error.body,
    ) from error


_RESPONSES_TRANSLATABLE_FIELDS = frozenset(
    {
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "model",
        "parallel_tool_calls",
        "reasoning",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
    }
)

_CHAT_TRANSLATABLE_FIELDS = frozenset(
    {
        "messages",
        "model",
        "max_completion_tokens",
        "max_tokens",
        "parallel_tool_calls",
        "reasoning_effort",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
    }
)


def _function_tool(function: dict[str, Any]) -> FunctionTool:
    """Normalize one protocol's function definition shape."""
    return FunctionTool(
        name=function["name"],
        description=function.get("description"),
        parameters=function.get("parameters") or {},
        strict=function.get("strict") or False,
    )


def _responses_tools(tools: list[Any] | None) -> list[FunctionTool] | None:
    if tools is None:
        return None
    normalized: list[FunctionTool] = []
    for tool in tools:
        tool = dict(tool)
        if tool.get("type") != "function":
            raise UnsupportedInput(f"Unsupported Responses tool type: {tool.get('type')}")
        if tool.get("allowed_callers") not in (None, ["direct"]):
            raise UnsupportedInput(
                "Only directly callable Responses function tools are supported."
            )
        if tool.get("defer_loading"):
            raise UnsupportedInput("Deferred Responses function tools are not supported.")
        if tool.get("output_schema") is not None:
            raise UnsupportedInput("Responses function output schemas are not supported.")
        normalized.append(_function_tool(tool))
    return normalized


def _responses_tool_choice(choice: Any) -> ToolChoice | None:
    if choice is None or isinstance(choice, str):
        return choice
    choice = dict(choice)
    if choice.get("type") != "function":
        raise UnsupportedInput(f"Unsupported Responses tool choice: {choice.get('type')}")
    return FunctionToolChoice(name=choice["name"])


def _chat_tools(tools: list[Any] | None) -> list[FunctionTool] | None:
    if tools is None:
        return None
    normalized: list[FunctionTool] = []
    for tool in tools:
        tool = dict(tool)
        if tool.get("type") != "function":
            raise UnsupportedInput(f"Unsupported Chat Completions tool type: {tool.get('type')}")
        normalized.append(_function_tool(dict(tool["function"])))
    return normalized


def _chat_tool_choice(choice: Any) -> ToolChoice | None:
    if choice is None or isinstance(choice, str):
        return choice
    choice = dict(choice)
    if choice.get("type") != "function":
        raise UnsupportedInput(f"Unsupported Chat Completions tool choice: {choice.get('type')}")
    return FunctionToolChoice(name=choice["function"]["name"])


def _validate_responses_function_metadata(
    item: dict[str, Any],
    *,
    statuses: set[str | None],
) -> None:
    caller = item.get("caller")
    if caller is not None and caller.get("type") != "direct":
        raise UnsupportedInput("Only direct Responses function calls are supported.")
    if item.get("namespace") is not None:
        raise UnsupportedInput("Namespaced Responses function calls are not supported.")
    if item.get("status") not in statuses:
        raise UnsupportedInput(
            f"Unsupported Responses function call status: {item.get('status')!r}."
        )


def _reasoning_part_added(
    representation: ReasoningRepresentation,
    *,
    item_id: str,
    output_index: int,
    text: str = "",
) -> dict[str, Any]:
    """Build the event that opens a translated reasoning part's text lifecycle."""
    if representation == "raw":
        return {
            "type": "response.content_part.added",
            "content_index": 0,
            "item_id": item_id,
            "output_index": output_index,
            "part": {"type": "reasoning_text", "text": text},
        }
    return {
        "type": "response.reasoning_summary_part.added",
        "item_id": item_id,
        "output_index": output_index,
        "part": {"type": "summary_text", "text": text},
        "summary_index": 0,
    }


def _reasoning_part_completed(
    representation: ReasoningRepresentation,
    *,
    item_id: str,
    output_index: int,
    part: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the events that close a translated reasoning part's text lifecycle."""
    if representation == "raw":
        return [
            {
                "type": "response.reasoning_text.done",
                "content_index": 0,
                "item_id": item_id,
                "output_index": output_index,
                "text": part["text"],
            },
            {
                "type": "response.content_part.done",
                "content_index": 0,
                "item_id": item_id,
                "output_index": output_index,
                "part": part,
            },
        ]
    return [
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": item_id,
            "output_index": output_index,
            "summary_index": 0,
            "text": part["text"],
        },
        {
            "type": "response.reasoning_summary_part.done",
            "item_id": item_id,
            "output_index": output_index,
            "part": part,
            "summary_index": 0,
        },
    ]


def _validate_completed_responses_reasoning(part: Thinking) -> Thinking:
    if part.summary and not part.signature:
        raise UnsupportedInput("A Responses reasoning summary requires a signature.")
    return part


class OpenAIResponsesAdapter(
    BaseAdapter[
        OpenAIResponsesCreateRequest,
        OpenAIResponsesStreamEvent,
        OpenAIResponsesResponse,
    ]
):
    """Translate the downstream OpenAI Responses protocol to and from normalized IR."""

    async def _project_response(
        self,
        events: AsyncIterator[OpenAIResponsesStreamEvent],
        request: NormalizedRequest,
    ) -> OpenAIResponsesResponse:
        del request
        response: OpenAIResponsesResponse | None = None
        async for event in events:
            if event.type in {"response.completed", "response.incomplete"}:
                response = event.response
        if response is None:
            raise ValueError("Responses event stream ended without a terminal response.")
        return response

    def _normalize_message(
        self,
        message: dict[str, Any],
    ) -> list[SystemPrompt | UserInput | Thinking | TextOutput | ToolCall | ToolReturn]:
        message_type = message.get("type", "message")
        if message_type == "reasoning":
            content = message.get("content") or []
            summary = message.get("summary") or []
            if not all(part.get("type") == "reasoning_text" for part in content):
                raise UnsupportedInput("Reasoning content should only contain `reasoning_text`.")
            if not all(part.get("type") == "summary_text" for part in summary):
                raise UnsupportedInput("Reasoning summary should only contain `summary_text`.")
            return [
                _validate_completed_responses_reasoning(
                    Thinking(
                        content="".join(part["text"] for part in content),
                        summary="".join(part["text"] for part in summary),
                        id=message.get("id"),
                        signature=message.get("encrypted_content"),
                    )
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
                content: list[TextInput | ImageUrlInput] = []
                for part in output:
                    part_type = part.get("type") if isinstance(part, dict) else None
                    if part_type == "input_text":
                        content.append(TextInput(content=part["text"]))
                    elif part_type == "input_image":
                        image_url = part.get("image_url")
                        if image_url is None:
                            raise UnsupportedInput("Responses input_image requires image_url.")
                        content.append(ImageUrlInput(url=image_url, detail=part.get("detail")))
                    else:
                        raise UnsupportedInput(
                            f"Unsupported Responses function output content: {part_type}."
                        )
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
            raise UnsupportedInput(f"Unsupported Responses input type: {message_type}")

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
                        raise UnsupportedInput(
                            f"Responses {part_type} prompt_cache_breakpoint is not "
                            "supported for translation."
                        )
                    if part_type == "input_text":
                        parts.append(TextInput(content=part["text"]))
                    elif part_type == "input_image":
                        image_url = part.get("image_url")
                        if image_url is None:
                            raise UnsupportedInput("Responses input_image requires image_url.")
                        parts.append(ImageUrlInput(url=image_url, detail=part.get("detail")))
                    else:
                        raise UnsupportedInput(
                            f"Unsupported Responses user content type: {part_type}"
                        )
                return [UserInput(content=parts)]
            if role not in {"system", "developer", "assistant"}:
                raise UnsupportedInput(f"Unsupported Responses message role: {role}")

            expected_part_type = "output_text" if role == "assistant" else "input_text"
            if any(
                not isinstance(part, dict) or part.get("type") != expected_part_type
                for part in content
            ):
                raise UnsupportedInput(f"Responses {role} message content cannot be translated.")
            text_parts = [part["text"] for part in content]
        else:
            raise UnsupportedInput("Responses message content must be text.")

        if role in {"system", "developer"}:
            return [SystemPrompt(content=text) for text in text_parts]
        if role == "user":
            return [UserInput(content=[TextInput(content=text) for text in text_parts])]
        if role == "assistant":
            return [TextOutput(content=text) for text in text_parts]
        raise UnsupportedInput(f"Unsupported Responses message role: {role}")

    def _normalize_request(self, request: OpenAIResponsesCreateRequest) -> NormalizedRequest:
        self.reject_unsupported_request_fields(
            request,
            _RESPONSES_TRANSLATABLE_FIELDS,
            protocol="Responses",
        )
        unsupported_includes = set(request.include or []) - {"reasoning.encrypted_content"}
        if unsupported_includes:
            raise UnsupportedInput(
                "Unsupported Responses include values for cross-protocol translation: "
                f"{sorted(unsupported_includes)}."
            )
        if request.store is True:
            raise UnsupportedInput(
                "Cross-protocol Responses requests do not support `store=true`."
            )
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
                    raise UnsupportedInput(f"Unsupported Responses input value: {type(item)}")
                normalized_input.extend(self._normalize_message(item))

        if request.instructions is not None:
            normalized_input = [
                SystemPrompt(content=request.instructions),
                *(normalized_input or []),
            ]

        return NormalizedRequest(
            input=normalized_input,
            max_output_tokens=request.max_output_tokens,
            max_tool_calls=request.max_tool_calls,
            model=request.model,
            parallel_tool_calls=request.parallel_tool_calls,
            reasoning=request.reasoning,
            stream=request.stream,
            stream_options=request.stream_options,
            temperature=request.temperature,
            tool_choice=_responses_tool_choice(request.tool_choice),
            tools=_responses_tools(request.tools),
            top_p=request.top_p,
        )

    async def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        sequence_number = 0
        response_start: ResponseStartEvent | None = None
        item_ids: dict[int, str] = {}
        reasoning_text_started: set[int] = set()
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
                part = normalized_event.part
                output_index = normalized_event.index
                if isinstance(part, Thinking):
                    if self._capabilities.thinking == "raw":
                        if part.summary or part.signature:
                            raise ValueError(
                                "Raw Responses reasoning cannot contain a summary or signature."
                            )
                    elif part.content:
                        raise ValueError("Summarized Responses reasoning cannot contain raw text.")
                    item_id = part.id or uuid4().hex
                    item: dict[str, Any] = {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "content": [],
                        "summary": [],
                    }
                    if part.signature:
                        item["encrypted_content"] = part.signature
                elif isinstance(part, ToolCall):
                    item_id = part.id or f"fc_{uuid4().hex}"
                    item = {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "arguments": part.args or "",
                        "call_id": part.tool_call_id,
                        "name": part.tool_name,
                    }
                elif isinstance(part, TextOutput):
                    item_id = part.id or f"msg_{uuid4().hex}"
                    item = {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "content": [],
                        "role": "assistant",
                    }
                else:
                    raise ValueError(f"Unsupported output part: {type(part)}")
                item_ids[output_index] = item_id
                yield event(
                    {
                        "type": "response.output_item.added",
                        "item": item,
                        "output_index": output_index,
                    }
                )
                # Seed the part lifecycle from a non-empty start snapshot so the
                # streamed text stays consistent with the completed item.
                if isinstance(part, Thinking) and (part.content or part.summary):
                    representation = self._capabilities.thinking
                    reasoning_text_started.add(output_index)
                    yield event(
                        _reasoning_part_added(
                            representation,
                            item_id=item_id,
                            output_index=output_index,
                            text=part.content if representation == "raw" else part.summary,
                        )
                    )
                elif isinstance(part, TextOutput):
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
                                "text": part.content,
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
                    delta = normalized_event.delta
                    if self._capabilities.thinking == "raw":
                        if delta.summary_delta or delta.signature_delta:
                            raise ValueError(
                                "Raw Responses reasoning cannot contain a summary or signature."
                            )
                        representation = "raw"
                        text_delta = delta.content_delta
                    else:
                        if delta.content_delta:
                            raise ValueError(
                                "Summarized Responses reasoning cannot contain raw text."
                            )
                        representation = "summary"
                        text_delta = delta.summary_delta
                    if not text_delta:
                        # Responses has no mid-stream signature event; a signature-only
                        # delta is preserved through the completed item's
                        # encrypted_content instead.
                        continue
                    if normalized_event.index not in reasoning_text_started:
                        reasoning_text_started.add(normalized_event.index)
                        yield event(
                            _reasoning_part_added(
                                representation,
                                item_id=item_id,
                                output_index=normalized_event.index,
                            )
                        )
                    yield event(
                        {
                            "type": (
                                "response.reasoning_text.delta"
                                if representation == "raw"
                                else "response.reasoning_summary_text.delta"
                            ),
                            "delta": text_delta,
                            "item_id": item_id,
                            "output_index": normalized_event.index,
                            "content_index" if representation == "raw" else "summary_index": 0,
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
                completed_item: dict[str, Any]

                if isinstance(part, Thinking):
                    if self._capabilities.thinking == "raw":
                        if part.summary or part.signature:
                            raise ValueError(
                                "Raw Responses reasoning cannot contain a summary or signature."
                            )
                    elif part.content:
                        raise ValueError("Summarized Responses reasoning cannot contain raw text.")
                    if part.summary and not part.signature:
                        raise ValueError("A Responses reasoning summary requires a signature.")
                    representation = self._capabilities.thinking
                    completed_item = {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "completed",
                        "content": [],
                        "summary": [],
                    }
                    if part.signature:
                        completed_item["encrypted_content"] = part.signature
                    # An opaque or completely empty reasoning part has no text
                    # lifecycle to project; its completed item carries only the
                    # encrypted snapshot, when present.
                    if output_index in reasoning_text_started or part.content or part.summary:
                        completed_part = {
                            "type": (
                                "reasoning_text" if representation == "raw" else "summary_text"
                            ),
                            "text": part.content if representation == "raw" else part.summary,
                        }
                        if output_index not in reasoning_text_started:
                            # The part text arrived without deltas; open the text
                            # lifecycle at completion so it stays balanced.
                            part_events.append(
                                _reasoning_part_added(
                                    representation,
                                    item_id=item_id,
                                    output_index=output_index,
                                )
                            )
                        part_events.extend(
                            _reasoning_part_completed(
                                representation,
                                item_id=item_id,
                                output_index=output_index,
                                part=completed_part,
                            )
                        )
                        completed_item["content" if representation == "raw" else "summary"] = [
                            completed_part
                        ]

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

    def _handle_error(self, error: Exception) -> NoReturn:
        cause = error.__cause__
        if self.native_request is None and isinstance(cause, OpenAIAPIStatusError):
            # Same-protocol passthrough: the upstream exception is already
            # downstream-native, so preserve its SDK subclass. Reconstruct a
            # fresh instance rather than re-raise the cause: chaining it
            # `from error` would make the two exceptions each other's cause.
            # This only works on errors that have a Normalized exception class in molto
            # TODO: Handle bare `APIError` (e.g. mid-stream error events): it
            # has no `APIStatusError` cause and passes through unchanged until
            # an IR recognition rule exists.
            raise type(cause)(
                cause.message,
                response=cause.response,
                body=cause.body,
            ) from error
        _raise_openai_error(error)


class OpenAIChatCompletionsAdapter(
    BaseAdapter[
        OpenAIChatCompletionsCreateRequest,
        OpenAIChatCompletionsStreamChunk,
        OpenAIChatCompletionsResponse,
    ]
):
    """Translate Chat Completions requests and normalized response events."""

    async def _project_response(
        self,
        events: AsyncIterator[OpenAIChatCompletionsStreamChunk],
        request: NormalizedRequest,
    ) -> OpenAIChatCompletionsResponse:
        del request
        state = ChatCompletionStreamState()
        finished = False
        async for chunk in events:
            state.handle_chunk(chunk)
            finished = finished or any(
                choice.finish_reason is not None for choice in chunk.choices
            )

        if not finished:
            raise ValueError("Chat Completions event stream ended without a finish reason.")

        # The SDK accumulator uses parsed snapshot models and retains streaming-only
        # tool indices. Strip those fields before validating the native non-streaming
        # response model.
        data = state.current_completion_snapshot.model_dump(
            exclude={
                "choices": {
                    "__all__": {
                        "message": {
                            "parsed": True,
                            "tool_calls": {
                                "__all__": {
                                    "index": True,
                                    "function": {"parsed_arguments": True},
                                }
                            },
                        }
                    }
                }
            }
        )
        for choice in data["choices"]:
            choice["message"]["role"] = "assistant"
        return OpenAIChatCompletionsResponse.model_validate(data)

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
                    raise UnsupportedInput(f"Unsupported Chat user content type: {part_type}")
            return [UserInput(content=parts)]

        if role == "tool":
            if isinstance(content, str):
                tool_content = [TextInput(content=content)]
            elif isinstance(content, list):
                if any(
                    isinstance(part, dict) and part.get("type") == "image_url" for part in content
                ):
                    raise UnsupportedInput(
                        "Chat Completions cannot represent image tool result content."
                    )
                text_parts = [
                    part.get("text")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if len(text_parts) != len(content) or not all(
                    isinstance(text, str) for text in text_parts
                ):
                    raise UnsupportedInput("Only text Chat Completions tool content is supported.")
                tool_content = [TextInput(content=text) for text in text_parts]
            else:
                raise UnsupportedInput("Chat Completions tool content must be text.")
            return [
                ToolReturn(
                    tool_call_id=message["tool_call_id"],
                    tool_name=None,
                    content=tool_content,
                )
            ]

        if role != "assistant" and not isinstance(content, str):
            raise UnsupportedInput("Only string non-user Chat Completions content is supported.")

        if role in {"system", "developer"}:
            return [SystemPrompt(content=content)]
        if role == "user":
            return [UserInput(content=[TextInput(content=content)])]
        if role == "assistant":
            if content is not None and not isinstance(content, str):
                raise UnsupportedInput(
                    "Only string Chat Completions assistant content is supported."
                )
            items: list[Thinking | TextOutput | ToolCall] = []
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is not None:
                if not isinstance(reasoning_content, str):
                    raise UnsupportedInput("Chat Completions reasoning_content must be a string.")
                items.append(self._classify_thinking(reasoning_content))
            if content is not None:
                items.append(TextOutput(content=content))
            for tool_call in message.get("tool_calls") or []:
                tool_call = dict(tool_call)
                if tool_call.get("type") != "function":
                    raise UnsupportedInput(
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
        raise UnsupportedInput(f"Unsupported Chat Completions message role: {role}")

    def _normalize_request(self, request: OpenAIChatCompletionsCreateRequest) -> NormalizedRequest:
        self.reject_unsupported_request_fields(
            request,
            _CHAT_TRANSLATABLE_FIELDS,
            protocol="Chat Completions",
        )
        if request.max_completion_tokens is not None and request.max_tokens is not None:
            raise UnsupportedInput(
                "Chat Completions cannot set both `max_completion_tokens` and `max_tokens`."
            )
        normalized_input: list[
            SystemPrompt | UserInput | Thinking | TextOutput | ToolCall | ToolReturn
        ] = []
        for message in request.messages:
            normalized_input.extend(self._normalize_message(dict(message)))

        stream_options = request.stream_options or {}
        include_obfuscation = stream_options.get("include_obfuscation")
        if stream_options.get("include_usage") is False:
            raise UnsupportedInput("Cross-protocol Chat Completions always includes usage.")
        data: dict[str, Any] = {
            "input": normalized_input,
            "model": request.model,
            "stream": request.stream,
        }
        if include_obfuscation is not None:
            data["stream_options"] = {"include_obfuscation": include_obfuscation}
        max_output_tokens = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens
        )
        if max_output_tokens is not None:
            data["max_output_tokens"] = max_output_tokens
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

        def delta_chunk(
            metadata: ResponseStartEvent | ResponseEndEvent, delta: dict[str, Any]
        ) -> BaseModel:
            """Build a choice-delta chunk that does not finish the response."""
            return chunk(
                metadata,
                choices=[
                    {
                        "index": 0,
                        "delta": delta,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            )

        async for normalized_event in events:
            if isinstance(normalized_event, ResponseStartEvent):
                response_start = normalized_event

            elif isinstance(normalized_event, PartStartEvent):
                if not isinstance(normalized_event.part, (Thinking, TextOutput, ToolCall)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                if response_start is None:
                    raise ValueError("Received a part before its response started.")
                # A part start may already carry an initial snapshot; preserve it
                # in the stream instead of waiting for deltas that may not exist.
                if isinstance(normalized_event.part, Thinking):
                    part = normalized_event.part
                    if part.signature:
                        raise ValueError("Chat Completions cannot preserve reasoning signatures.")
                    if self._capabilities.thinking == "raw":
                        if part.summary:
                            raise ValueError(
                                "Raw Chat Completions reasoning cannot contain a summary."
                            )
                        start_text = part.content
                    else:
                        if part.content:
                            raise ValueError(
                                "Summarized Chat Completions reasoning cannot contain raw text."
                            )
                        start_text = part.summary
                elif isinstance(normalized_event.part, TextOutput):
                    start_text = normalized_event.part.content
                else:
                    start_text = ""
                if not initial_chunk_sent:
                    initial_chunk_sent = True
                    if isinstance(normalized_event.part, Thinking):
                        delta = {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": start_text,
                        }
                    elif isinstance(normalized_event.part, TextOutput):
                        delta = {"role": "assistant", "content": start_text, "refusal": None}
                    else:
                        delta = {"role": "assistant", "content": None, "refusal": None}
                    yield delta_chunk(response_start, delta)
                elif start_text:
                    key = (
                        "reasoning_content"
                        if isinstance(normalized_event.part, Thinking)
                        else "content"
                    )
                    yield delta_chunk(response_start, {key: start_text})
                if isinstance(normalized_event.part, ToolCall):
                    tool_index = len(tool_indices)
                    tool_indices[normalized_event.index] = tool_index
                    yield delta_chunk(
                        response_start,
                        {
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
                    )

            elif isinstance(normalized_event, PartDeltaEvent):
                if response_start is None:
                    raise ValueError("Received a delta before its response started.")
                if isinstance(normalized_event.delta, ThinkingDelta):
                    thinking_delta = normalized_event.delta
                    if thinking_delta.signature_delta:
                        raise ValueError("Chat Completions cannot preserve reasoning signatures.")
                    if self._capabilities.thinking == "raw":
                        if thinking_delta.summary_delta:
                            raise ValueError(
                                "Raw Chat Completions reasoning cannot contain a summary."
                            )
                        reasoning_delta = thinking_delta.content_delta
                    else:
                        if thinking_delta.content_delta:
                            raise ValueError(
                                "Summarized Chat Completions reasoning cannot contain raw text."
                            )
                        reasoning_delta = thinking_delta.summary_delta
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
                yield delta_chunk(response_start, delta)

            elif isinstance(normalized_event, PartEndEvent):
                if not isinstance(normalized_event.part, (Thinking, TextOutput, ToolCall)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                if isinstance(normalized_event.part, Thinking):
                    part = normalized_event.part
                    if part.signature:
                        raise ValueError("Chat Completions cannot preserve reasoning signatures.")
                    if self._capabilities.thinking == "raw" and part.summary:
                        raise ValueError(
                            "Raw Chat Completions reasoning cannot contain a summary."
                        )
                    if self._capabilities.thinking == "summary" and part.content:
                        raise ValueError(
                            "Summarized Chat Completions reasoning cannot contain raw text."
                        )

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

    def _handle_error(self, error: Exception) -> NoReturn:
        cause = error.__cause__
        if self.native_request is None and isinstance(cause, OpenAIAPIStatusError):
            # TODO: Handle bare `APIError` (e.g. mid-stream error events); see
            # OpenAIResponsesAdapter._handle_error for the passthrough rule.
            raise type(cause)(
                cause.message,
                response=cause.response,
                body=cause.body,
            ) from error
        _raise_openai_error(error)
