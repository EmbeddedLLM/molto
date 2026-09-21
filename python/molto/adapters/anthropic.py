"""Translate Anthropic Messages at Molto's downstream protocol boundary.

See `docs/protocols/anthropic-messages.md` for supported semantics and
compatibility mappings.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, NoReturn

from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import NotFoundError as AnthropicNotFoundError
from pydantic_core import from_json

from molto.adapters.base import BaseAdapter
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesResponse,
    AnthropicMessagesStreamEvent,
    AnthropicMessagesStreamEventAdapter,
)
from molto.types.exceptions import (
    APIException,
    UnsupportedInput,
)
from molto.types.normalized import (
    AssistantInput,
    CacheBreakpoint,
    FunctionTool,
    FunctionToolChoice,
    ImageUrlInput,
    Input,
    ModelResponseStreamEvent,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    PromptCache,
    Reasoning,
    ReasoningEffort,
    ResponseEndEvent,
    ResponseStartEvent,
    StreamOptions,
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
    UserInput,
)
from molto.utils.io import json_dumps

_STOP_REASONS: dict[TerminationReason, str] = {
    "max_output_tokens": "max_tokens",
    "model_context_window_exceeded": "model_context_window_exceeded",
    "pause": "pause_turn",
    "refusal": "refusal",
    "stop": "end_turn",
    "stop_sequence": "stop_sequence",
    "tool_call": "tool_use",
}

_EXTENDED_THINKING_EFFORTS: tuple[tuple[int, ReasoningEffort], ...] = (
    (1024, "minimal"),
    (2048, "low"),
    (10_000, "medium"),
    (16_384, "high"),
    (32_768, "xhigh"),
)

_MESSAGES_TRANSLATABLE_FIELDS = frozenset(
    {
        "cache_control",
        "max_tokens",
        "messages",
        "model",
        "output_config",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_p",
    }
)


def _reject_cache_breakpoint(block: dict[str, Any]) -> None:
    """Reject markers on blocks with no corresponding OpenAI content part."""
    if block.get("cache_control") is not None:
        raise UnsupportedInput(
            f"Anthropic {block.get('type')} cache_control is not supported for translation."
        )


def _cache_breakpoint(block: dict[str, Any]) -> CacheBreakpoint | None:
    control = block.get("cache_control")
    return CacheBreakpoint(ttl=control.get("ttl")) if control is not None else None


def _reasoning_effort_from_budget(budget_tokens: int) -> ReasoningEffort:
    return next(
        (effort for budget, effort in _EXTENDED_THINKING_EFFORTS if budget_tokens <= budget),
        "max",
    )


def _normalize_tool(tool: dict[str, Any]) -> FunctionTool:
    if tool.get("type") not in {None, "custom"}:
        raise UnsupportedInput(f"Unsupported Anthropic tool type: {tool.get('type')!r}.")
    unsupported_fields = {
        field
        for field in (
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "eager_input_streaming",
            "input_examples",
        )
        if tool.get(field) not in (None, False, [])
    }
    if unsupported_fields:
        raise UnsupportedInput(
            f"Unsupported Anthropic custom tool fields: {sorted(unsupported_fields)}."
        )
    return FunctionTool(
        name=tool["name"],
        description=tool.get("description"),
        parameters=tool["input_schema"],
        strict=tool.get("strict", False),
    )


class AnthropicMessagesAdapter(
    BaseAdapter[
        AnthropicMessagesCreateRequest,
        AnthropicMessagesStreamEvent,
        AnthropicMessagesResponse,
    ]
):
    """Translate the downstream Anthropic Messages protocol to and from normalized IR."""

    async def _project_response(
        self,
        events: AsyncIterator[AnthropicMessagesStreamEvent],
        request: NormalizedRequest,
    ) -> AnthropicMessagesResponse:
        message: AnthropicMessagesResponse | None = None
        open_blocks: set[int] = set()
        tool_input_deltas: dict[int, list[str]] = {}
        received_message_delta = False
        received_message_stop = False

        async for event in events:
            if event.type == "message_start":
                if message is not None:
                    raise ValueError("Received more than one Anthropic message start.")
                message = event.message.model_copy(deep=True)
                continue

            if message is None:
                raise ValueError(f"Received {event.type} before the Anthropic message start.")

            if event.type == "content_block_start":
                if event.index != len(message.content):
                    raise ValueError(
                        "Anthropic content blocks must start in contiguous index order."
                    )
                message.content.append(event.content_block.model_copy(deep=True))
                open_blocks.add(event.index)
            elif event.type == "content_block_delta":
                if event.index not in open_blocks:
                    raise ValueError("Received an Anthropic content delta outside an open block.")
                content_block = message.content[event.index]
                if event.delta.type == "text_delta":
                    if content_block.type != "text":
                        raise ValueError("Received an Anthropic text delta for a non-text block.")
                    content_block.text += event.delta.text
                elif event.delta.type == "thinking_delta":
                    if content_block.type != "thinking":
                        raise ValueError(
                            "Received an Anthropic thinking delta for a non-thinking block."
                        )
                    content_block.thinking += event.delta.thinking
                elif event.delta.type == "signature_delta":
                    if content_block.type != "thinking":
                        raise ValueError(
                            "Received an Anthropic signature delta for a non-thinking block."
                        )
                    content_block.signature = event.delta.signature
                elif event.delta.type == "input_json_delta":
                    if content_block.type != "tool_use":
                        raise ValueError(
                            "Received an Anthropic tool-input delta for a non-tool block."
                        )
                    tool_input_deltas.setdefault(event.index, []).append(event.delta.partial_json)
                elif event.delta.type == "citations_delta":
                    if content_block.type != "text":
                        raise ValueError(
                            "Received an Anthropic citation delta for a non-text block."
                        )
                    if content_block.citations is None:
                        content_block.citations = []
                    content_block.citations.append(event.delta.citation)
            elif event.type == "content_block_stop":
                if event.index not in open_blocks:
                    raise ValueError("Received an Anthropic content stop outside an open block.")
                content_block = message.content[event.index]
                if event.index in tool_input_deltas:
                    raw_input = "".join(tool_input_deltas.pop(event.index))
                    if raw_input:
                        # Match SDK accumulation when a token limit truncates tool arguments.
                        tool_input = from_json(raw_input, allow_partial=True)
                        if not isinstance(tool_input, dict):
                            raise ValueError("Anthropic tool input must be a JSON object.")
                        if content_block.type != "tool_use":
                            raise ValueError("Received Anthropic tool input for a non-tool block.")
                        content_block.input = tool_input
                open_blocks.remove(event.index)
            elif event.type == "message_delta":
                if open_blocks:
                    raise ValueError(
                        "Received an Anthropic message delta before all blocks ended."
                    )
                message.stop_reason = event.delta.stop_reason
                message.stop_sequence = event.delta.stop_sequence
                if event.delta.stop_details is not None:
                    message.stop_details = event.delta.stop_details
                for field_name, value in event.usage:
                    if value is not None:
                        setattr(message.usage, field_name, value)
                received_message_delta = True
            elif event.type == "message_stop":
                if open_blocks:
                    raise ValueError("Received an Anthropic message stop before all blocks ended.")
                if not received_message_delta:
                    raise ValueError("Received an Anthropic message stop before its final delta.")
                received_message_stop = True

        if message is None or not received_message_stop:
            raise ValueError("Anthropic event stream ended before the message stop event.")
        return message

    @staticmethod
    def _normalize_input_block(block: dict[str, Any]) -> TextInput | ImageUrlInput:
        block_type = block.get("type")
        if block_type == "text":
            return TextInput(content=block["text"], cache_breakpoint=_cache_breakpoint(block))
        if block_type == "image":
            source = block.get("source") or {}
            source_type = source.get("type")
            if source_type == "base64":
                return ImageUrlInput(
                    url=f"data:{source['media_type']};base64,{source['data']}",
                    cache_breakpoint=_cache_breakpoint(block),
                )
            if source_type == "url":
                return ImageUrlInput(url=source["url"], cache_breakpoint=_cache_breakpoint(block))
            raise UnsupportedInput(f"Unsupported Anthropic image source: {source_type}")
        raise UnsupportedInput(f"Unsupported Anthropic user content block: {block_type}")

    @staticmethod
    def _normalize_system_blocks(content_blocks: list[dict[str, Any]]) -> list[SystemPrompt]:
        unsupported = [
            block.get("type") for block in content_blocks if block.get("type") != "text"
        ]
        if unsupported:
            raise UnsupportedInput(f"Unsupported Anthropic system content blocks: {unsupported}")
        return [
            SystemPrompt(content=block["text"], cache_breakpoint=_cache_breakpoint(block))
            for block in content_blocks
        ]

    def _normalize_user_blocks(
        self, content_blocks: list[dict[str, Any]]
    ) -> list[UserInput | ToolReturn]:
        items: list[UserInput | ToolReturn] = []
        user_parts: list[TextInput | ImageUrlInput] = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type in {"text", "image"}:
                user_parts.append(self._normalize_input_block(block))
                continue
            if block_type != "tool_result":
                raise UnsupportedInput(f"Unsupported Anthropic user content block: {block_type}")
            if user_parts:
                raise UnsupportedInput(
                    "Anthropic tool results must precede user content in a message."
                )
            items.append(self._normalize_tool_result(block))
        if user_parts:
            items.append(UserInput(content=user_parts))
        return items

    def _normalize_tool_result(self, block: dict[str, Any]) -> ToolReturn:
        _reject_cache_breakpoint(block)
        content = block.get("content", "")
        if isinstance(content, str):
            normalized_content = [TextInput(content=content)]
        else:
            parts = list(content)
            normalized_content = [self._normalize_input_block(part) for part in parts]
        return ToolReturn(
            content=normalized_content,
            tool_call_id=block["tool_use_id"],
            outcome="failed" if block.get("is_error") else "success",
        )

    def _normalize_assistant_blocks(
        self, content_blocks: list[dict[str, Any]]
    ) -> list[Thinking | TextOutput | ToolCall]:
        items: list[Thinking | TextOutput | ToolCall] = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "thinking":
                _reject_cache_breakpoint(block)
                part = self._classify_thinking(block["thinking"], signature=block.get("signature"))
                if self._capabilities.thinking == "summary" and not part.signature:
                    raise UnsupportedInput("An Anthropic reasoning summary requires a signature.")
                items.append(part)
            elif block_type == "text":
                items.append(
                    TextOutput(content=block["text"], cache_breakpoint=_cache_breakpoint(block))
                )
            elif block_type == "tool_use":
                _reject_cache_breakpoint(block)
                caller = block.get("caller")
                caller_type = (
                    caller.get("type")
                    if isinstance(caller, dict)
                    else getattr(caller, "type", None)
                )
                if caller is not None and caller_type != "direct":
                    raise UnsupportedInput(
                        "Only directly invoked Anthropic tool uses can be translated."
                    )
                items.append(
                    ToolCall(
                        tool_name=block["name"],
                        args=json_dumps(block["input"]),
                        tool_call_id=block["id"],
                    )
                )
            else:
                raise UnsupportedInput(
                    f"Unsupported Anthropic assistant content block: {block_type}"
                )
        return items

    def _normalize_request(self, request: AnthropicMessagesCreateRequest) -> NormalizedRequest:
        self.reject_unsupported_request_fields(
            request,
            _MESSAGES_TRANSLATABLE_FIELDS,
            protocol="Anthropic Messages",
        )
        unsupported_output_config = {
            field
            for field, value in (request.output_config or {}).items()
            if field != "effort" and value is not None
        }
        if unsupported_output_config:
            raise UnsupportedInput(
                "Unsupported Anthropic output_config fields for cross-protocol translation: "
                f"{sorted(unsupported_output_config)}."
            )
        normalized_input: Input = []
        if isinstance(request.system, str):
            normalized_input.append(SystemPrompt(content=request.system))
        elif request.system is not None:
            normalized_input.extend(self._normalize_system_blocks(list(request.system)))

        for message in request.messages:
            if isinstance(message.content, str):
                content_blocks: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
            else:
                content_blocks = [dict(block) for block in message.content]

            if message.role == "system":
                normalized_input.extend(self._normalize_system_blocks(content_blocks))
            elif message.role == "user":
                normalized_input.extend(self._normalize_user_blocks(content_blocks))
            else:
                normalized_input.append(
                    AssistantInput(content=self._normalize_assistant_blocks(content_blocks))
                )

        data: dict[str, Any] = {
            "input": normalized_input,
            "max_output_tokens": request.max_tokens,
            "model": request.model,
            "stream": request.stream,
            "stream_options": StreamOptions(include_obfuscation=False),
        }
        if request.cache_control is not None:
            # Top-level control is Anthropic's opt-in to provider-chosen breakpoints that
            # advance with the conversation; its `ephemeral` type is pinned by the request model.
            data["prompt_cache"] = PromptCache(
                mode="implicit",
                ttl=request.cache_control.get("ttl"),
            )
        if request.tools is not None:
            data["tools"] = [_normalize_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            choice_type = request.tool_choice["type"]
            if choice_type == "auto":
                # Auto is the target protocols' default and Messages also uses it
                # as the carrier for disable_parallel_tool_use.
                pass
            elif choice_type == "none":
                data["tool_choice"] = "none"
            elif choice_type == "any":
                data["tool_choice"] = "required"
            elif choice_type == "tool":
                data["tool_choice"] = FunctionToolChoice(name=request.tool_choice["name"])
            else:
                raise UnsupportedInput(f"Unsupported Anthropic tool choice: {choice_type!r}.")
            if "disable_parallel_tool_use" in request.tool_choice:
                data["parallel_tool_calls"] = not request.tool_choice["disable_parallel_tool_use"]
        if request.thinking is not None:
            thinking_type = request.thinking.get("type")
            if thinking_type == "disabled":
                data["reasoning"] = Reasoning(effort="none")
            elif request.thinking.get("display", "summarized") != "summarized":
                raise UnsupportedInput(
                    "Only Anthropic thinking with summarized display is currently supported."
                )
            elif thinking_type in {"adaptive", "enabled"}:
                output_effort = (
                    request.output_config.get("effort")
                    if request.output_config is not None
                    else None
                )
                effort = output_effort or (
                    "high"
                    if thinking_type == "adaptive"
                    else _reasoning_effort_from_budget(request.thinking["budget_tokens"])
                )
                data["reasoning"] = Reasoning(effort=effort, summary="detailed")
            else:
                raise UnsupportedInput(f"Unsupported Anthropic thinking type: {thinking_type}")
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        normalized = NormalizedRequest.model_validate(data)
        if (
            normalized.prompt_cache is None
            and next(normalized.iter_cache_breakpoints(), None) is not None
        ):
            # Without top-level cache_control, Messages caches only the marked prefixes.
            normalized.prompt_cache = PromptCache(mode="explicit")
        return normalized

    async def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        def event(data: dict[str, Any]) -> AnthropicMessagesStreamEvent:
            return AnthropicMessagesStreamEventAdapter.validate_python(data)

        thinking_parts: set[int] = set()
        thinking_signatures: dict[int, str] = {}

        async for normalized_event in events:
            if isinstance(normalized_event, ResponseStartEvent):
                yield event(
                    {
                        "type": "message_start",
                        "message": {
                            "id": normalized_event.id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": normalized_event.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": 0,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                                "output_tokens": 0,
                                "service_tier": "standard",
                            },
                        },
                    }
                )
            elif isinstance(normalized_event, PartStartEvent):
                # A part start may already carry an initial snapshot; preserve it
                # in the block instead of waiting for deltas that may not exist.
                if isinstance(normalized_event.part, Thinking):
                    part = normalized_event.part
                    if self._capabilities.thinking == "raw":
                        if part.summary or part.signature:
                            raise ValueError(
                                "A raw Anthropic thinking block cannot contain a summary "
                                "or signature."
                            )
                        thinking = part.content
                    else:
                        if part.content:
                            raise ValueError(
                                "An Anthropic summary block cannot contain raw reasoning."
                            )
                        thinking = part.summary
                        if part.signature:
                            thinking_signatures[normalized_event.index] = part.signature
                    thinking_parts.add(normalized_event.index)
                    content_block = {
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": part.signature or "",
                    }
                elif isinstance(normalized_event.part, TextOutput):
                    content_block = {"type": "text", "text": normalized_event.part.content}
                elif isinstance(normalized_event.part, ToolCall):
                    content_block = {
                        "type": "tool_use",
                        "id": normalized_event.part.tool_call_id,
                        "name": normalized_event.part.tool_name,
                        "input": {},
                        "caller": {"type": "direct"},
                    }
                else:
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                yield event(
                    {
                        "type": "content_block_start",
                        "index": normalized_event.index,
                        "content_block": content_block,
                    }
                )
                if isinstance(normalized_event.part, ToolCall) and normalized_event.part.args:
                    yield event(
                        {
                            "type": "content_block_delta",
                            "index": normalized_event.index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": normalized_event.part.args,
                            },
                        }
                    )
            elif isinstance(normalized_event, PartDeltaEvent):
                if isinstance(normalized_event.delta, ThinkingDelta):
                    delta = normalized_event.delta
                    if normalized_event.index not in thinking_parts:
                        raise ValueError("Received a thinking delta before its block started.")
                    if self._capabilities.thinking == "raw":
                        if delta.summary_delta or delta.signature_delta:
                            raise ValueError(
                                "A raw Anthropic thinking block cannot contain a summary "
                                "or signature."
                            )
                        reasoning_delta = delta.content_delta
                    else:
                        if delta.content_delta:
                            raise ValueError(
                                "An Anthropic summary block cannot contain raw reasoning."
                            )
                        reasoning_delta = delta.summary_delta
                    if reasoning_delta:
                        yield event(
                            {
                                "type": "content_block_delta",
                                "index": normalized_event.index,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": reasoning_delta,
                                },
                            }
                        )
                    if delta.signature_delta:
                        thinking_signatures[normalized_event.index] = delta.signature_delta
                        yield event(
                            {
                                "type": "content_block_delta",
                                "index": normalized_event.index,
                                "delta": {
                                    "type": "signature_delta",
                                    "signature": delta.signature_delta,
                                },
                            }
                        )
                elif isinstance(normalized_event.delta, TextDelta):
                    yield event(
                        {
                            "type": "content_block_delta",
                            "index": normalized_event.index,
                            "delta": {
                                "type": "text_delta",
                                "text": normalized_event.delta.content_delta,
                            },
                        }
                    )
                elif isinstance(normalized_event.delta, ToolCallDelta):
                    if (
                        normalized_event.delta.tool_name_delta is not None
                        or normalized_event.delta.tool_call_id is not None
                    ):
                        raise ValueError(
                            "Anthropic cannot update a tool name or tool call ID after "
                            "the content block starts."
                        )
                    args_delta = normalized_event.delta.args_delta
                    if args_delta is not None:
                        yield event(
                            {
                                "type": "content_block_delta",
                                "index": normalized_event.index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args_delta,
                                },
                            }
                        )
                else:
                    raise ValueError(f"Unsupported output delta: {type(normalized_event.delta)}")
            elif isinstance(normalized_event, PartEndEvent):
                if not isinstance(normalized_event.part, (Thinking, TextOutput, ToolCall)):
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                if isinstance(normalized_event.part, Thinking):
                    part = normalized_event.part
                    if normalized_event.index not in thinking_parts:
                        raise ValueError("Received a thinking end before its block started.")
                    if self._capabilities.thinking == "raw":
                        if part.summary or part.signature:
                            raise ValueError(
                                "A raw Anthropic thinking block cannot contain a summary "
                                "or signature."
                            )
                    else:
                        if part.content:
                            raise ValueError(
                                "An Anthropic summary block cannot contain raw reasoning."
                            )
                        if not part.signature:
                            raise ValueError(
                                "An Anthropic reasoning summary requires a signature."
                            )
                        if thinking_signatures.get(normalized_event.index) != part.signature:
                            raise ValueError(
                                "An Anthropic reasoning signature must be streamed before "
                                "the block ends."
                            )
                yield event({"type": "content_block_stop", "index": normalized_event.index})
            elif isinstance(normalized_event, ResponseEndEvent):
                stop_reason = _STOP_REASONS.get(normalized_event.termination_reason, "end_turn")
                usage = normalized_event.usage
                raw_usage: dict[str, Any] = {"output_tokens": 0}
                if usage is not None:
                    cache_creation = usage.input_tokens_details.cache_write_tokens
                    cache_read = usage.input_tokens_details.cached_tokens
                    raw_usage = {
                        "input_tokens": usage.input_tokens - cache_creation - cache_read,
                        "cache_creation_input_tokens": cache_creation,
                        "cache_read_input_tokens": cache_read,
                        "output_tokens": usage.output_tokens,
                        "output_tokens_details": {
                            "thinking_tokens": usage.output_tokens_details.reasoning_tokens
                        },
                    }
                yield event(
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": normalized_event.stop_sequence,
                        },
                        "usage": raw_usage,
                    }
                )
                yield event({"type": "message_stop"})

    def _handle_error(self, error: Exception) -> NoReturn:
        cause = error.__cause__
        if (not isinstance(error, APIException)) or cause is None:
            raise error
        error_map = {
            401: AnthropicAuthenticationError,
            404: AnthropicNotFoundError,
        }
        # Same-protocol passthrough: the upstream exception is already
        # downstream-native, so preserve its SDK subclass. Reconstruct a
        # fresh instance rather than re-raise the cause: chaining it
        # `from error` would make the two exceptions each other's cause.
        if isinstance(cause, AnthropicAPIStatusError):
            raise type(cause)(
                cause.message,
                response=cause.response,
                body=cause.body,
            ) from error
        # TODO: Handle bare `APIError` (e.g. mid-stream error events): it
        # has no `APIStatusError` cause and passes through unchanged until
        # an IR recognition rule exists.
        else:
            error_type = error_map.get(
                getattr(cause, "status_code", None), AnthropicAPIStatusError
            )
            raise error_type(
                error.message,
                response=error.response,
                body=error.body,
            ) from error
