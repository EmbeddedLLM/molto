"""Translate Anthropic Messages at Molto's downstream protocol boundary.

See ``docs/openai-responses-anthropic-messages-parity.md`` for supported
semantics, compatibility mappings, and matched-cassette policy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from molto.adapters.base import BaseAdapter
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesStreamEvent,
    AnthropicMessagesStreamEventAdapter,
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


def _reasoning_effort_from_budget(budget_tokens: int) -> ReasoningEffort:
    return next(
        (effort for budget, effort in _EXTENDED_THINKING_EFFORTS if budget_tokens <= budget),
        "max",
    )


class AnthropicMessagesAdapter(
    BaseAdapter[AnthropicMessagesCreateRequest, AnthropicMessagesStreamEvent]
):
    """Translate the downstream Anthropic Messages protocol to and from normalized IR."""

    @staticmethod
    def _normalize_user_block(block: dict[str, Any]) -> TextInput | ImageUrlInput:
        block_type = block.get("type")
        if block_type == "text":
            return TextInput(content=block["text"])
        if block_type == "image":
            source = block.get("source") or {}
            source_type = source.get("type")
            if source_type == "base64":
                return ImageUrlInput(url=f"data:{source['media_type']};base64,{source['data']}")
            if source_type == "url":
                return ImageUrlInput(url=source["url"])
            raise ValueError(f"Unsupported Anthropic image source: {source_type}")
        raise ValueError(f"Unsupported Anthropic user content block: {block_type}")

    def _normalize_request(self, request: AnthropicMessagesCreateRequest) -> NormalizedRequest:
        normalized_input: list[
            SystemPrompt | UserInput | Thinking | TextOutput | ToolCall | ToolReturn
        ] = []
        if isinstance(request.system, str):
            normalized_input.append(SystemPrompt(content=request.system))
        elif request.system is not None:
            for block in request.system:
                normalized_input.append(SystemPrompt(content=block["text"]))

        for message in request.messages:
            if isinstance(message.content, str):
                content_blocks: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
            else:
                content_blocks = [dict(block) for block in message.content]

            if message.role == "system":
                unsupported = [
                    block.get("type") for block in content_blocks if block.get("type") != "text"
                ]
                if unsupported:
                    raise ValueError(
                        f"Unsupported Anthropic {message.role} content blocks: {unsupported}"
                    )
                normalized_input.extend(
                    SystemPrompt(content=block["text"]) for block in content_blocks
                )
                continue

            if message.role == "user":
                user_parts: list[TextInput | ImageUrlInput] = []
                for block in content_blocks:
                    block_type = block.get("type")
                    if block_type in {"text", "image"}:
                        user_parts.append(self._normalize_user_block(block))
                        continue
                    if block_type != "tool_result":
                        raise ValueError(
                            f"Unsupported Anthropic {message.role} content block: {block_type}"
                        )
                    if user_parts:
                        raise ValueError(
                            "Anthropic tool results must precede user content in a message."
                        )

                    tool_call_id = block["tool_use_id"]
                    content = block.get("content", "")
                    if isinstance(content, str):
                        normalized_content = [TextInput(content=content)]
                    else:
                        content = list(content)
                        unsupported = [
                            part.get("type")
                            for part in content
                            if not isinstance(part, dict) or part.get("type") != "text"
                        ]
                        if unsupported:
                            raise ValueError(
                                f"Unsupported Anthropic tool result content blocks: {unsupported}"
                            )
                        normalized_content = [TextInput(content=part["text"]) for part in content]
                    normalized_input.append(
                        ToolReturn(
                            content=normalized_content,
                            tool_call_id=tool_call_id,
                            outcome="failed" if block.get("is_error") else "success",
                        )
                    )
                if user_parts:
                    normalized_input.append(UserInput(content=user_parts))
                continue

            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "thinking":
                    normalized_input.append(
                        Thinking(
                            content=(
                                block["thinking"] if self._capabilities.thinking == "raw" else ""
                            ),
                            summary=(
                                block["thinking"]
                                if self._capabilities.thinking == "summary"
                                else ""
                            ),
                            signature=block.get("signature"),
                        )
                    )
                elif block_type == "text":
                    normalized_input.append(TextOutput(content=block["text"]))
                elif block_type == "tool_use":
                    caller = block.get("caller")
                    caller_type = (
                        caller.get("type")
                        if isinstance(caller, dict)
                        else getattr(caller, "type", None)
                    )
                    if caller is not None and caller_type != "direct":
                        raise ValueError(
                            "Only directly invoked Anthropic tool uses can be translated."
                        )
                    tool_call_id = block["id"]
                    tool_name = block["name"]
                    normalized_input.append(
                        ToolCall(
                            tool_name=tool_name,
                            args=json_dumps(block["input"]),
                            tool_call_id=tool_call_id,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported Anthropic {message.role} content block: {block_type}"
                    )

        data: dict[str, Any] = {
            "input": normalized_input,
            "max_output_tokens": request.max_tokens,
            "model": request.model,
            "stream": request.stream,
            "stream_options": StreamOptions(include_obfuscation=False),
        }
        if request.tools is not None:
            normalized_tools: list[FunctionTool] = []
            for tool in request.tools:
                if tool.get("type") not in {None, "custom"}:
                    raise ValueError(f"Unsupported Anthropic tool type: {tool.get('type')!r}.")
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
                    raise ValueError(
                        f"Unsupported Anthropic custom tool fields: {sorted(unsupported_fields)}."
                    )
                normalized_tools.append(
                    FunctionTool(
                        name=tool["name"],
                        description=tool.get("description"),
                        parameters=tool["input_schema"],
                        strict=tool.get("strict", False),
                    )
                )
            data["tools"] = normalized_tools
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
                raise ValueError(f"Unsupported Anthropic tool choice: {choice_type!r}.")
            if "disable_parallel_tool_use" in request.tool_choice:
                data["parallel_tool_calls"] = not request.tool_choice["disable_parallel_tool_use"]
        if request.thinking is not None:
            thinking_type = request.thinking.get("type")
            if thinking_type == "disabled":
                data["reasoning"] = Reasoning(effort="none")
            elif request.thinking.get("display", "summarized") != "summarized":
                raise ValueError(
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
                raise ValueError(f"Unsupported Anthropic thinking type: {thinking_type}")
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        return NormalizedRequest.model_validate(data)

    async def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        del request

        def event(data: dict[str, Any]) -> AnthropicMessagesStreamEvent:
            return AnthropicMessagesStreamEventAdapter.validate_python(data)

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
                if isinstance(normalized_event.part, Thinking):
                    content_block = {
                        "type": "thinking",
                        "thinking": "",
                        "signature": normalized_event.part.signature or "",
                    }
                elif isinstance(normalized_event.part, TextOutput):
                    content_block = {"type": "text", "text": ""}
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
                    reasoning_delta = (
                        normalized_event.delta.content_delta
                        if normalized_event.delta.content_delta is not None
                        else normalized_event.delta.summary_delta
                    )
                    if reasoning_delta is not None:
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
                    if normalized_event.delta.signature_delta is not None:
                        yield event(
                            {
                                "type": "content_block_delta",
                                "index": normalized_event.index,
                                "delta": {
                                    "type": "signature_delta",
                                    "signature": normalized_event.delta.signature_delta,
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
