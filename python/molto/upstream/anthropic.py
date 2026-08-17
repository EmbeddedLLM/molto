from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from pydantic_ai.profiles.anthropic import (
    ANTHROPIC_THINKING_BUDGET_MAP,
    anthropic_model_profile,
    resolve_anthropic_effort,
)

from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesStreamEvent,
)
from molto.types.normalized import (
    InputTokenDetails,
    ModelResponseStreamEvent,
    NormalizedRequest,
    OutputTokenDetails,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ReasoningEffort,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TerminationReason,
    TextDelta,
    TextOutput,
    Thinking,
    ThinkingDelta,
    Usage,
    UserInput,
)
from molto.upstream.base import BaseClient

_TERMINATION_REASONS: dict[str, TerminationReason] = {
    "end_turn": "stop",
    "max_tokens": "max_output_tokens",
    "model_context_window_exceeded": "model_context_window_exceeded",
    "pause_turn": "pause",
    "refusal": "refusal",
    "stop_sequence": "stop_sequence",
    "tool_use": "tool_call",
}


def _is_current_or_future_adaptive_claude(model: str) -> bool:
    if model.startswith("claude-mythos-preview"):
        return True
    for family in ("fable", "haiku", "mythos", "opus", "sonnet"):
        prefix = f"claude-{family}-"
        if model.startswith(prefix):
            major = model.removeprefix(prefix).partition("-")[0]
            return major.isdigit() and int(major) >= 5
    return False


def _model_thinking_capabilities(model: str) -> tuple[bool, bool]:
    # Anthropic-compatible endpoints in the matched fixtures use non-Claude model
    # IDs and expose adaptive thinking. Known Claude IDs use the installed model
    # profile, supplemented for newer families not yet present in the pinned table.
    if not model.startswith("claude-"):
        return True, True
    profile = anthropic_model_profile(model)
    current_or_future = _is_current_or_future_adaptive_claude(model)
    # Pydantic AI has exposed model profiles as both dataclasses and TypedDicts.
    if isinstance(profile, dict):
        supports_adaptive = profile.get("anthropic_supports_adaptive_thinking", False)
        supports_xhigh = profile.get("anthropic_supports_xhigh_effort", False)
    else:
        supports_adaptive = profile.anthropic_supports_adaptive_thinking
        supports_xhigh = profile.anthropic_supports_xhigh_effort
    return (
        supports_adaptive or current_or_future,
        supports_xhigh or current_or_future,
    )


def _anthropic_effort(effort: ReasoningEffort, *, supports_xhigh: bool) -> str:
    if effort == "max":
        return "max"
    return resolve_anthropic_effort(effort or "high", supports_xhigh=supports_xhigh)


def _thinking_budget(effort: ReasoningEffort) -> int:
    if effort == "max":
        return ANTHROPIC_THINKING_BUDGET_MAP["xhigh"]
    return ANTHROPIC_THINKING_BUDGET_MAP[effort or True]


class AnthropicMessagesClient(
    BaseClient[AnthropicMessagesCreateRequest, AnthropicMessagesStreamEvent]
):
    """Anthropic Messages transport for native and normalized streaming paths."""

    DEFAULT_BASE_URL = "https://api.anthropic.com"

    def _make_anthropic_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(
            api_key=self._api_key,
            base_url=self._base_url or self.DEFAULT_BASE_URL,
            http_client=self._http_client,
        )

    async def native_events(
        self, request: AnthropicMessagesCreateRequest
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        sdk_client = self._make_anthropic_client()
        stream = await sdk_client.messages.create(**request.model_dump())
        if not isinstance(stream, AsyncStream):
            raise TypeError("Messages request did not return a stream.")
        async with stream:
            async for event in stream:
                yield event

    @staticmethod
    def _request_parts(
        request: NormalizedRequest,
    ) -> tuple[str | None, list[dict[str, Any]], bool]:
        if isinstance(request.input, str):
            return (
                None,
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": request.input}],
                    }
                ],
                False,
            )

        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        assistant_blocks: list[dict[str, Any]] = []
        has_thinking = False

        def flush_assistant() -> None:
            if assistant_blocks:
                messages.append({"role": "assistant", "content": assistant_blocks.copy()})
                assistant_blocks.clear()

        for item in request.input or []:
            if isinstance(item, SystemPrompt):
                flush_assistant()
                system_parts.append(item.content)
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
                has_thinking = True
                block = {"type": "thinking", "thinking": item.content or item.summary}
                if item.signature is not None:
                    block["signature"] = item.signature
                assistant_blocks.append(block)
            elif isinstance(item, TextOutput):
                assistant_blocks.append({"type": "text", "text": item.content})
            else:
                raise ValueError(f"Unsupported Anthropic input item: {type(item)}")
        flush_assistant()
        return "\n\n".join(system_parts) or None, messages, has_thinking

    @classmethod
    def _request_body(cls, request: NormalizedRequest) -> dict[str, Any]:
        system, messages, has_thinking = cls._request_parts(request)
        body: dict[str, Any] = {
            "max_tokens": request.max_output_tokens,
            "messages": messages,
            "model": request.model,
            "stream": request.stream,
        }
        if system is not None:
            body["system"] = system
        effort = request.reasoning.effort if request.reasoning is not None else None
        if effort == "none":
            body["thinking"] = {"type": "disabled"}
        elif has_thinking or request.reasoning is not None:
            adaptive, supports_xhigh = _model_thinking_capabilities(request.model or "")
            if adaptive:
                body["thinking"] = {"type": "adaptive", "display": "summarized"}
                body["output_config"] = {
                    "effort": _anthropic_effort(effort, supports_xhigh=supports_xhigh)
                }
            else:
                budget_tokens = _thinking_budget(effort)
                max_tokens = request.max_output_tokens
                if max_tokens is None or budget_tokens >= max_tokens:
                    raise ValueError(
                        "Anthropic extended-thinking budget "
                        f"{budget_tokens} must be less than max_output_tokens {max_tokens}."
                    )
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                    "display": "summarized",
                }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        return AnthropicMessagesCreateRequest.model_validate(body).model_dump()

    async def events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        metadata: ResponseStartEvent | None = None
        input_tokens = 0
        cache_creation = 0
        cache_read = 0
        usage: Usage | None = None
        termination_reason: TerminationReason | None = None
        stop_sequence: str | None = None
        parts: dict[int, tuple[Thinking | TextOutput, str, str]] = {}

        native_request = AnthropicMessagesCreateRequest.model_validate(self._request_body(request))
        async for event in self.native_events(native_request):
            if isinstance(event, RawMessageStartEvent):
                message = event.message
                metadata = ResponseStartEvent(
                    id=message.id,
                    model=message.model,
                    service_tier=message.usage.service_tier,
                )
                input_tokens = message.usage.input_tokens
                cache_creation = message.usage.cache_creation_input_tokens or 0
                cache_read = message.usage.cache_read_input_tokens or 0
                yield metadata
            elif isinstance(event, RawContentBlockStartEvent):
                block = event.content_block
                if block.type == "thinking":
                    part = Thinking(
                        content=(block.thinking if self._capabilities.thinking == "raw" else ""),
                        summary=(
                            block.thinking if self._capabilities.thinking == "summary" else ""
                        ),
                        signature=block.signature,
                    )
                elif block.type == "text":
                    part = TextOutput(content=block.text)
                else:
                    raise ValueError(f"Unsupported Anthropic output block: {block.type}")
                signature = (part.signature or "") if isinstance(part, Thinking) else ""
                content = (
                    part.content
                    if not isinstance(part, Thinking) or self._capabilities.thinking == "raw"
                    else part.summary
                )
                parts[event.index] = (part, content, signature)
                yield PartStartEvent(index=event.index, part=part)
            elif isinstance(event, RawContentBlockDeltaEvent):
                delta = event.delta
                part_state = parts.get(event.index)
                part = part_state[0] if part_state is not None else None
                if delta.type == "thinking_delta" and isinstance(part, Thinking):
                    parts[event.index] = (
                        part,
                        part_state[1] + delta.thinking,
                        part_state[2],
                    )
                    yield PartDeltaEvent(
                        index=event.index,
                        delta=ThinkingDelta(
                            content_delta=(
                                delta.thinking if self._capabilities.thinking == "raw" else None
                            ),
                            summary_delta=(
                                delta.thinking
                                if self._capabilities.thinking == "summary"
                                else None
                            ),
                        ),
                    )
                elif delta.type == "signature_delta" and isinstance(part, Thinking):
                    parts[event.index] = (
                        part,
                        part_state[1],
                        part_state[2] + delta.signature,
                    )
                    yield PartDeltaEvent(
                        index=event.index,
                        delta=ThinkingDelta(signature_delta=delta.signature),
                    )
                elif delta.type == "text_delta" and isinstance(part, TextOutput):
                    parts[event.index] = (
                        part,
                        part_state[1] + delta.text,
                        part_state[2],
                    )
                    yield PartDeltaEvent(
                        index=event.index,
                        delta=TextDelta(content_delta=delta.text),
                    )
                else:
                    raise ValueError(f"Unsupported Anthropic output delta: {delta.type}")
            elif isinstance(event, RawContentBlockStopEvent):
                part_state = parts.get(event.index)
                if part_state is None:
                    raise ValueError("Received an Anthropic block stop before its start.")
                part, content, signature = part_state
                completed_part = (
                    Thinking(
                        content=content if self._capabilities.thinking == "raw" else "",
                        summary=content if self._capabilities.thinking == "summary" else "",
                        signature=signature,
                    )
                    if isinstance(part, Thinking)
                    else TextOutput(content=content)
                )
                yield PartEndEvent(index=event.index, part=completed_part)
            elif isinstance(event, RawMessageDeltaEvent):
                if event.delta.stop_reason is not None:
                    try:
                        termination_reason = _TERMINATION_REASONS[event.delta.stop_reason]
                    except KeyError as exc:
                        raise ValueError(
                            f"Unsupported Anthropic stop reason: {event.delta.stop_reason}"
                        ) from exc
                    stop_sequence = event.delta.stop_sequence
                raw_usage = event.usage
                if raw_usage.input_tokens is not None:
                    input_tokens = raw_usage.input_tokens
                if raw_usage.cache_creation_input_tokens is not None:
                    cache_creation = raw_usage.cache_creation_input_tokens
                if raw_usage.cache_read_input_tokens is not None:
                    cache_read = raw_usage.cache_read_input_tokens
                output_details = raw_usage.output_tokens_details
                reasoning_tokens = (
                    output_details.thinking_tokens if output_details is not None else 0
                )
                usage = Usage(
                    input_tokens=input_tokens + cache_creation + cache_read,
                    input_tokens_details=InputTokenDetails(
                        cached_tokens=cache_read,
                        cache_write_tokens=cache_creation,
                    ),
                    output_tokens=raw_usage.output_tokens,
                    output_tokens_details=OutputTokenDetails(reasoning_tokens=reasoning_tokens),
                    total_tokens=input_tokens
                    + cache_creation
                    + cache_read
                    + raw_usage.output_tokens,
                )
            elif isinstance(event, RawMessageStopEvent):
                if metadata is None:
                    raise ValueError("Anthropic stream ended before its message started.")
                if termination_reason is None:
                    raise ValueError("Anthropic stream ended without a stop reason.")
                yield ResponseEndEvent(
                    id=metadata.id,
                    created_at=metadata.created_at,
                    model=metadata.model,
                    service_tier=metadata.service_tier,
                    termination_reason=termination_reason,
                    stop_sequence=stop_sequence,
                    usage=usage,
                )
