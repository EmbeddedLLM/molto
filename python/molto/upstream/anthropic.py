from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, NoReturn

from anthropic import APIStatusError, AsyncAnthropic, AsyncStream
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from pydantic import ValidationError
from pydantic_ai.profiles.anthropic import (
    ANTHROPIC_THINKING_BUDGET_MAP,
    anthropic_model_profile,
    resolve_anthropic_effort,
)

from molto.providers import ProviderCapabilities
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesStreamEvent,
)
from molto.types.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    ResourceNotFoundError,
    UnsupportedInput,
)
from molto.types.normalized import (
    FunctionToolChoice,
    ImageUrlInput,
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
from molto.upstream.base import BaseClient
from molto.utils.io import json_dumps, json_loads

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


@dataclass
class _PartState:
    """Accumulated wire state for one streamed Anthropic content block."""

    part: Thinking | TextOutput | ToolCall
    content: str = ""
    signature: str = ""


def _cache_control(request: NormalizedRequest) -> dict[str, Any] | None:
    """Project normalized cache intent onto Messages' top-level automatic caching."""
    cache = request.prompt_cache
    if cache is None:
        return None
    if cache.retention is not None:
        raise UnsupportedInput("Anthropic Messages cannot represent prompt cache retention.")
    # Messages derives cache identity from content. Dropping an affinity key alone must
    # not enable paid cache writes; explicit mode likewise adds no automatic breakpoint.
    if cache.mode == "explicit" or (cache.mode is None and cache.ttl is None):
        return None
    return _ephemeral_cache_control(cache.ttl)


def _ephemeral_cache_control(ttl: str | None) -> dict[str, Any]:
    control: dict[str, Any] = {"type": "ephemeral"}
    if ttl is not None:
        # TTL is a minimum lifetime; Anthropic's 1h option satisfies OpenAI's 30m.
        control["ttl"] = "1h" if ttl == "30m" else ttl
    return control


def _content_block(
    part: SystemPrompt | TextInput | ImageUrlInput | TextOutput,
    request: NormalizedRequest,
) -> dict[str, Any]:
    block = (
        _image_block(part)
        if isinstance(part, ImageUrlInput)
        else {"type": "text", "text": part.content}
    )
    if part.cache_breakpoint is not None:
        ttl = part.cache_breakpoint.ttl
        if ttl is None and request.prompt_cache is not None:
            ttl = request.prompt_cache.ttl
        block["cache_control"] = _ephemeral_cache_control(ttl)
    return block


def _validate_cache_controls(body: dict[str, Any]) -> None:
    """Check Messages' slot and lifetime constraints after choosing native boundaries."""
    blocks = list(body["system"]) if isinstance(body.get("system"), list) else []
    blocks.extend(block for message in body["messages"] for block in message["content"])
    controls = []
    for block in blocks:
        if block["type"] == "tool_result":
            controls.extend(
                part["cache_control"] for part in block["content"] if "cache_control" in part
            )
        if "cache_control" in block:
            controls.append(block["cache_control"])
    automatic = body.get("cache_control")
    if automatic is not None:
        last = next((block for block in reversed(blocks) if block["type"] != "thinking"), {})
        existing = last.get("cache_control")
        if existing is None:
            controls.append(automatic)
        elif existing.get("ttl", "5m") != automatic.get("ttl", "5m"):
            raise UnsupportedInput(
                "Anthropic automatic and final explicit cache lifetimes must match."
            )
    if len(controls) > 4:
        raise UnsupportedInput(
            "Anthropic Messages supports at most four cache breakpoints, including automatic caching."
        )
    seen_short_ttl = False
    for control in controls:
        if control.get("ttl", "5m") == "5m":
            seen_short_ttl = True
        elif seen_short_ttl:
            raise UnsupportedInput("Anthropic 1h cache breakpoints must precede 5m breakpoints.")


def _tools(request: NormalizedRequest) -> list[dict[str, Any]] | None:
    if request.tools is None:
        return None
    tools: list[dict[str, Any]] = []
    for tool in request.tools:
        native_tool: dict[str, Any] = {
            "name": tool.name,
            "input_schema": tool.parameters,
        }
        if tool.description is not None:
            native_tool["description"] = tool.description
        if tool.strict:
            native_tool["strict"] = True
        tools.append(native_tool)
    return tools


def _tool_choice(request: NormalizedRequest) -> dict[str, Any] | None:
    has_tools = bool(request.tools)
    if not has_tools and (
        request.tool_choice == "required" or isinstance(request.tool_choice, FunctionToolChoice)
    ):
        raise UnsupportedInput("A required Anthropic tool choice needs at least one tool.")
    if request.tool_choice is not None and has_tools:
        if request.tool_choice == "auto":
            tool_choice: dict[str, Any] = {"type": "auto"}
        elif request.tool_choice == "none":
            tool_choice = {"type": "none"}
        elif request.tool_choice == "required":
            tool_choice = {"type": "any"}
        elif isinstance(request.tool_choice, FunctionToolChoice):
            tool_choice = {"type": "tool", "name": request.tool_choice.name}
        else:
            raise UnsupportedInput(f"Unsupported tool choice: {request.tool_choice!r}.")
    elif request.parallel_tool_calls is not None and has_tools:
        # Parallel-call control rides on an explicit auto choice.
        tool_choice = {"type": "auto"}
    else:
        return None
    # Messages cannot encode parallel-call control when tool use is disabled.
    if request.parallel_tool_calls is not None and tool_choice["type"] != "none":
        tool_choice["disable_parallel_tool_use"] = not request.parallel_tool_calls
    return tool_choice


def _thinking_configuration(request: NormalizedRequest, *, has_thinking: bool) -> dict[str, Any]:
    """Build the thinking-related request body fields for the target model."""
    effort = request.reasoning.effort if request.reasoning is not None else None
    if effort == "none":
        return {"thinking": {"type": "disabled"}}
    if not has_thinking and request.reasoning is None:
        return {}
    adaptive, supports_xhigh = _model_thinking_capabilities(request.model or "")
    if adaptive:
        return {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": _anthropic_effort(effort, supports_xhigh=supports_xhigh)},
        }
    budget_tokens = _thinking_budget(effort)
    max_tokens = request.max_output_tokens
    if max_tokens is None or budget_tokens >= max_tokens:
        raise UnsupportedInput(
            "Anthropic extended-thinking budget "
            f"{budget_tokens} must be less than max_output_tokens {max_tokens}."
        )
    return {
        "thinking": {
            "type": "enabled",
            "budget_tokens": budget_tokens,
            "display": "summarized",
        }
    }


def _image_block(part: ImageUrlInput) -> dict[str, Any]:
    # Anthropic has no image-fidelity control; an explicit non-default detail
    # request would be silently lost, so it is rejected instead.
    if part.detail not in (None, "auto"):
        raise UnsupportedInput(
            f"Anthropic Messages cannot represent image detail {part.detail!r}."
        )
    if not part.url.startswith("data:"):
        return {"type": "image", "source": {"type": "url", "url": part.url}}
    metadata, separator, data = part.url.partition(",")
    media_type = metadata.removeprefix("data:")
    if not separator or not media_type.endswith(";base64"):
        raise UnsupportedInput("Anthropic image blocks require a base64 data URL or a remote URL.")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type.removesuffix(";base64"),
            "data": data,
        },
    }


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
        try:
            sdk_client = self._make_anthropic_client()
            stream = await sdk_client.messages.create(**request.model_dump())
            if not isinstance(stream, AsyncStream):
                raise TypeError("Messages request did not return a stream.")
            async with stream:
                async for event in stream:
                    yield event
        except Exception as error:
            self._handle_error(error)

    @staticmethod
    def _request_parts(
        request: NormalizedRequest,
        *,
        capabilities: ProviderCapabilities | None = None,
    ) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]], bool]:
        system_parts: list[SystemPrompt] = []
        messages: list[dict[str, Any]] = []
        assistant_blocks: list[dict[str, Any]] = []
        tool_result_blocks: list[dict[str, Any]] = []
        has_thinking = False

        def flush_assistant() -> None:
            if assistant_blocks:
                messages.append({"role": "assistant", "content": assistant_blocks.copy()})
                assistant_blocks.clear()

        def flush_tool_results() -> None:
            if tool_result_blocks:
                messages.append({"role": "user", "content": tool_result_blocks.copy()})
                tool_result_blocks.clear()

        for item in request.input or []:
            if isinstance(item, SystemPrompt):
                flush_assistant()
                flush_tool_results()
                system_parts.append(item)
            elif isinstance(item, UserInput):
                flush_assistant()
                flush_tool_results()
                content = [_content_block(part, request) for part in item.content]
                messages.append({"role": "user", "content": content})
            elif isinstance(item, Thinking):
                flush_tool_results()
                has_thinking = True
                representation = (
                    "raw"
                    if item.content
                    else "summary"
                    if item.summary or item.signature
                    else None
                )
                if representation == "summary" and not item.signature:
                    raise UnsupportedInput("An Anthropic reasoning summary requires a signature.")
                if (
                    representation is not None
                    and capabilities is not None
                    and representation != capabilities.thinking
                ):
                    raise UnsupportedInput(
                        "Anthropic thinking history does not match the provider's reasoning "
                        "representation."
                    )
                block = {
                    "type": "thinking",
                    "thinking": item.content if representation == "raw" else item.summary,
                }
                if representation == "summary":
                    block["signature"] = item.signature
                assistant_blocks.append(block)
            elif isinstance(item, TextOutput):
                flush_tool_results()
                assistant_blocks.append(_content_block(item, request))
            elif isinstance(item, ToolCall):
                flush_tool_results()
                tool_input = {} if item.args is None else json_loads(item.args)
                if not isinstance(tool_input, dict):
                    raise UnsupportedInput("Anthropic tool arguments must contain a JSON object.")
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": item.tool_call_id,
                        "name": item.tool_name,
                        "input": tool_input,
                    }
                )
            elif isinstance(item, ToolReturn):
                flush_assistant()
                if item.outcome not in {"success", "failed"}:
                    raise UnsupportedInput(
                        "Anthropic Messages cannot preserve denied or interrupted tool outcomes."
                    )
                block = {
                    "type": "tool_result",
                    "tool_use_id": item.tool_call_id,
                    "content": [_content_block(part, request) for part in item.content],
                }
                if item.outcome == "failed":
                    block["is_error"] = True
                tool_result_blocks.append(block)
            else:
                raise UnsupportedInput(f"Unsupported Anthropic input item: {type(item)}")
        flush_assistant()
        flush_tool_results()
        system = (
            [_content_block(part, request) for part in system_parts]
            if any(part.cache_breakpoint is not None for part in system_parts)
            else "\n\n".join(part.content for part in system_parts) or None
        )
        return system, messages, has_thinking

    @classmethod
    def _request_body(
        cls,
        request: NormalizedRequest,
        *,
        capabilities: ProviderCapabilities | None = None,
    ) -> dict[str, Any]:
        cls._validate_request(request)
        system, messages, has_thinking = cls._request_parts(request, capabilities=capabilities)
        body: dict[str, Any] = {
            "max_tokens": request.max_output_tokens,
            "messages": messages,
            "model": request.model,
            "stream": request.stream,
        }
        if system is not None:
            body["system"] = system
        tools = _tools(request)
        if tools is not None:
            body["tools"] = tools
        tool_choice = _tool_choice(request)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        cache_control = _cache_control(request)
        if cache_control is not None:
            body["cache_control"] = cache_control
        _validate_cache_controls(body)
        body.update(_thinking_configuration(request, has_thinking=has_thinking))
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        try:
            return AnthropicMessagesCreateRequest.model_validate(body).model_dump()
        except ValidationError as error:
            raise UnsupportedInput(str(error)) from error

    @staticmethod
    def _validate_request(request: NormalizedRequest) -> None:
        if next(request.iter_cache_breakpoints(), None) is not None:
            seen_message = False
            for item in request.input or []:
                if isinstance(item, SystemPrompt):
                    if seen_message:
                        raise UnsupportedInput(
                            "Anthropic Messages cannot preserve cache prefixes when hoisting "
                            "mid-conversation instructions."
                        )
                else:
                    seen_message = True
        if request.max_tool_calls is not None:
            raise UnsupportedInput("Anthropic Messages cannot represent max_tool_calls.")
        if (
            request.stream_options is not None
            and request.stream_options.include_obfuscation is True
        ):
            raise UnsupportedInput("Anthropic Messages cannot enable stream obfuscation.")
        if request.reasoning is not None and request.reasoning.summary not in (
            None,
            "auto",
            "detailed",
        ):
            raise UnsupportedInput(
                "Anthropic Messages cannot represent concise reasoning summaries."
            )
        for item in request.input or []:
            if isinstance(item, ToolReturn):
                if item.tool_name is not None:
                    raise UnsupportedInput(
                        "Anthropic Messages cannot represent tool return names."
                    )

    async def _events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        metadata: ResponseStartEvent | None = None
        input_tokens = 0
        cache_creation = 0
        cache_read = 0
        usage: Usage | None = None
        termination_reason: TerminationReason | None = None
        stop_sequence: str | None = None
        parts: dict[int, _PartState] = {}
        # Only a thinking block that starts completely empty (no text or
        # signature) is deferred: if no deltas ever arrive, its part lifecycle is
        # omitted from the IR entirely. A block that already carries text or a
        # signature starts immediately, so a stalled or failed stream cannot lose
        # the observed content.
        pending_thinking: set[int] = set()
        # Source block indices map to dense normalized indices so an omitted
        # thinking block cannot leave a gap for downstream projections.
        part_indices: dict[int, int] = {}

        def pop_pending_thinking(index: int) -> Thinking | None:
            if index not in pending_thinking:
                return None
            pending_thinking.discard(index)
            part = parts[index].part
            return part if isinstance(part, Thinking) else None

        def start_part(
            source_index: int, part: Thinking | TextOutput | ToolCall
        ) -> PartStartEvent:
            # Assign the dense index only when the start is actually emitted, so
            # an omitted thinking block never consumes a normalized index.
            part_indices[source_index] = len(part_indices)
            return PartStartEvent(index=part_indices[source_index], part=part)

        native_request = AnthropicMessagesCreateRequest.model_validate(
            self._request_body(request, capabilities=self._capabilities)
        )
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
                    part: Thinking | TextOutput | ToolCall = self._classify_thinking(
                        block.thinking, signature=block.signature
                    )
                elif block.type == "text":
                    part = TextOutput(content=block.text)
                elif block.type == "tool_use":
                    if block.caller is not None and block.caller.type != "direct":
                        raise ValueError(
                            "Only directly invoked Anthropic tool uses can be translated."
                        )
                    part = ToolCall(
                        tool_name=block.name,
                        args="" if not block.input else json_dumps(block.input),
                        tool_call_id=block.id,
                    )
                else:
                    raise ValueError(f"Unsupported Anthropic output block: {block.type}")
                if (
                    isinstance(part, Thinking)
                    and not part.content
                    and not part.summary
                    and not part.signature
                ):
                    parts[event.index] = _PartState(part)
                    pending_thinking.add(event.index)
                elif isinstance(part, Thinking):
                    parts[event.index] = _PartState(part, block.thinking, part.signature or "")
                    yield start_part(event.index, part)
                else:
                    parts[event.index] = _PartState(
                        part, part.content if isinstance(part, TextOutput) else part.args
                    )
                    yield start_part(event.index, part)
            elif isinstance(event, RawContentBlockDeltaEvent):
                delta = event.delta
                state = parts.get(event.index)
                part = state.part if state is not None else None
                if delta.type == "thinking_delta" and isinstance(part, Thinking):
                    # An empty delta must not materialize a deferred block.
                    if not delta.thinking and event.index in pending_thinking:
                        continue
                    if (pending := pop_pending_thinking(event.index)) is not None:
                        yield start_part(event.index, pending)
                    state.content += delta.thinking
                    yield PartDeltaEvent(
                        index=part_indices[event.index],
                        delta=self._classify_thinking_delta(delta.thinking),
                    )
                elif delta.type == "signature_delta" and isinstance(part, Thinking):
                    if self._capabilities.thinking == "raw":
                        continue
                    # An empty signature carries no state and must not
                    # materialize a deferred block.
                    if not delta.signature and event.index in pending_thinking:
                        continue
                    if (pending := pop_pending_thinking(event.index)) is not None:
                        yield start_part(event.index, pending)
                    # Anthropic defines signature_delta as the complete replacement
                    # signature for the block, not an append-only fragment.
                    state.signature = delta.signature
                    yield PartDeltaEvent(
                        index=part_indices[event.index],
                        delta=ThinkingDelta(signature_delta=delta.signature),
                    )
                elif delta.type == "text_delta" and isinstance(part, TextOutput):
                    state.content += delta.text
                    yield PartDeltaEvent(
                        index=part_indices[event.index],
                        delta=TextDelta(content_delta=delta.text),
                    )
                elif delta.type == "input_json_delta" and isinstance(part, ToolCall):
                    state.content += delta.partial_json
                    yield PartDeltaEvent(
                        index=part_indices[event.index],
                        delta=ToolCallDelta(args_delta=delta.partial_json),
                    )
                else:
                    raise ValueError(f"Unsupported Anthropic output delta: {delta.type}")
            elif isinstance(event, RawContentBlockStopEvent):
                state = parts.get(event.index)
                if state is None:
                    raise ValueError("Received an Anthropic block stop before its start.")
                if pop_pending_thinking(event.index) is not None:
                    # A deferred block that received no deltas stayed empty.
                    continue
                part = state.part
                if isinstance(part, Thinking):
                    if self._capabilities.thinking == "summary" and not state.signature:
                        raise ValueError("An Anthropic reasoning summary requires a signature.")
                    completed_part: Thinking | TextOutput | ToolCall = self._classify_thinking(
                        state.content, signature=state.signature
                    )
                elif isinstance(part, ToolCall):
                    completed_part = ToolCall(
                        tool_name=part.tool_name,
                        args=state.content,
                        tool_call_id=part.tool_call_id,
                    )
                else:
                    completed_part = TextOutput(content=state.content)
                yield PartEndEvent(index=part_indices[event.index], part=completed_part)
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

    def _handle_error(self, error: Exception) -> NoReturn:
        if isinstance(error, APIStatusError):
            if error.status_code == 401:
                raise AuthenticationError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
            if error.status_code == 404 and "model:" in error.message:
                raise ModelNotFoundError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
            if error.status_code == 404:
                raise ResourceNotFoundError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
        raise error
