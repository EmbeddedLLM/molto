"""Translate Anthropic Messages at Molto's downstream protocol boundary.

The current implementation targets streaming text and raw reasoning. Several
cross-protocol differences are intentional and should remain visible when this
scope expands:

- Anthropic does not provide a response creation timestamp, while the normalized
  response lifecycle requires one. The normalized start event therefore captures
  the local wall-clock time. This is suitable for a synthesized downstream
  timestamp but cannot be compared with a provider-backed OpenAI timestamp.
- The IR records reasoning effort, not Anthropic's thinking mode or token budget.
  ``output_config.effort`` is preserved when provided; adaptive thinking defaults
  to high effort, and manual extended-thinking budgets are reduced to the nearest
  normalized effort preset. On projection, the client uses model capabilities to
  choose adaptive thinking with explicit effort or manual thinking with a preset
  budget. These are compatibility mappings, not exact semantic identities.
- Official Anthropic thinking history requires the original signature. Some
  compatible providers and the matched raw-reasoning cassettes omit it, so the
  local request schema narrowly accepts an absent signature. Such history is not
  guaranteed to work against the official Anthropic API; add official Claude
  cassette coverage with real signatures before claiming that support. The
  matched cross-protocol fixtures instead assume OpenAI serves a Messages API and
  uses the same opaque state for Responses ``encrypted_content`` and Messages
  ``signature``. That mapping is valid only within the same provider; an
  OpenAI-issued value must not be sent to Anthropic as though Anthropic signed it.
- A translated Messages stream reports zero usage in ``message_start`` when the
  upstream protocol exposes usage only at completion. Buffering to backfill it
  would violate Molto's live-streaming contract, so terminal usage is the
  authoritative cross-protocol value.
- Normalized termination reasons preserve the distinctions shared by these
  protocols. Anthropic's ``stop_sequence`` also retains its matched sequence.
  A source reason with no faithful Anthropic equivalent is coerced to
  ``end_turn`` so translated streams remain consumable; currently
  ``content_filter`` is the only such reason. This deliberately loses the source
  distinction and must remain documented as a compatibility fallback.
- Anthropic's response service-tier vocabulary has no faithful mapping from
  OpenAI tier labels. Every translated Messages stream therefore reports the
  target default ``standard``. Native Messages passthrough remains untouched.
- Cache-token mappings are semantic interpretations backed by the matched
  fixtures rather than identical provider schemas.
- Cache-creation TTL breakdowns, ``inference_geo``, ``server_tool_use``,
  ``stop_details``, and ``container`` do not have normalized equivalents and are
  not synthesized across protocols. A Messages ``ping`` is transport keepalive,
  not model output.
- The request wrapper accepts additional native fields, including service tier,
  stop sequences, tools, tool choice, metadata, cache control, container,
  inference geography, ``top_k``, and user profile ID. They are not forwarded by
  the current normalized path. ``output_config.effort`` is the supported
  exception. Only text and thinking content blocks are implemented; other block
  variants are rejected explicitly.

Matched-cassette policy follows from those constraints: keep
``include_obfuscation: false`` on OpenAI requests because Anthropic has no
obfuscation equivalent; retain recorded provider quirks such as an SSE event name
that disagrees with its payload discriminator; and remove or regenerate stale
transport-computed headers such as ``content-length`` through cassette tooling
rather than editing recorded response data by hand.
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
)

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

    def _normalize_request(self, request: AnthropicMessagesCreateRequest) -> NormalizedRequest:
        normalized_input: list[SystemPrompt | TextInput | Thinking | TextOutput] = []
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

            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text" and message.role == "system":
                    normalized_input.append(SystemPrompt(content=block["text"]))
                elif block_type == "thinking" and message.role == "assistant":
                    normalized_input.append(
                        Thinking(
                            content=block["thinking"],
                            summary="",
                            signature=block.get("signature"),
                        )
                    )
                elif block_type == "text":
                    item_type = TextOutput if message.role == "assistant" else TextInput
                    normalized_input.append(item_type(content=block["text"]))
                else:
                    raise ValueError(
                        f"Unsupported Anthropic {message.role} content block: {block_type}"
                    )

        data: dict[str, Any] = {
            "input": (
                normalized_input[0].content
                if len(normalized_input) == 1 and isinstance(normalized_input[0], TextInput)
                else normalized_input
            ),
            "max_output_tokens": request.max_tokens,
            "model": request.model,
            "stream": request.stream,
            "stream_options": StreamOptions(include_obfuscation=False),
        }
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
                else:
                    raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                yield event(
                    {
                        "type": "content_block_start",
                        "index": normalized_event.index,
                        "content_block": content_block,
                    }
                )
            elif isinstance(normalized_event, PartDeltaEvent):
                if isinstance(normalized_event.delta, ThinkingDelta):
                    if normalized_event.delta.content_delta is not None:
                        yield event(
                            {
                                "type": "content_block_delta",
                                "index": normalized_event.index,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": normalized_event.delta.content_delta,
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
                else:
                    raise ValueError(f"Unsupported output delta: {type(normalized_event.delta)}")
            elif isinstance(normalized_event, PartEndEvent):
                if not isinstance(normalized_event.part, (Thinking, TextOutput)):
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
