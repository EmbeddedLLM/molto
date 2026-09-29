"""Translate Anthropic Messages at Molto's downstream protocol boundary.

See `docs/protocols/anthropic-messages.md` for supported semantics and
compatibility mappings.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any, NoReturn

from anthropic import APIError as AnthropicAPIError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import NotFoundError as AnthropicNotFoundError
from httpx import Response
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
    AuthenticationError,
    UnsupportedInput,
)
from molto.types.normalized import (
    AssistantInput,
    CacheBreakpoint,
    CitationDelta,
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
    SearchCall,
    SearchLocation,
    SearchResult,
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
    WebCitation,
    WebSearchSource,
    WebSearchTool,
    WebSearchToolChoice,
)
from molto.utils.io import json_dumps
from molto.utils.web_search import (
    citation_from_messages,
    messages_text_blocks,
    search_call_from_messages,
    search_call_to_messages,
    search_result_from_messages,
    search_result_to_messages,
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


def _validate_search_source(source: dict[str, Any]) -> None:
    """Reject unsupported source fields in Messages request history."""
    unsupported_fields = {
        key for key in source if key not in WebSearchSource.model_fields and key != "type"
    }
    if unsupported_fields:
        raise UnsupportedInput(
            f"Unsupported Anthropic search-result source fields: {sorted(unsupported_fields)}."
        )


def _citations(citations: list[dict[str, Any]] | None, text: str) -> list[WebCitation]:
    result = []
    for citation in citations or []:
        citation = dict(citation)
        if citation["type"] != "web_search_result_location":
            raise UnsupportedInput(f"Unsupported Anthropic citation: {citation['type']}.")
        result.append(citation_from_messages(citation, text=text))
    return result


async def _ordered_parts(
    events: AsyncIterator[ModelResponseStreamEvent],
) -> AsyncIterator[ModelResponseStreamEvent]:
    """Stream parts in start order, yielding only the end for snapshot-based parts."""
    # Responses may stream "Alpha. Beta." before citing only "Beta.". Messages
    # needs separate uncited/cited blocks; an emitted block cannot be split later.
    # Search actions and queries can also arrive late: the final snapshot tells
    # us to emit a search, omit browsing, or reject unsupported multiple queries.
    # If part A expands into two blocks, part B must follow both, so later parts
    # wait to preserve order and receive the correct consecutive block indices.
    # Ordinary parts stream their chunks at the FIFO head; citation/search parts
    # wait for their final snapshots, without buffering the whole response.
    pending: dict[int, deque[ModelResponseStreamEvent]] = {}
    snapshots: set[int] = set()
    async for event in events:
        if not isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent)):
            if isinstance(event, ResponseEndEvent) and (snapshots or any(pending.values())):
                raise ValueError("Messages projection ended before its parts completed.")
            yield event
            continue
        if isinstance(event, PartStartEvent):
            pending[event.index] = deque()
            part = event.part
            if isinstance(part, (SearchCall, SearchResult)) or (
                isinstance(part, TextOutput) and (part.defer_citations or part.citations)
            ):
                snapshots.add(event.index)
        if isinstance(event, PartEndEvent):
            snapshots.discard(event.index)
        # Completed snapshots contain all citations and search metadata; retaining
        # their intermediate events would only duplicate upstream accumulation.
        if event.index not in snapshots:
            pending[event.index].append(event)
        while pending:
            index = next(iter(pending))
            chunks = pending[index]
            if not chunks:
                break
            current = chunks.popleft()
            if isinstance(current, PartEndEvent):
                del pending[index]
            yield current
    # Preserve partial ordinary streams when all received chunks were emitted;
    # only unfinished snapshots or blocked chunks would otherwise be lost.
    if snapshots or any(pending.values()):
        raise ValueError("Messages projection ended before its parts completed.")


class AnthropicMessagesAdapter(
    BaseAdapter[
        AnthropicMessagesCreateRequest,
        AnthropicMessagesStreamEvent,
        AnthropicMessagesResponse,
    ]
):
    """Translate the downstream Anthropic Messages protocol to and from normalized IR."""

    @staticmethod
    def _translate_tool(tool: dict[str, Any]) -> FunctionTool | WebSearchTool:
        """Translate a Messages tool declaration into its normalized schema."""
        if tool.get("type") == "web_search_20250305":
            unsupported = set(tool) - {
                "type",
                "name",
                "allowed_domains",
                "blocked_domains",
                "user_location",
                "max_uses",
                "allowed_callers",
            }
            if unsupported or tool.get("allowed_callers") not in (None, [], ["direct"]):
                raise UnsupportedInput(
                    f"Unsupported Anthropic search tool fields: {sorted(unsupported or {'allowed_callers'})}."
                )
            location = tool.get("user_location")
            return WebSearchTool(
                allowed_domains=tool.get("allowed_domains"),
                blocked_domains=tool.get("blocked_domains"),
                user_location=SearchLocation(
                    **{key: value for key, value in location.items() if key != "type"}
                )
                if location is not None
                else None,
            )
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
                    if content_block.type not in {"tool_use", "server_tool_use"}:
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
                        if content_block.type not in {"tool_use", "server_tool_use"}:
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
    ) -> list[Thinking | TextOutput | ToolCall | SearchCall | SearchResult]:
        items: list[Thinking | TextOutput | ToolCall | SearchCall | SearchResult] = []
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
                    TextOutput(
                        content=block["text"],
                        cache_breakpoint=_cache_breakpoint(block),
                        citations=_citations(block.get("citations"), block["text"]),
                    )
                )
            elif block_type == "server_tool_use" and block["name"] == "web_search":
                _reject_cache_breakpoint(block)
                caller = block.get("caller")
                if caller is not None and dict(caller).get("type") != "direct":
                    raise UnsupportedInput(
                        "Only directly invoked Anthropic hosted searches can be translated."
                    )
                items.append(search_call_from_messages(block))
            elif block_type == "web_search_tool_result":
                _reject_cache_breakpoint(block)
                content = block["content"]
                if isinstance(content, list):
                    content = [dict(source) for source in content]
                    for source in content:
                        _validate_search_source(source)
                else:
                    content = dict(content)
                items.append(search_result_from_messages({**block, "content": content}))
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
            data["tools"] = [self._translate_tool(tool) for tool in request.tools]
            searches = [
                tool for tool in request.tools if tool.get("type") == "web_search_20250305"
            ]
            if len(searches) > 1:
                raise UnsupportedInput("Only one hosted web search tool can be translated.")
            if searches and "max_uses" in searches[0]:
                data["max_tool_calls"] = searches[0]["max_uses"]
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
                data["tool_choice"] = (
                    WebSearchToolChoice()
                    if request.tool_choice["name"] == "web_search"
                    and any(isinstance(tool, WebSearchTool) for tool in data.get("tools", []))
                    else FunctionToolChoice(name=request.tool_choice["name"])
                )
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

        indices: dict[int, int] = {}
        next_index = 0
        omitted_calls: set[str] = set()
        thinking_parts: set[int] = set()
        thinking_signatures: dict[int, str] = {}

        async with aclosing(_ordered_parts(events)) as ordered_events:
            async for normalized_event in ordered_events:
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
                    index = indices[normalized_event.index] = next_index
                    next_index += 1
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
                                thinking_signatures[index] = part.signature
                        thinking_parts.add(index)
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
                            "index": index,
                            "content_block": content_block,
                        }
                    )
                    if isinstance(normalized_event.part, ToolCall) and normalized_event.part.args:
                        yield event(
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": normalized_event.part.args,
                                },
                            }
                        )
                elif isinstance(normalized_event, PartDeltaEvent):
                    index = indices[normalized_event.index]
                    if isinstance(normalized_event.delta, CitationDelta):
                        raise ValueError(
                            "Citation-bearing text must declare deferred citations before streaming."
                        )
                    if isinstance(normalized_event.delta, ThinkingDelta):
                        delta = normalized_event.delta
                        if index not in thinking_parts:
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
                                    "index": index,
                                    "delta": {
                                        "type": "thinking_delta",
                                        "thinking": reasoning_delta,
                                    },
                                }
                            )
                        if delta.signature_delta:
                            thinking_signatures[index] = delta.signature_delta
                            yield event(
                                {
                                    "type": "content_block_delta",
                                    "index": index,
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
                                "index": index,
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
                                    "index": index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args_delta,
                                    },
                                }
                            )
                    else:
                        raise ValueError(
                            f"Unsupported output delta: {type(normalized_event.delta)}"
                        )
                elif isinstance(normalized_event, PartEndEvent):
                    if normalized_event.index not in indices:
                        part = normalized_event.part
                        if isinstance(part, TextOutput):
                            blocks = messages_text_blocks(part)
                        elif isinstance(part, SearchCall):
                            if part.action != "search":
                                omitted_calls.add(part.id)
                                continue
                            blocks = [search_call_to_messages(part)]
                        elif isinstance(part, SearchResult):
                            if part.tool_call_id in omitted_calls or (
                                part.sources is None and part.error is None
                            ):
                                continue
                            blocks = [search_result_to_messages(part)]
                        else:
                            raise ValueError(f"Unsupported output part: {type(part)}")
                        for block in blocks:
                            yield event(
                                {
                                    "type": "content_block_start",
                                    "index": next_index,
                                    "content_block": block,
                                }
                            )
                            yield event({"type": "content_block_stop", "index": next_index})
                            next_index += 1
                        continue
                    index = indices[normalized_event.index]
                    if not isinstance(normalized_event.part, (Thinking, TextOutput, ToolCall)):
                        raise ValueError(f"Unsupported output part: {type(normalized_event.part)}")
                    if isinstance(normalized_event.part, Thinking):
                        part = normalized_event.part
                        if index not in thinking_parts:
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
                            if thinking_signatures.get(index) != part.signature:
                                raise ValueError(
                                    "An Anthropic reasoning signature must be streamed before "
                                    "the block ends."
                                )
                    yield event({"type": "content_block_stop", "index": index})
                elif isinstance(normalized_event, ResponseEndEvent):
                    stop_reason = _STOP_REASONS.get(
                        normalized_event.termination_reason, "end_turn"
                    )
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
                    search_requests = normalized_event.extra_fields.get("web_search_requests")
                    if search_requests is None and usage is not None:
                        search_requests = usage.extra_fields.get("web_search_requests")
                    if search_requests is not None:
                        raw_usage["server_tool_use"] = {
                            "web_search_requests": search_requests,
                            "web_fetch_requests": 0,
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
        if error.response is None:
            if isinstance(error, AuthenticationError):
                assert error.request is not None
                # SDK auth errors require HTTP metadata even for local credential failures.
                # Keep the synthetic response out of the normalized upstream evidence.
                raise AnthropicAuthenticationError(
                    error.message,
                    response=Response(401, request=error.request),
                    body=error.body,
                ) from error
            raise error
        if getattr(cause, "status_code", None) is None:
            raise AnthropicAPIError(
                error.message,
                request=error.response.request,
                body=error.body,
            ) from error
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
        else:
            error_type = error_map.get(
                getattr(cause, "status_code", None), AnthropicAPIStatusError
            )
            raise error_type(
                error.message,
                response=error.response,
                body=error.body,
            ) from error
