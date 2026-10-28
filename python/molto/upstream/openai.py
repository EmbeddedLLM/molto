"""OpenAI Responses and Chat Completions upstream transport and normalization."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from httpx import Response
from openai import APIError, APIStatusError, AsyncStream, OpenAIError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import NotFoundError as OpenAINotFoundError
from openai.types.responses import ResponseCompletedEvent, ResponseStreamEvent
from pydantic import BaseModel, ValidationError

from molto.types.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ResourceNotFoundError,
    UnsupportedInput,
)
from molto.types.normalized import (
    AssistantInput,
    CacheBreakpoint,
    CitationDelta,
    FunctionTool,
    FunctionToolChoice,
    ImageUrlInput,
    InputTokenDetails,
    ModelResponseStreamEvent,
    NormalizedRequest,
    OutputTokenDetails,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    SearchCall,
    SearchResult,
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
    WebSearchTool,
    WebSearchToolChoice,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsResponse,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import OpenAIResponsesCreateRequest, OpenAIResponsesResponse
from molto.upstream.base import BaseClient
from molto.utils.web_search import (
    citation_from_responses,
    citation_to_responses,
    search_call_from_responses,
    search_call_to_responses,
    search_result_from_responses,
    search_source_to_responses,
)

_CHAT_TERMINATION_REASONS: dict[str, TerminationReason] = {
    "content_filter": "content_filter",
    "function_call": "tool_call",
    "length": "max_output_tokens",
    "stop": "stop",
    "tool_calls": "tool_call",
}


@dataclass
class _SearchSnapshot:
    """Search state observed at item completion or response termination."""

    output_index: int
    item: Any


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


def _responses_tool_choice(request: NormalizedRequest) -> Any:
    if isinstance(request.tool_choice, WebSearchToolChoice):
        return {"type": "allowed_tools", "mode": "required", "tools": [{"type": "web_search"}]}
    if isinstance(request.tool_choice, FunctionToolChoice):
        return {"type": "function", "name": request.tool_choice.name}
    return request.tool_choice


def _chat_tool_choice(request: NormalizedRequest) -> Any:
    if isinstance(request.tool_choice, FunctionToolChoice):
        return {
            "type": "function",
            "function": {"name": request.tool_choice.name},
        }
    return request.tool_choice


def _prompt_cache_fields(request: NormalizedRequest) -> dict[str, Any]:
    """Project normalized cache intent onto the fields both OpenAI protocols share."""
    cache = request.prompt_cache
    fields: dict[str, Any] = {}
    options: dict[str, Any] = {}
    ttls = [breakpoint.ttl for breakpoint in request.iter_cache_breakpoints()]
    if cache is not None:
        if cache.mode is not None:
            options["mode"] = cache.mode
        if cache.key is not None:
            fields["prompt_cache_key"] = cache.key
        if cache.retention is not None:
            fields["prompt_cache_retention"] = cache.retention
        ttls.append(cache.ttl)
    if any(ttl is not None for ttl in ttls):
        # OpenAI only supports one request-wide TTL. This rounds 5m up and
        # approximates 1h with a shorter minimum under the documented cache policy.
        options["ttl"] = "30m"
    if options:
        fields["prompt_cache_options"] = options
    return fields


def _prompt_cache_breakpoint_fields(breakpoint: CacheBreakpoint | None) -> dict[str, Any]:
    return {"prompt_cache_breakpoint": {"mode": "explicit"}} if breakpoint is not None else {}


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
        text = ""
        citations = []
        for part in item.content:
            if part.type != "output_text":
                continue
            for annotation in part.annotations:
                if annotation.type == "url_citation":
                    citation = citation_from_responses(annotation.model_dump())
                    citation.start_index += len(text)
                    citation.end_index += len(text)
                    citations.append(citation)
            text += part.text
        return TextOutput(content=text, id=item.id, citations=citations)
    if item.type == "function_call":
        _validate_responses_function_call(item, statuses={None, "completed"})
        return _function_call_part(item)
    raise ValueError(f"Unsupported Responses output item: {item.type}")


def _search_parts(item: Any) -> tuple[SearchCall, SearchResult | None]:
    """Split a Responses item into a normalized invocation and optional linked result.

    Responses bundles both into one SDK item; Molto models them separately.
    For example, a completed search becomes a call carrying the query and a
    result carrying source URLs, linked by the call ID. Failed searches produce
    a result with error="unavailable"; unfinished searches and browsing actions
    produce only a call. Missing sources stay None rather than becoming [].
    The caller emits each returned part's start/end events.
    """
    data = item.model_dump()
    call = search_call_from_responses(data)
    result = None
    if call.action == "search" and call.status in {"completed", "failed"}:
        result = search_result_from_responses(data)
    return call, result


async def _ordered_items(
    events: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[ResponseStreamEvent | _SearchSnapshot]:
    """Stream items in start order, yielding search snapshots once classified."""
    # A search at index 1 may start without an action, then answer text at index 2
    # may arrive before the search's query and sources. Emitting that text now
    # would put the answer before its search in the normalized stream. Keep an
    # item FIFO: ordinary items stream at the head, while searches wait for their
    # final snapshot to distinguish search from browsing and link any results.
    # Interrupted searches may obtain that snapshot only from the terminal response.
    pending: dict[int, deque[ResponseStreamEvent | _SearchSnapshot]] = {}
    searches: dict[int, Any] = {}
    async for event in events:
        terminal = event.type in {"response.completed", "response.incomplete"}
        index = getattr(event, "output_index", None)
        if terminal:
            final_items = {item.id: item for item in event.response.output}
            for search_index, initial in searches.items():
                pending[search_index].append(
                    _SearchSnapshot(search_index, final_items.get(initial.id, initial))
                )
            searches.clear()
        elif event.type.startswith("response.web_search_call."):
            continue
        elif index is None:
            yield event
            continue
        else:
            chunks = pending.setdefault(index, deque())
            if event.type == "response.output_item.added" and event.item.type == "web_search_call":
                searches[index] = event.item
            elif (
                event.type == "response.output_item.done" and event.item.type == "web_search_call"
            ):
                searches.pop(index, None)
                chunks.append(_SearchSnapshot(index, event.item))
            else:
                chunks.append(event)
        while pending:
            index = next(iter(pending))
            chunks = pending[index]
            if not chunks:
                if terminal:
                    del pending[index]
                    continue
                break
            current = chunks.popleft()
            if isinstance(current, _SearchSnapshot) or current.type == "response.output_item.done":
                del pending[index]
            yield current
        if terminal:
            yield event
    if any(pending.values()):
        raise ValueError("Responses stream ended before queued items could be emitted.")


def _merge_search_result(items: list[dict[str, Any]], result: SearchResult) -> None:
    """Fold a normalized history result into a Responses search item in place.

    Responses bundles invocation and result together. For example, a result
    linked to call "s1" adds source URLs to that item's action and marks it
    completed, or failed when an error is present. Update the most recent
    matching call; if none exists, append one with the same ID and no query.
    Supplied sources replace the list (including []); None leaves it unchanged.
    """
    search_item = None
    for item in reversed(items):
        if item.get("type") == "web_search_call" and item["id"] == result.tool_call_id:
            search_item = item
            break

    if search_item is None:
        # Responses stores the invocation and result in one item. Standalone
        # history results retain their identity without inventing query metadata.
        search_item = search_call_to_responses(SearchCall(id=result.tool_call_id))
        items.append(search_item)

    search_item["status"] = "failed" if result.error is not None else "completed"
    if result.sources is not None:
        search_item["action"]["sources"] = [
            search_source_to_responses(source) for source in result.sources
        ]


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


class OpenAIResponsesClient(
    BaseClient[OpenAIResponsesCreateRequest, ResponseStreamEvent, OpenAIResponsesResponse]
):
    """OpenAI Responses transport for native and normalized streaming paths."""

    @staticmethod
    def _translate_tool(tool: FunctionTool | WebSearchTool) -> dict[str, Any]:
        """Translate one normalized tool into a Responses declaration."""
        if isinstance(tool, FunctionTool):
            return {"type": "function", **_function_tool_payload(tool)}
        if tool.blocked_domains is not None:
            raise UnsupportedInput(
                "Messages tools[].blocked_domains cannot be represented by Responses web search."
            )
        payload: dict[str, Any] = {"type": "web_search"}
        if tool.allowed_domains is not None:
            payload["filters"] = {"allowed_domains": tool.allowed_domains}
        if tool.search_context_size is not None:
            payload["search_context_size"] = tool.search_context_size
        if tool.user_location is not None:
            payload["user_location"] = {"type": "approximate", **tool.user_location.model_dump()}
        return payload

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    async def native_events(
        self, request: OpenAIResponsesCreateRequest
    ) -> AsyncIterator[ResponseStreamEvent]:
        upstream_response = None
        try:
            sdk_client = self._make_sdk_client()
            stream = await sdk_client.responses.create(
                **request.model_dump(),
                extra_body=self._extra_body,
                extra_headers=self._extra_headers,
            )
            if not isinstance(stream, AsyncStream):
                raise TypeError("Responses request did not return a stream.")
            self._capture_response_headers(stream.response.headers)
            upstream_response = stream.response
            async with stream:
                async for event in stream:
                    if isinstance(event, ResponseCompletedEvent):
                        usage = event.response.usage
                        if (
                            usage is not None
                            and usage.input_tokens_details.cache_write_tokens is None
                        ):
                            usage.input_tokens_details.cache_write_tokens = 0
                    # The SDK already constructs ResponseStreamEvent instances. Apart from
                    # the provider omission above, avoid validating every event twice.
                    yield event
        except Exception as error:
            self._handle_error(error, response=upstream_response)

    async def native_response(
        self, request: OpenAIResponsesCreateRequest
    ) -> OpenAIResponsesResponse:
        try:
            sdk_client = self._make_sdk_client()
            response = await sdk_client.responses.create(
                **request.model_dump(),
                extra_body=self._extra_body,
                extra_headers=self._extra_headers,
            )
            if isinstance(response, AsyncStream):
                raise TypeError("Responses request did not return a non-streaming response.")
            return response
        except Exception as error:
            self._handle_error(error)

    @staticmethod
    def _append_assistant_items(
        items: list[dict[str, Any]],
        parts: list[Thinking | TextOutput | ToolCall | SearchCall | SearchResult],
        *,
        message_id: str | None = None,
    ) -> None:
        """Link searches across accumulated history while grouping text within this message."""
        if message_id is not None and any(not isinstance(part, TextOutput) for part in parts):
            raise UnsupportedInput(
                "A Responses assistant message ID requires text-only message content."
            )
        message: dict[str, Any] | None = None
        for item in parts:
            if not isinstance(item, TextOutput):
                message = None
            if isinstance(item, TextOutput):
                if message is None:
                    message = {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [],
                    }
                    if message_id is not None:
                        message["id"] = message_id
                    items.append(message)
                message["content"].append(
                    {
                        "type": "output_text",
                        "text": item.content,
                        "annotations": [
                            citation_to_responses(citation) for citation in item.citations
                        ],
                        "logprobs": [],
                        **_prompt_cache_breakpoint_fields(item.cache_breakpoint),
                    }
                )
            elif isinstance(item, SearchCall):
                items.append(search_call_to_responses(item))
            elif isinstance(item, SearchResult):
                _merge_search_result(items, item)
            elif isinstance(item, Thinking):
                if item.summary and not item.signature:
                    raise UnsupportedInput("A Responses reasoning summary requires a signature.")
                # OpenAI rejects output lifecycle status on reasoning history input.
                reasoning_item = {
                    "type": "reasoning",
                    "summary": (
                        [{"type": "summary_text", "text": item.summary}] if item.summary else []
                    ),
                    "content": (
                        [{"type": "reasoning_text", "text": item.content}] if item.content else []
                    ),
                }
                if item.id is not None:
                    reasoning_item["id"] = item.id
                if item.signature is not None:
                    reasoning_item["encrypted_content"] = item.signature
                items.append(reasoning_item)
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
        if not parts:
            message = {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [],
            }
            if message_id is not None:
                message["id"] = message_id
            items.append(message)

    @staticmethod
    def _process_input(request: NormalizedRequest) -> list[dict[str, Any]] | None:
        if request.input is None:
            return None

        items: list[dict[str, Any]] = []
        for item in request.input:
            if isinstance(item, SystemPrompt):
                items.append(
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": item.content,
                                **_prompt_cache_breakpoint_fields(item.cache_breakpoint),
                            }
                        ],
                    }
                )
            elif isinstance(item, UserInput):
                content: list[dict[str, Any]] = []
                for part in item.content:
                    if isinstance(part, TextInput):
                        content.append(
                            {
                                "type": "input_text",
                                "text": part.content,
                                **_prompt_cache_breakpoint_fields(part.cache_breakpoint),
                            }
                        )
                    elif isinstance(part, ImageUrlInput):
                        image = {
                            "type": "input_image",
                            "image_url": part.url,
                            **_prompt_cache_breakpoint_fields(part.cache_breakpoint),
                        }
                        if part.detail is not None:
                            image["detail"] = part.detail
                        content.append(image)
                items.append({"role": "user", "content": content})
            elif isinstance(item, AssistantInput):
                OpenAIResponsesClient._append_assistant_items(
                    items, item.content, message_id=item.id
                )
            elif isinstance(item, (Thinking, ToolCall, SearchCall)):
                OpenAIResponsesClient._append_assistant_items(items, [item])
            elif isinstance(item, SearchResult):
                _merge_search_result(items, item)
            elif isinstance(item, ToolReturn):
                if item.outcome != "success":
                    raise UnsupportedInput(
                        "OpenAI protocols cannot preserve non-successful tool outcomes."
                    )
                output: list[dict[str, Any]] = []
                for part in item.content:
                    if isinstance(part, TextInput):
                        output.append(
                            {
                                "type": "input_text",
                                "text": part.content,
                                **_prompt_cache_breakpoint_fields(part.cache_breakpoint),
                            }
                        )
                    elif isinstance(part, ImageUrlInput):
                        image = {
                            "type": "input_image",
                            "image_url": part.url,
                            **_prompt_cache_breakpoint_fields(part.cache_breakpoint),
                        }
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
                raise UnsupportedInput(f"Unsupported Responses input item: {type(item)}")
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
        if request.tools is not None:
            body["tools"] = [cls._translate_tool(tool) for tool in request.tools]
        tool_choice = _responses_tool_choice(request)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if request.reasoning is not None:
            body["reasoning"] = request.reasoning.model_dump(exclude={"extra_fields"})
        if request.stream_options is not None:
            body["stream_options"] = request.stream_options.model_dump(exclude={"extra_fields"})
        if any(isinstance(tool, WebSearchTool) for tool in request.tools or []):
            body["include"] = ["web_search_call.action.sources"]
        body.update(_prompt_cache_fields(request))
        try:
            return OpenAIResponsesCreateRequest.model_validate(body).model_dump()
        except ValidationError as error:
            raise UnsupportedInput(str(error)) from error

    async def _events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        request_body = self._request_body(request)
        native_request = OpenAIResponsesCreateRequest.model_validate(request_body)
        # SDK extra_body can introduce or replace the projected tool declarations.
        tools = (self._extra_body or {}).get("tools", native_request.tools) or []
        search_enabled = any(tool["type"] == "web_search" for tool in tools)
        if search_enabled and self._extra_body is not None:
            # SDK extra_body wins over typed request parameters. Retain caller
            # include entries while keeping search sources enabled on translation.
            includes = ["web_search_call.action.sources", *(native_request.include or [])]
            includes.extend(self._extra_body.get("include") or [])
            unique_includes = list(dict.fromkeys(includes))
            self._extra_body = {**self._extra_body, "include": unique_includes}
        # Only a reasoning item whose `added` snapshot is completely empty (no
        # content, summary, or encrypted state) is deferred: if nothing ever
        # arrives, its part lifecycle is omitted from the IR entirely. An item
        # that already carries content or encrypted state at `added` starts
        # immediately, so a stalled or failed stream cannot lose it.
        pending_reasoning: dict[int, Thinking] = {}
        # Source output indices map to dense normalized indices so an omitted
        # reasoning item cannot leave a gap for downstream projections.
        part_indices: dict[int, int] = {}
        text_indices: dict[tuple[int, int], int] = {}
        started_text_items: set[int] = set()
        ended_text_items: set[int] = set()
        next_index = 0
        # SDK overrides replace history as well as tools; buffering must reflect
        # the input actually sent upstream, including cited follow-ups without tools.
        input_items = (self._extra_body or {}).get("input", request_body.get("input"))
        defer_citations = search_enabled or (
            isinstance(input_items, list)
            and any(
                item.get("type") == "web_search_call"
                or any(
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and part.get("annotations")
                    for part in item.get("content") or []
                )
                for item in input_items
            )
        )

        def start_part(
            source_index: int, part: Thinking | TextOutput | ToolCall | SearchCall | SearchResult
        ) -> PartStartEvent:
            nonlocal next_index
            part_indices[source_index] = next_index
            next_index += 1
            return PartStartEvent(index=part_indices[source_index], part=part)

        def part_index(source_index: int) -> int:
            # A delta or completion can reference an item whose start was never
            # seen; keep its source index rather than inventing a dense one.
            return part_indices.get(source_index, source_index)

        async with (
            aclosing(self.native_events(native_request)) as native_events,
            aclosing(_ordered_items(native_events)) as ordered_events,
        ):
            async for event in ordered_events:
                if isinstance(event, _SearchSnapshot):
                    if getattr(event.item, "action", None) is None:
                        continue
                    call, result = _search_parts(event.item)
                    yield start_part(event.output_index, call)
                    yield PartEndEvent(index=part_index(event.output_index), part=call)
                    if result is not None:
                        index = next_index
                        next_index += 1
                        yield PartStartEvent(index=index, part=result)
                        yield PartEndEvent(index=index, part=result)
                    continue
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
                    if isinstance(part, TextOutput):
                        part.defer_citations = defer_citations
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
                            raise ValueError(
                                "Summarized Responses reasoning cannot contain raw text."
                            )
                        if (start := pending_reasoning.pop(event.output_index, None)) is not None:
                            yield start_part(event.output_index, start)
                        yield PartDeltaEvent(
                            index=part_index(event.output_index),
                            delta=ThinkingDelta(content_delta=added_part.text),
                        )
                    elif added_part.type == "output_text":
                        key = (event.output_index, event.content_index)
                        # Non-text snapshots may precede the first text content.
                        if event.output_index not in started_text_items:
                            started_text_items.add(event.output_index)
                            text_indices[key] = part_index(event.output_index)
                        else:
                            index = next_index
                            next_index += 1
                            text_indices[key] = index
                            yield PartStartEvent(
                                index=index,
                                part=TextOutput(
                                    content="", id=event.item_id, defer_citations=defer_citations
                                ),
                            )
                        if added_part.text:
                            yield PartDeltaEvent(
                                index=text_indices[key],
                                delta=TextDelta(content_delta=added_part.text),
                            )
                        for annotation in added_part.annotations:
                            if annotation.type == "url_citation":
                                yield PartDeltaEvent(
                                    index=text_indices[key],
                                    delta=CitationDelta(
                                        citation=citation_from_responses(annotation.model_dump())
                                    ),
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
                        index=text_indices.get(
                            (event.output_index, event.content_index),
                            part_index(event.output_index),
                        ),
                        delta=TextDelta(content_delta=event.delta),
                    )
                elif event_type == "response.output_text.annotation.added":
                    annotation = event.annotation
                    # SDK versions expose annotation deltas as dictionaries or models.
                    if isinstance(annotation, BaseModel):
                        annotation = annotation.model_dump()
                    if (
                        not isinstance(annotation, dict)
                        or annotation.get("type") != "url_citation"
                    ):
                        raise ValueError(f"Unsupported Responses annotation: {event.annotation}")
                    yield PartDeltaEvent(
                        index=text_indices.get(
                            (event.output_index, event.content_index),
                            part_index(event.output_index),
                        ),
                        delta=CitationDelta(citation=citation_from_responses(annotation)),
                    )
                elif event_type == "response.content_part.done":
                    if event.part.type == "output_text" and event.output_index in part_indices:
                        ended_text_items.add(event.output_index)
                        yield PartEndEvent(
                            index=text_indices.get(
                                (event.output_index, event.content_index),
                                part_index(event.output_index),
                            ),
                            part=TextOutput(
                                content=event.part.text,
                                id=event.item_id,
                                defer_citations=defer_citations,
                                citations=[
                                    citation_from_responses(annotation.model_dump())
                                    for annotation in event.part.annotations
                                    if annotation.type == "url_citation"
                                ],
                            ),
                        )
                elif event_type == "response.function_call_arguments.delta":
                    yield PartDeltaEvent(
                        index=part_index(event.output_index),
                        delta=ToolCallDelta(args_delta=event.delta),
                    )
                elif event_type == "response.output_item.done":
                    if event.item.type == "message" and event.output_index in ended_text_items:
                        continue
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
                            termination_reason = "unknown"
                        else:
                            termination_reason = incomplete_details.reason
                    usage = _responses_usage(metadata.usage)
                    tool_usage = getattr(metadata, "tool_usage", None)
                    search_usage = (
                        (tool_usage.get("web_search") or {})
                        if isinstance(tool_usage, dict)
                        else {}
                    )
                    if (
                        defer_citations
                        and usage is not None
                        and search_usage.get("num_requests") is not None
                    ):
                        usage.extra_fields["web_search_requests"] = search_usage["num_requests"]
                    yield ResponseEndEvent(
                        id=metadata.id,
                        created_at=int(metadata.created_at),
                        model=metadata.model,
                        service_tier=metadata.service_tier,
                        termination_reason=termination_reason,
                        usage=usage,
                        **(
                            {"web_search_requests": search_usage["num_requests"]}
                            if defer_citations and search_usage.get("num_requests") is not None
                            else {}
                        ),
                    )
                elif event_type in _IGNORED_RESPONSES_EVENTS:
                    continue
                else:
                    raise ValueError(f"Unsupported Responses stream event: {event_type}")

    def _handle_error(self, error: Exception, *, response: Response | None = None) -> NoReturn:
        if (
            isinstance(error, OpenAIError)
            and not isinstance(error, APIError)
            and str(error).startswith("Missing credentials.")
        ):
            raise AuthenticationError(
                str(error),
                response=None,
                body=None,
                request=self._make_authentication_request("responses"),
            ) from error
        if isinstance(error, APIStatusError):
            # Specific error codes take precedence over broader SDK classes,
            # which also recognize errors without a structured body.
            if error.code == "context_length_exceeded":
                raise ContextLengthExceededError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
            if error.code == "model_not_found":
                raise ModelNotFoundError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
            if isinstance(error, OpenAIAuthenticationError):
                raise AuthenticationError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
            if isinstance(error, OpenAINotFoundError):
                raise ResourceNotFoundError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
        elif isinstance(error, APIError):
            # SSE APIError has no response; native_events supplies the stream response.
            if error.code == "context_length_exceeded" and response is not None:
                raise ContextLengthExceededError(
                    error.message,
                    response=response,
                    body=error.body,
                ) from error
        raise error


class OpenAIChatCompletionsClient(
    BaseClient[
        OpenAIChatCompletionsCreateRequest,
        OpenAIChatCompletionsStreamChunk,
        OpenAIChatCompletionsResponse,
    ]
):
    """OpenAI Chat Completions transport for native and normalized streaming paths."""

    @staticmethod
    def _translate_tool(tool: FunctionTool) -> dict[str, Any]:
        """Translate one normalized function tool into a Chat Completions declaration."""
        return {"type": "function", "function": _function_tool_payload(tool)}

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    async def native_events(
        self, request: OpenAIChatCompletionsCreateRequest
    ) -> AsyncIterator[OpenAIChatCompletionsStreamChunk]:
        try:
            sdk_client = self._make_sdk_client()
            stream = await sdk_client.chat.completions.create(
                **request.model_dump(),
                extra_body=self._extra_body,
                extra_headers=self._extra_headers,
            )
            if not isinstance(stream, AsyncStream):
                raise TypeError("Chat Completions request did not return a stream.")
            self._capture_response_headers(stream.response.headers)
            async with stream:
                async for chunk in stream:
                    # The SDK already constructs ChatCompletionChunk instances. Trust that
                    # boundary to avoid serializing and validating every streaming chunk twice.
                    yield chunk
        except Exception as error:
            self._handle_error(error)

    async def native_response(
        self, request: OpenAIChatCompletionsCreateRequest
    ) -> OpenAIChatCompletionsResponse:
        try:
            sdk_client = self._make_sdk_client()
            response = await sdk_client.chat.completions.create(
                **request.model_dump(),
                extra_body=self._extra_body,
                extra_headers=self._extra_headers,
            )
            if isinstance(response, AsyncStream):
                raise TypeError(
                    "Chat Completions request did not return a non-streaming response."
                )
            return response
        except Exception as error:
            self._handle_error(error)

    @staticmethod
    def _assistant_message(parts: list[Thinking | TextOutput | ToolCall]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant"}
        reasoning_parts: list[str] = []
        text_parts: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, Thinking):
                if part.signature is not None:
                    raise UnsupportedInput(
                        "Chat Completions cannot represent thinking signatures."
                    )
                if text_parts or "tool_calls" in message:
                    raise UnsupportedInput(
                        "Chat Completions cannot preserve this assistant reasoning order."
                    )
                reasoning_parts.append(part.content or part.summary)
            elif isinstance(part, TextOutput):
                if "tool_calls" in message:
                    raise UnsupportedInput(
                        "Chat Completions cannot preserve assistant text after tool calls."
                    )
                text_parts.append(
                    {
                        "type": "text",
                        "text": part.content,
                        **_prompt_cache_breakpoint_fields(part.cache_breakpoint),
                    }
                )
            elif isinstance(part, ToolCall):
                message.setdefault("tool_calls", []).append(
                    {
                        "id": part.tool_call_id,
                        "type": "function",
                        "function": {"name": part.tool_name, "arguments": part.args or ""},
                    }
                )
        if reasoning_parts:
            message["reasoning_content"] = "\n".join(reasoning_parts)
        if len(text_parts) == 1 and "prompt_cache_breakpoint" not in text_parts[0]:
            message["content"] = text_parts[0]["text"]
        else:
            message["content"] = text_parts if text_parts or not parts else None
        return message

    @staticmethod
    def _messages(request: NormalizedRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        assistant_parts: list[Thinking | TextOutput | ToolCall] = []
        explicit_assistant = False
        # Chat requires the complete tool-response block immediately after the
        # assistant, so image-bearing user messages wait until that block ends.
        deferred_image_messages: list[dict[str, Any]] = []

        def flush_assistant() -> None:
            nonlocal explicit_assistant
            if explicit_assistant or assistant_parts:
                messages.append(OpenAIChatCompletionsClient._assistant_message(assistant_parts))
                assistant_parts.clear()
                explicit_assistant = False

        def flush_deferred_images() -> None:
            if deferred_image_messages:
                messages.extend(deferred_image_messages)
                deferred_image_messages.clear()

        for item in request.input or []:
            if not isinstance(item, ToolReturn):
                flush_deferred_images()
            if isinstance(item, SystemPrompt):
                flush_assistant()
                messages.append(
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "text",
                                "text": item.content,
                                **_prompt_cache_breakpoint_fields(item.cache_breakpoint),
                            }
                        ],
                    }
                )
            elif isinstance(item, UserInput):
                flush_assistant()
                content: list[dict[str, Any]] = []
                for part in item.content:
                    if isinstance(part, TextInput):
                        content.append(
                            {
                                "type": "text",
                                "text": part.content,
                                **_prompt_cache_breakpoint_fields(part.cache_breakpoint),
                            }
                        )
                    elif isinstance(part, ImageUrlInput):
                        image_url = {"url": part.url}
                        if part.detail is not None:
                            image_url["detail"] = part.detail
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": image_url,
                                **_prompt_cache_breakpoint_fields(part.cache_breakpoint),
                            }
                        )
                messages.append({"role": "user", "content": content})
            elif isinstance(item, AssistantInput):
                # Standalone Responses reasoning/calls can share a wire message,
                # but two explicit source messages must retain their boundary.
                if (
                    explicit_assistant
                    or any(isinstance(part, ToolCall) for part in assistant_parts)
                    or any(isinstance(part, Thinking) for part in item.content)
                ):
                    flush_assistant()
                assistant_parts.extend(item.content)
                explicit_assistant = True
            elif isinstance(item, Thinking):
                flush_assistant()
                assistant_parts.append(item)
            elif isinstance(item, ToolCall):
                assistant_parts.append(item)
            elif isinstance(item, ToolReturn):
                flush_assistant()
                if item.outcome != "success":
                    raise UnsupportedInput(
                        "OpenAI protocols cannot preserve non-successful tool outcomes."
                    )
                if (
                    deferred_image_messages
                    or any(isinstance(part, ImageUrlInput) for part in item.content)
                ) and any(part.cache_breakpoint is not None for part in item.content):
                    # TODO: Some CC implementation such as vLLM and SGLang supports image tool content
                    # TODO: Only raise this if the upstream CC does not support image tool content
                    raise UnsupportedInput(
                        "Chat Completions cannot preserve prompt cache breakpoints when "
                        "tool-result images must move past the marked part."
                    )
                content: list[dict[str, Any]] = []
                image_content: list[dict[str, Any]] = []
                for part in item.content:
                    if isinstance(part, TextInput):
                        content.append(
                            {
                                "type": "text",
                                "text": part.content,
                                **_prompt_cache_breakpoint_fields(part.cache_breakpoint),
                            }
                        )
                    elif isinstance(part, ImageUrlInput):
                        image_url = {"url": part.url}
                        if part.detail is not None:
                            image_url["detail"] = part.detail
                        image_content.append({"type": "image_url", "image_url": image_url})
                if image_content:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                f"The image result of tool call {item.tool_call_id!r} will be "
                                "attached as user prompt after all tool responses."
                            ),
                        }
                    )
                    deferred_image_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Image returned by tool call {item.tool_call_id!r}:",
                                },
                                *image_content,
                            ],
                        }
                    )
                messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": item.tool_call_id,
                    }
                )
            else:
                raise UnsupportedInput(f"Unsupported Chat Completions input item: {type(item)}")
        flush_assistant()
        flush_deferred_images()
        return messages

    @classmethod
    def _request_body(cls, request: NormalizedRequest) -> dict[str, Any]:
        cls._validate_request(request)
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
        if request.tools is not None:
            body["tools"] = [cls._translate_tool(tool) for tool in request.tools]
        tool_choice = _chat_tool_choice(request)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if request.reasoning is not None and request.reasoning.effort is not None:
            body["reasoning_effort"] = request.reasoning.effort
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        body.update(_prompt_cache_fields(request))
        try:
            return OpenAIChatCompletionsCreateRequest.model_validate(body).model_dump()
        except ValidationError as error:
            raise UnsupportedInput(str(error)) from error

    @staticmethod
    def _validate_request(request: NormalizedRequest) -> None:
        if (
            any(isinstance(tool, WebSearchTool) for tool in request.tools or [])
            or isinstance(request.tool_choice, WebSearchToolChoice)
            or any(
                isinstance(part, (SearchCall, SearchResult))
                or isinstance(part, TextOutput)
                and bool(part.citations)
                for item in request.input or []
                for part in (item.content if isinstance(item, AssistantInput) else [item])
            )
        ):
            raise UnsupportedInput(
                "Hosted web search cannot use Chat Completions upstream; select Responses or Messages."
            )
        if request.max_tool_calls is not None:
            raise UnsupportedInput("Chat Completions cannot represent max_tool_calls.")
        if request.reasoning is not None and request.reasoning.summary not in (
            None,
            "auto",
            "detailed",
        ):
            raise UnsupportedInput(
                "Chat Completions cannot represent concise reasoning summaries."
            )
        for item in request.input or []:
            if isinstance(item, ToolReturn) and item.tool_name is not None:
                raise UnsupportedInput("Chat Completions cannot represent tool return names.")

    async def _events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        if (self._extra_body or {}).get("web_search_options") is not None:
            raise UnsupportedInput(
                "Hosted web search cannot use Chat Completions upstream; select Responses or Messages."
            )
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

        native_request = OpenAIChatCompletionsCreateRequest.model_validate(
            self._request_body(request)
        )
        async with aclosing(self.native_events(native_request)) as native_events:
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

                            name_delta = (
                                function_delta.name if function_delta is not None else None
                            )
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

    def _handle_error(self, error: Exception) -> NoReturn:
        if (
            isinstance(error, OpenAIError)
            and not isinstance(error, APIError)
            and str(error).startswith("Missing credentials.")
        ):
            # The OpenAI SDK raises a generic OpenAIError for missing credentials when SDK is used.
            raise AuthenticationError(
                str(error),
                response=None,
                body=None,
                request=self._make_authentication_request("chat/completions"),
            ) from error
        if isinstance(error, APIStatusError):
            # Specific error codes take precedence over broader SDK classes,
            # which also recognize errors without a structured body.
            if error.code == "context_length_exceeded":
                raise ContextLengthExceededError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
            if error.code == "model_not_found":
                raise ModelNotFoundError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
            if isinstance(error, OpenAIAuthenticationError):
                raise AuthenticationError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
            if isinstance(error, OpenAINotFoundError):
                raise ResourceNotFoundError(
                    error.message,
                    response=error.response,
                    body=error.body,
                ) from error
        raise error
