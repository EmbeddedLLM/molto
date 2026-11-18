"""OpenTelemetry attribute keys and span attributes for coordinator calls.

Only the `Molto` coordinators use this module; adapters and upstream clients stay
telemetry-free. See `docs/observability.md` for the span contract.
"""

import traceback
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, fields
from typing import Any, Literal, TypeAlias

import anthropic
import openai
from anthropic.types import MessageDeltaUsage, OutputTokensDetails, ServerToolUsage
from anthropic.types import Usage as AnthropicUsage
from httpx import URL, InvalidURL
from loguru import logger
from openai.types import CompletionUsage
from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from molto.providers import Endpoint, Provider
from molto.types.anthropic_messages import AnthropicMessagesResponse, AnthropicMessagesStreamEvent
from molto.types.exceptions import APIException
from molto.types.normalized import (
    ModelResponseStreamEvent,
    PartDeltaEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
)
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsResponse,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import OpenAIResponsesResponse, OpenAIResponsesStreamEvent
from molto.upstream.openai import responses_web_search_requests
from molto.version import __version__

# Keys follow the GenAI semantic conventions. They are plain strings rather than
# imports from `opentelemetry-semantic-conventions`, which exposes GenAI keys only
# from a private, Development-status module that lags the current spec.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"
MOLTO_DOWNSTREAM_PROTOCOL = "molto.downstream.protocol"
MOLTO_ROUTE = "molto.route"
MOLTO_UPSTREAM_ENDPOINT = "molto.upstream.endpoint"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
# The conventions define no web-search or server-tool usage count, so this one is Molto's.
MOLTO_USAGE_WEB_SEARCH_REQUESTS = "molto.usage.web_search_requests"
ERROR_TYPE = "error.type"
HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"
MOLTO_ERROR_PHASE = "molto.error.phase"
MOLTO_STREAM_OUTCOME = "molto.stream.outcome"
FIRST_TOKEN_EVENT = "molto.first_token"

# Every span attribute the contract defines. Spans carry no other keys, so no
# attribute can hold request or completion content (docs/observability.md).
ATTRIBUTE_KEYS = frozenset(
    {
        GEN_AI_OPERATION_NAME,
        GEN_AI_PROVIDER_NAME,
        GEN_AI_REQUEST_CHOICE_COUNT,
        GEN_AI_REQUEST_FREQUENCY_PENALTY,
        GEN_AI_REQUEST_MAX_TOKENS,
        GEN_AI_REQUEST_MODEL,
        GEN_AI_REQUEST_PRESENCE_PENALTY,
        GEN_AI_REQUEST_SEED,
        GEN_AI_REQUEST_STREAM,
        GEN_AI_REQUEST_TEMPERATURE,
        GEN_AI_REQUEST_TOP_K,
        GEN_AI_REQUEST_TOP_P,
        SERVER_ADDRESS,
        SERVER_PORT,
        MOLTO_DOWNSTREAM_PROTOCOL,
        MOLTO_ROUTE,
        MOLTO_UPSTREAM_ENDPOINT,
        GEN_AI_RESPONSE_FINISH_REASONS,
        GEN_AI_RESPONSE_ID,
        GEN_AI_RESPONSE_MODEL,
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        GEN_AI_USAGE_INPUT_TOKENS,
        GEN_AI_USAGE_OUTPUT_TOKENS,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        MOLTO_USAGE_WEB_SEARCH_REQUESTS,
        ERROR_TYPE,
        HTTP_RESPONSE_STATUS_CODE,
        MOLTO_ERROR_PHASE,
        MOLTO_STREAM_OUTCOME,
    }
)

ErrorPhase = Literal["setup", "stream"]
# How a streaming call ended: the caller consumed it, closed it early, cancelled it, or
# dropped it without closing it, or it failed. Set on every streaming span, including
# failures before the first event.
StreamOutcome = Literal["completed", "closed", "cancelled", "error", "abandoned"]
# Native extractors can meet a provider that omits a field; `SpanRecorder.set_known` drops
# the `None` values.
MaybeAttributes: TypeAlias = dict[str, AttributeValue | None]

# The cross-vendor operation type for every coordinator, not Molto's protocol name:
# backends recognize LLM spans by well-known operation names.
OPERATION_CHAT = "chat"

# Downstream request fields holding the output-token limit, in precedence order.
# Chat Completions still accepts the deprecated `max_tokens`. The first field with an
# integer value wins, so a mistyped preferred field does not hide a valid fallback.
_MAX_TOKENS_FIELDS: dict[Endpoint, tuple[str, ...]] = {
    "responses": ("max_output_tokens",),
    "chat/completions": ("max_completion_tokens", "max_tokens"),
    "messages": ("max_tokens",),
}
_DEFAULT_PORTS = {"http": 80, "https": 443}


def get_tracer() -> trace.Tracer:
    """Return Molto's tracer from the application's provider, a no-op without an SDK."""
    return trace.get_tracer("molto", __version__)


def span_name(attributes: Mapping[str, AttributeValue]) -> str:
    """Name a span `{operation} {model}`, or `{operation}` when the request has no model."""
    model = attributes.get(GEN_AI_REQUEST_MODEL)
    return f"{OPERATION_CHAT} {model}" if model else OPERATION_CHAT


@dataclass(frozen=True)
class ResolvedRoute:
    """Route values a coordinator resolves before transport.

    Attributes:
        protocol: Downstream protocol, from the coordinator.
        endpoint: Effective upstream endpoint.
        provider: Resolved provider, reported unchanged as `gen_ai.provider.name`.
        base_url: Effective upstream base URL: the caller's, or the client's default.
    """

    protocol: Endpoint
    endpoint: Endpoint
    provider: Provider
    base_url: str


def start_attributes(
    route: ResolvedRoute, body: Mapping[str, Any] | BaseModel, *, streaming: bool
) -> dict[str, AttributeValue]:
    """Return the span attributes known before transport.

    Args:
        route: The coordinator's resolved route.
        body: Downstream request: a native dictionary or a validated request model.
        streaming: Whether the coordinator streams the call, decided by its own rule.
    """
    attributes: dict[str, AttributeValue] = {
        GEN_AI_OPERATION_NAME: OPERATION_CHAT,
        GEN_AI_PROVIDER_NAME: route.provider,
        MOLTO_DOWNSTREAM_PROTOCOL: route.protocol,
        MOLTO_UPSTREAM_ENDPOINT: route.endpoint,
        MOLTO_ROUTE: "native" if route.protocol == route.endpoint else "translated",
    }
    # Native bodies are unvalidated dictionaries, so record a field only when it
    # has the expected type; telemetry must never reject or alter a request.
    model = _field(body, "model")
    if isinstance(model, str) and model:
        attributes[GEN_AI_REQUEST_MODEL] = model
    for name in _MAX_TOKENS_FIELDS[route.protocol]:
        value = _field(body, name)
        if _is_int(value):
            attributes[GEN_AI_REQUEST_MAX_TOKENS] = value
            break
    for key, name in (
        (GEN_AI_REQUEST_TEMPERATURE, "temperature"),
        (GEN_AI_REQUEST_TOP_P, "top_p"),
        (GEN_AI_REQUEST_FREQUENCY_PENALTY, "frequency_penalty"),
        (GEN_AI_REQUEST_PRESENCE_PENALTY, "presence_penalty"),
    ):
        value = _field(body, name)
        if _is_int(value) or isinstance(value, float):
            attributes[key] = float(value)
    # Molto's request types define these fields on one protocol each (Messages `top_k`,
    # Chat `seed` and `n`, and the penalties above), but native passthrough forwards
    # unknown fields, so they are read from any native body as sent. Translation rejects
    # them, and the span of a rejected call still shows the value that caused it.
    for key, name in (
        (GEN_AI_REQUEST_TOP_K, "top_k"),
        (GEN_AI_REQUEST_SEED, "seed"),
    ):
        value = _field(body, name)
        if _is_int(value):
            attributes[key] = value
    # The convention reports a choice count only when it differs from the default of one.
    choice_count = _field(body, "n")
    if _is_int(choice_count) and choice_count != 1:
        attributes[GEN_AI_REQUEST_CHOICE_COUNT] = choice_count
    if streaming:
        attributes[GEN_AI_REQUEST_STREAM] = True
    attributes.update(_server_attributes(route.base_url))
    return attributes


def _field(body: Mapping[str, Any] | BaseModel, name: str) -> Any:
    if isinstance(body, Mapping):
        return body.get(name)
    return getattr(body, name, None)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _server_attributes(base_url: str) -> dict[str, AttributeValue]:
    # An unparsable URL must surface from the SDK at transport, not from telemetry.
    try:
        url = URL(base_url)
    except InvalidURL:
        return {}
    if not url.host:
        return {}
    port = url.port or _DEFAULT_PORTS.get(url.scheme)
    return {SERVER_ADDRESS: url.host} | ({SERVER_PORT: port} if port else {})


def usage_attributes(
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    web_search_requests: int | None = None,
) -> dict[str, AttributeValue]:
    """Return token-usage attributes.

    `input_tokens` must already include cached tokens and `output_tokens` reasoning
    tokens, as every provider Molto supports reports them.
    """
    attributes: dict[str, AttributeValue] = {}
    if _is_int(input_tokens):
        attributes[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    if _is_int(output_tokens):
        attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
    # Normalized usage stores a missing detail count as zero, so detail counts are
    # reported only when non-zero on every route, as the convention recommends.
    for key, value in (
        (GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens),
        (GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, cache_write_tokens),
        (GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning_tokens),
        (MOLTO_USAGE_WEB_SEARCH_REQUESTS, web_search_requests),
    ):
        if _is_int(value) and value:
            attributes[key] = value
    return attributes


# Native passthrough does not validate upstream output, so the native extractors below
# read every field that a provider may omit with a default; telemetry must never turn
# a successful call into a failure.


def normalized_event_attributes(event: ModelResponseStreamEvent) -> dict[str, AttributeValue]:
    """Return end attributes carried by a translated route's normalized lifecycle event.

    Translated spans report Molto's normalized termination reason, not the
    upstream provider's own value.
    """
    if isinstance(event, ResponseStartEvent):
        return {GEN_AI_RESPONSE_ID: event.id, GEN_AI_RESPONSE_MODEL: event.model}
    if not isinstance(event, ResponseEndEvent):
        return {}
    attributes: dict[str, AttributeValue] = {
        GEN_AI_RESPONSE_ID: event.id,
        GEN_AI_RESPONSE_MODEL: event.model,
        GEN_AI_RESPONSE_FINISH_REASONS: [event.termination_reason],
    }
    if event.usage is not None:
        details = event.usage.input_tokens_details
        attributes |= usage_attributes(
            event.usage.input_tokens,
            event.usage.output_tokens,
            cache_read_tokens=details.cached_tokens,
            cache_write_tokens=details.cache_write_tokens,
            reasoning_tokens=event.usage.output_tokens_details.reasoning_tokens,
            web_search_requests=event.usage.web_search_requests,
        )
    return attributes


def is_normalized_content_delta(event: ModelResponseStreamEvent) -> bool:
    """Return whether a normalized event carries text, reasoning, or tool-call arguments."""
    if not isinstance(event, PartDeltaEvent):
        return False
    delta = event.delta
    if isinstance(delta, TextDelta):
        return bool(delta.content_delta)
    if isinstance(delta, ThinkingDelta):
        # A signature-only delta replaces opaque state and carries no content.
        return bool(delta.content_delta or delta.summary_delta)
    if isinstance(delta, ToolCallDelta):
        return bool(delta.args_delta)
    return False


def responses_response_attributes(response: OpenAIResponsesResponse) -> MaybeAttributes:
    """Return end attributes for a native Responses `Response`.

    Responses has no finish-reason field, so the span reports the incomplete reason
    when the response is incomplete and its terminal status otherwise.
    """
    status = getattr(response, "status", None)
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None) if status == "incomplete" else None
    attributes: MaybeAttributes = {
        GEN_AI_RESPONSE_ID: getattr(response, "id", None),
        GEN_AI_RESPONSE_MODEL: getattr(response, "model", None),
        GEN_AI_RESPONSE_FINISH_REASONS: [reason or status] if reason or status else None,
    }
    usage = getattr(response, "usage", None)
    if usage is not None:
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        attributes |= usage_attributes(
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            cache_read_tokens=getattr(input_details, "cached_tokens", None),
            cache_write_tokens=getattr(input_details, "cache_write_tokens", None),
            reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
            web_search_requests=responses_web_search_requests(response),
        )
    return attributes


def is_responses_content_delta(event: OpenAIResponsesStreamEvent) -> bool:
    """Return whether a native Responses event carries text, reasoning, or tool-call arguments."""
    return getattr(event, "type", None) in _RESPONSES_CONTENT_DELTAS and bool(
        getattr(event, "delta", None)
    )


def chat_completions_usage_attributes(usage: CompletionUsage) -> dict[str, AttributeValue]:
    """Return token-usage attributes for Chat Completions usage."""
    details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return usage_attributes(
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        cache_read_tokens=getattr(details, "cached_tokens", None),
        cache_write_tokens=getattr(details, "cache_write_tokens", None),
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", None),
    )


def chat_completions_finish_reasons(choices: Any) -> dict[int, str]:
    """Return each choice's finish reason by choice index, skipping unfinished choices."""
    reasons: dict[int, str] = {}
    for choice in choices or []:
        reason = getattr(choice, "finish_reason", None)
        if reason:
            index = getattr(choice, "index", None)
            reasons[index if _is_int(index) else 0] = reason
    return reasons


def chat_completions_response_attributes(
    completion: OpenAIChatCompletionsResponse,
) -> MaybeAttributes:
    """Return end attributes for a native non-streaming `ChatCompletion`."""
    reasons = chat_completions_finish_reasons(getattr(completion, "choices", None))
    attributes: MaybeAttributes = {
        GEN_AI_RESPONSE_ID: getattr(completion, "id", None),
        GEN_AI_RESPONSE_MODEL: getattr(completion, "model", None),
        GEN_AI_RESPONSE_FINISH_REASONS: [reasons[index] for index in sorted(reasons)] or None,
    }
    usage = getattr(completion, "usage", None)
    if usage is not None:
        attributes |= chat_completions_usage_attributes(usage)
    return attributes


def is_chat_completions_content_delta(chunk: OpenAIChatCompletionsStreamChunk) -> bool:
    """Return whether a native Chat chunk carries text, reasoning, or tool-call arguments."""
    for choice in getattr(chunk, "choices", None) or []:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        # `reasoning_content` is a compatible-provider extension, such as vLLM's.
        if any(
            getattr(delta, field, None) for field in ("content", "refusal", "reasoning_content")
        ):
            return True
        for call in getattr(delta, "tool_calls", None) or []:
            if getattr(getattr(call, "function", None), "arguments", None):
                return True
    return False


@dataclass
class MessagesUsage:
    """Anthropic Messages token counts; `message_delta` updates the `message_start` snapshot."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    output_tokens_details: OutputTokensDetails | None = None
    server_tool_use: ServerToolUsage | None = None

    def update(self, usage: AnthropicUsage | MessageDeltaUsage | None) -> None:
        """Take every count `usage` reports, keeping earlier counts it omits."""
        for field in fields(self):
            value = getattr(usage, field.name, None)
            if value is not None:
                setattr(self, field.name, value)

    def attributes(self) -> dict[str, AttributeValue]:
        # Anthropic `input_tokens` excludes cached tokens; the convention and normalized
        # usage include them, so native and translated spans agree for the same call.
        cache_read = self.cache_read_input_tokens
        cache_write = self.cache_creation_input_tokens
        total_input = (
            self.input_tokens + (cache_read or 0) + (cache_write or 0)
            if _is_int(self.input_tokens)
            else None
        )
        return usage_attributes(
            total_input,
            self.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            # Anthropic `output_tokens` already includes thinking tokens.
            reasoning_tokens=getattr(self.output_tokens_details, "thinking_tokens", None),
            web_search_requests=getattr(self.server_tool_use, "web_search_requests", None),
        )


def messages_response_attributes(message: AnthropicMessagesResponse) -> MaybeAttributes:
    """Return end attributes for a native non-streaming Messages `Message`."""
    stop_reason = getattr(message, "stop_reason", None)
    usage = MessagesUsage()
    usage.update(getattr(message, "usage", None))
    return {
        GEN_AI_RESPONSE_ID: getattr(message, "id", None),
        GEN_AI_RESPONSE_MODEL: getattr(message, "model", None),
        GEN_AI_RESPONSE_FINISH_REASONS: [stop_reason] if stop_reason else None,
        **usage.attributes(),
    }


def is_messages_content_delta(event: AnthropicMessagesStreamEvent) -> bool:
    """Return whether a native Messages event carries text, reasoning, or tool-call arguments."""
    if getattr(event, "type", None) != "content_block_delta":
        return False
    delta = getattr(event, "delta", None)
    field = _MESSAGES_CONTENT_DELTAS.get(getattr(delta, "type", None))
    return field is not None and bool(getattr(delta, field, None))


def error_attributes(error: BaseException, phase: ErrorPhase) -> dict[str, AttributeValue]:
    """Return low-cardinality error attributes; the message is never an attribute."""
    normalized = next(
        (
            candidate
            for candidate in (error, error.__cause__)
            if isinstance(candidate, APIException)
        ),
        None,
    )
    # A recognized failure reports its normalized class, so native and translated
    # routes agree; the outward exception is the caller's SDK-specific class.
    attributes: dict[str, AttributeValue] = {
        ERROR_TYPE: type(normalized or error).__name__,
        MOLTO_ERROR_PHASE: phase,
    }
    if normalized is not None:
        status = normalized.response.status_code if normalized.response is not None else None
    elif isinstance(error, (openai.APIStatusError, anthropic.APIStatusError)):
        status = error.status_code
    else:
        status = None
    # Only a real upstream error response: missing credentials carry a synthetic 401
    # on the outward exception alone, and a mid-stream failure can arrive on HTTP 200.
    if _is_int(status) and status >= 400:
        attributes[HTTP_RESPONSE_STATUS_CODE] = status
    return attributes


def exception_event_attributes(error: BaseException) -> dict[str, AttributeValue]:
    """Return content-free overrides for the attributes `Span.record_exception` builds.

    Exception messages can quote request content (a validation error's input, or an
    upstream error echoing a field), and the default stack trace repeats the message
    of every chained cause. Report the class name as the message and a stack trace of
    frames only: frame lines show code locations, never runtime values.
    """
    lines: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        lines.append(f"{type(current).__module__}.{type(current).__qualname__}\n")
        lines.extend(traceback.format_tb(current.__traceback__))
        current = current.__cause__
    return {"exception.message": type(error).__name__, "exception.stacktrace": "".join(lines)}


_RESPONSES_CONTENT_DELTAS = frozenset(
    {
        "response.output_text.delta",
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
        "response.custom_tool_call_input.delta",
        "response.mcp_call_arguments.delta",
        "response.refusal.delta",
    }
)
# The SDK raises only on a separate `error` event, so a `response.failed` that arrives
# without one passes through native streaming and must still be recorded.
_RESPONSES_TERMINAL_EVENTS = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)
# Messages content-block delta types that carry content, mapped to their text field.
_MESSAGES_CONTENT_DELTAS = {
    "text_delta": "text",
    "thinking_delta": "thinking",
    "input_json_delta": "partial_json",
}


async def observe[EventT](
    events: AsyncIterator[EventT], record: Callable[[EventT], None]
) -> AsyncIterator[EventT]:
    """Yield `events` unchanged, passing each to `record` first."""
    # Closing this wrapper must close the source and release its HTTP response.
    async with aclosing(events) as source:
        async for event in source:
            record_safely(record, event)
            yield event


def record_safely[ValueT](record: Callable[[ValueT], None], value: ValueT) -> None:
    """Pass `value` to `record`; a recording failure loses attributes, never the call.

    Native output is unvalidated, so a provider can send a value no extractor expects.
    """
    try:
        record(value)
    except Exception as error:
        # Only the class: loguru's traceback diagnosis prints local values, which can
        # hold request or completion content.
        logger.debug("Molto could not record span attributes: {}", type(error).__name__)


class SpanRecorder:
    """Record end attributes and the first-token event on a coordinator's span.

    The coordinator starts and ends the span; the recorder only adds what the
    upstream output reveals, so the span carries it before it ends. It holds the
    state that spans several stream events.
    """

    def __init__(self, span: Span, *, streaming: bool) -> None:
        self._span = span
        self._streaming = streaming
        self._first_token_recorded = False
        self._chat_identity: tuple[Any, Any] = (None, None)
        self._chat_finish_reasons: dict[int, str] = {}
        self._messages_usage = MessagesUsage()

    def set_known(self, attributes: MaybeAttributes) -> None:
        """Set the attributes whose value is known, skipping `None` values."""
        self._span.set_attributes(
            {key: value for key, value in attributes.items() if value is not None}
        )

    def record_normalized(self, event: ModelResponseStreamEvent) -> None:
        self.set_known(normalized_event_attributes(event))
        if is_normalized_content_delta(event):
            self._record_first_token()

    def record_responses_event(self, event: OpenAIResponsesStreamEvent) -> None:
        if getattr(event, "type", None) in _RESPONSES_TERMINAL_EVENTS:
            self.set_known(responses_response_attributes(getattr(event, "response", None)))
        elif is_responses_content_delta(event):
            self._record_first_token()

    def record_chat_completions_chunk(self, chunk: OpenAIChatCompletionsStreamChunk) -> None:
        # Every chunk repeats the response identity, so set it only when it changes.
        identity = (getattr(chunk, "id", None), getattr(chunk, "model", None))
        if identity != self._chat_identity:
            self._chat_identity = identity
            self.set_known({GEN_AI_RESPONSE_ID: identity[0], GEN_AI_RESPONSE_MODEL: identity[1]})
        if is_chat_completions_content_delta(chunk):
            self._record_first_token()
        # Each choice reports its finish reason in its own chunk.
        reasons = self._chat_finish_reasons | chat_completions_finish_reasons(
            getattr(chunk, "choices", None)
        )
        if reasons != self._chat_finish_reasons:
            self._chat_finish_reasons = reasons
            self.set_known({GEN_AI_RESPONSE_FINISH_REASONS: [reasons[i] for i in sorted(reasons)]})
        # Present only when the caller set `stream_options.include_usage`.
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.set_known(chat_completions_usage_attributes(usage))

    def record_messages_event(self, event: AnthropicMessagesStreamEvent) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            message = getattr(event, "message", None)
            self.set_known(
                {
                    GEN_AI_RESPONSE_ID: getattr(message, "id", None),
                    GEN_AI_RESPONSE_MODEL: getattr(message, "model", None),
                }
            )
            self._record_messages_usage(getattr(message, "usage", None))
        elif event_type == "message_delta":
            stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
            if stop_reason:
                self.set_known({GEN_AI_RESPONSE_FINISH_REASONS: [stop_reason]})
            self._record_messages_usage(getattr(event, "usage", None))
        elif is_messages_content_delta(event):
            self._record_first_token()

    def _record_messages_usage(self, usage: AnthropicUsage | MessageDeltaUsage | None) -> None:
        self._messages_usage.update(usage)
        self.set_known(self._messages_usage.attributes())

    def _record_first_token(self) -> None:
        # Time to first token matters only to a streaming caller.
        if self._streaming and not self._first_token_recorded:
            self._first_token_recorded = True
            self._span.add_event(FIRST_TOKEN_EVENT)
