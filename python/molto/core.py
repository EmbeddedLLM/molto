from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from time import time_ns
from typing import Any, Self

from httpx import AsyncClient, Headers
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from pydantic import BaseModel

from molto.adapters.anthropic import AnthropicMessagesAdapter, CitationPolicy
from molto.adapters.base import BaseAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.configs import ENV_CONFIG
from molto.providers import (
    DEFAULT_PROVIDER_BY_ENDPOINT,
    PROVIDER_CAPABILITIES,
    Endpoint,
    Provider,
)
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesResponse,
    AnthropicMessagesStreamEvent,
)
from molto.types.exceptions import UnsupportedInput
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsResponse,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStreamEvent,
    has_anthropic_search_metadata,
)
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.base import BaseClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from molto.utils.io import request_to_dict
from molto.utils.telemetry import (
    GEN_AI_REQUEST_STREAM,
    MOLTO_STREAM_OUTCOME,
    ErrorPhase,
    ResolvedRoute,
    SpanRecorder,
    StreamOutcome,
    chat_completions_response_attributes,
    error_attributes,
    exception_event_attributes,
    get_tracer,
    messages_response_attributes,
    observe,
    responses_response_attributes,
    span_name,
    start_attributes,
)

VLLM_DUMMY_API_KEY = "EMPTY"

# Each endpoint's client, used to resolve the effective destination for span attributes
# exactly as the client will before transport.
_UPSTREAM_CLIENTS: dict[Endpoint, type[BaseClient]] = {
    "responses": OpenAIResponsesClient,
    "chat/completions": OpenAIChatCompletionsClient,
    "messages": AnthropicMessagesClient,
}

_CREDENTIAL_HEADERS = frozenset(
    {
        "anthropic-api-key",
        "api-key",
        "authorization",
        "cookie",
        "openai-organization",
        "openai-project",
        "proxy-authorization",
        "x-api-key",
        "x-oai-attestation",
        "x-openai-api-key",
    }
)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_INGRESS_TRANSPORT_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "content-encoding",
        "content-length",
        "content-type",
        "expect",
        "forwarded",
        "host",
        "proxy-connection",
        "via",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
        "x-real-ip",
    }
)


def _upstream_headers(extra_headers: Mapping[str, str] | None) -> dict[str, str] | None:
    """Remove client credentials and connection-specific headers before forwarding."""
    if extra_headers is None:
        return None
    headers = Headers(extra_headers)
    connection_headers = {
        name.strip().lower()
        for value in headers.get_list("connection")
        for name in value.split(",")
        if name.strip()
    }
    excluded = (
        _CREDENTIAL_HEADERS | _HOP_BY_HOP_HEADERS | _INGRESS_TRANSPORT_HEADERS | connection_headers
    )
    return {name: value for name, value in headers.items() if name.lower() not in excluded}


def _failure_outcome(error: BaseException) -> StreamOutcome:
    # Cancellation, and any other exit that is not an `Exception` (such as
    # `KeyboardInterrupt`), is the caller stopping the call, not the call failing: it
    # leaves the status UNSET and reports `cancelled` (docs/observability.md).
    return "error" if isinstance(error, Exception) else "cancelled"


def _finish_span(
    span: Span,
    error: BaseException | None = None,
    phase: ErrorPhase = "setup",
    outcome: StreamOutcome | None = None,
    end_time: int | None = None,
) -> None:
    if outcome is not None:
        span.set_attribute(MOLTO_STREAM_OUTCOME, outcome)
    if isinstance(error, Exception):
        span.set_status(Status(StatusCode.ERROR))
        span.set_attributes(error_attributes(error, phase))
        span.record_exception(error, attributes=exception_event_attributes(error))
    span.end(end_time=end_time)


class MoltoStream[EventT](AsyncIterator[EventT]):
    """Replay prefetched events and manage the source stream lifecycle."""

    def __init__(
        self,
        events: list[EventT],
        stream: AsyncIterator[EventT],
        response_headers: Headers,
    ) -> None:
        self._events = events
        self._stream = stream
        self._closed = False
        self._span: Span | None = None
        # When the caller last received an event; an abandoned stream's span ends here.
        # The prefetch has just received the first event.
        self._last_event_ns = time_ns()
        self.response_headers = Headers(response_headers)

    def __aiter__(self) -> AsyncIterator[EventT]:
        return self

    async def __anext__(self) -> EventT:
        if self._closed:
            raise StopAsyncIteration
        if self._events:
            self._last_event_ns = time_ns()
            return self._events.pop()
        try:
            event = await anext(self._stream)
            self._last_event_ns = time_ns()
            return event
        except StopAsyncIteration:
            self._end_span()
            raise
        except BaseException as error:
            self._end_span(error)
            raise

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._events.clear()
        close = getattr(self._stream, "aclose", None)
        try:
            if close is not None:
                await close()
        except BaseException as error:
            self._end_span(error)
            raise
        # A stream already exhausted or failed has ended its span, so this records
        # `closed` only for a stream the caller stopped before its end.
        self._end_span(outcome="closed")

    def __del__(self) -> None:
        # A dropped, unclosed stream must still be visible (docs/observability.md,
        # "Streaming lifetime"). `aclose()` cannot be awaited here, so the upstream
        # response stays open; the span ends at the last event so its duration is not
        # inflated by when garbage collection happens to run.
        if getattr(self, "_span", None) is None:
            return
        try:
            self._end_span(outcome="abandoned", end_time=self._last_event_ns)
        except Exception:
            # Exceptions in `__del__` are only printed, and at interpreter shutdown the
            # SDK or exporter may already be gone; telemetry must not add noise here.
            pass

    def _end_span(
        self,
        error: BaseException | None = None,
        outcome: StreamOutcome = "completed",
        *,
        end_time: int | None = None,
    ) -> None:
        # Exhaustion, an error, `aclose()`, and garbage collection can all reach here;
        # end the span once. A failure decides the outcome over the given one.
        if self._span is not None:
            span, self._span = self._span, None
            final = _failure_outcome(error) if error is not None else outcome
            _finish_span(span, error, "stream", final, end_time)


async def _run_native[EventT, ResponseT: BaseModel](
    adapter: BaseAdapter[Any, EventT, ResponseT],
    upstream: BaseClient[Any, EventT, ResponseT],
    body: dict[str, Any],
    *,
    record_event: Callable[[EventT], None],
    record_response: Callable[[ResponseT], None],
) -> MoltoStream[EventT] | ResponseT:
    """Pass a same-protocol request through, recording its native output on the span.

    Streaming follows the coordinators' rule: only `body["stream"] is True` streams.
    """
    if body.get("stream") is True:
        return await _prefetch_stream(
            adapter.project_native_events(observe(upstream.native_events(body), record_event)),
            upstream.response_headers,
        )
    response = await adapter.project_native_response(upstream.native_response(body))
    record_response(response)
    return response


async def _prefetch_stream[EventT](
    stream: AsyncIterator[EventT], response_headers: Headers
) -> MoltoStream[EventT]:
    """Start a stream now so setup failures are raised by the coordinator await."""
    try:
        first_event = await anext(stream)
    except StopAsyncIteration:
        return MoltoStream([], stream, response_headers)
    return MoltoStream([first_event], stream, response_headers)


async def _traced[ResultT](
    protocol: Endpoint,
    endpoint: Endpoint,
    provider: Provider,
    base_url: str | None,
    body: Mapping[str, Any] | BaseModel,
    call: Callable[[SpanRecorder], Awaitable[ResultT]],
) -> ResultT:
    """Run one coordinator call inside its span.

    The span starts in the caller's context, so it parents to the caller's active
    span. It is current only while `call` sends the upstream request (the whole call
    when non-streaming, the prefetch when streaming), so instrumented `httpx` spans
    nest under it. `call` records end attributes through its `SpanRecorder`; a
    returned stream ends the span itself.
    """
    route = ResolvedRoute(
        protocol, endpoint, provider, _UPSTREAM_CLIENTS[endpoint].resolve_base_url(base_url)
    )
    attributes = start_attributes(route, body)
    span = get_tracer().start_span(
        span_name(attributes), kind=SpanKind.CLIENT, attributes=attributes
    )
    streaming = attributes.get(GEN_AI_REQUEST_STREAM) is True
    recorder = SpanRecorder(span, streaming=streaming)
    try:
        # `_finish_span` sets the status instead of `use_span`, whose ERROR status
        # description would carry the exception message.
        with trace.use_span(span, record_exception=False, set_status_on_exception=False):
            result = await call(recorder)
    except BaseException as error:
        # A streaming call that ends before its first event still reports how it ended.
        _finish_span(span, error, outcome=_failure_outcome(error) if streaming else None)
        raise
    if isinstance(result, MoltoStream):
        # The stream outlives this `await`, so it ends the span itself.
        result._span = span
    else:
        _finish_span(span)
    return result


class Molto:
    def __init__(
        self,
        http_client: AsyncClient | None = None,
    ) -> None:
        self._http_client = http_client or AsyncClient()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http_client.aclose()

    @staticmethod
    def _resolve_api_key(provider: Provider, api_key: str | None) -> str | None:
        if api_key is None and provider == "openai":
            return ENV_CONFIG.openai_api_key_plain
        if api_key is None and provider == "anthropic":
            return ENV_CONFIG.anthropic_api_key_plain
        if api_key is None and provider == "vllm":
            return VLLM_DUMMY_API_KEY
        return api_key

    @staticmethod
    def _resolve_provider(endpoint: Endpoint, provider: Provider | None) -> Provider:
        # Validate the endpoint before any adapter normalization so an unknown
        # endpoint cannot surface as an unrelated translation error.
        if endpoint not in DEFAULT_PROVIDER_BY_ENDPOINT:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}.")
        return provider or DEFAULT_PROVIDER_BY_ENDPOINT[endpoint]

    async def responses(
        self,
        body: OpenAIResponsesCreateRequest | dict[str, Any],
        endpoint: Endpoint = "responses",
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> MoltoStream[OpenAIResponsesStreamEvent] | OpenAIResponsesResponse:
        """
        Consume OpenAI Responses request and return Responses output.
        The upstream format can be OpenAI Responses, OpenAI Chat Completions,
        or Anthropic Messages.

        Args:
            body: Request dictionary or OpenAIResponsesCreateRequest.
            endpoint: Upstream wire protocol. Defaults to "responses".
            provider: Implementation serving the protocol and its capability profile.
                Defaults to the canonical provider for the endpoint.
            api_key (str | None, optional): API key for the provider. Defaults to None (infer from provider).
            base_url (str | None, optional): HTTP destination. Defaults to the client's native URL.
            extra_body: Additional fields merged into the upstream request body.
            extra_headers: Additional upstream headers. Client credentials and transport
                headers are removed before the provider request.
        """
        resolved_provider = self._resolve_provider(endpoint, provider)
        if not isinstance(body, (dict, OpenAIResponsesCreateRequest)):
            raise TypeError("body must be a dictionary or OpenAIResponsesCreateRequest.")
        return await _traced(
            "responses",
            endpoint,
            resolved_provider,
            base_url,
            body,
            lambda recorder: self._run_responses(
                body,
                endpoint=endpoint,
                resolved_provider=resolved_provider,
                api_key=api_key,
                base_url=base_url,
                extra_body=extra_body,
                extra_headers=extra_headers,
                recorder=recorder,
            ),
        )

    async def _run_responses(
        self,
        body: OpenAIResponsesCreateRequest | dict[str, Any],
        *,
        endpoint: Endpoint,
        resolved_provider: Provider,
        api_key: str | None,
        base_url: str | None,
        extra_body: Mapping[str, Any] | None,
        extra_headers: Mapping[str, str] | None,
        recorder: SpanRecorder,
    ) -> MoltoStream[OpenAIResponsesStreamEvent] | OpenAIResponsesResponse:
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        capabilities = PROVIDER_CAPABILITIES[resolved_provider]
        upstream_headers = _upstream_headers(extra_headers)

        if endpoint == "responses":
            body = request_to_dict(body)
            if has_anthropic_search_metadata(body.get("input")):
                raise UnsupportedInput(
                    "Anthropic search extra_content requires endpoint='messages' "
                    "and provider='anthropic'."
                )
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
            adapter = OpenAIResponsesAdapter(capabilities)
            return await _run_native(
                adapter,
                upstream,
                body,
                record_event=recorder.record_responses_event,
                record_response=lambda response: recorder.set(
                    responses_response_attributes(response)
                ),
            )

        request = OpenAIResponsesCreateRequest.model_validate(body)
        preserve_search_metadata = endpoint == "messages" and resolved_provider == "anthropic"
        if request.has_anthropic_search_metadata() and not preserve_search_metadata:
            raise UnsupportedInput(
                "Anthropic search extra_content requires endpoint='messages' "
                "and provider='anthropic'."
            )
        adapter = OpenAIResponsesAdapter(
            capabilities, preserve_anthropic_search_metadata=preserve_search_metadata
        )
        normalized_request = adapter.normalize_request(request)
        if endpoint == "chat/completions":
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
        elif endpoint == "messages":
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}.")
        projected_events = adapter.project_events(
            observe(
                upstream.events(normalized_request.model_copy(update={"stream": True})),
                recorder.record_normalized,
            ),
            normalized_request,
        )
        if request.stream is True:
            return await _prefetch_stream(projected_events, upstream.response_headers)
        return await adapter.project_response(projected_events, normalized_request)

    async def chat_completions(
        self,
        body: OpenAIChatCompletionsCreateRequest | dict[str, Any],
        endpoint: Endpoint = "chat/completions",
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> MoltoStream[OpenAIChatCompletionsStreamChunk] | OpenAIChatCompletionsResponse:
        """Consume a Chat Completions request using any supported upstream protocol.

        `endpoint` selects the wire protocol, `provider` selects the serving
        implementation and its capabilities, and `base_url` selects only the
        HTTP destination.
        """
        resolved_provider = self._resolve_provider(endpoint, provider)
        if not isinstance(body, (dict, OpenAIChatCompletionsCreateRequest)):
            raise TypeError("body must be a dictionary or OpenAIChatCompletionsCreateRequest.")
        return await _traced(
            "chat/completions",
            endpoint,
            resolved_provider,
            base_url,
            body,
            lambda recorder: self._run_chat_completions(
                body,
                endpoint=endpoint,
                resolved_provider=resolved_provider,
                api_key=api_key,
                base_url=base_url,
                extra_body=extra_body,
                extra_headers=extra_headers,
                recorder=recorder,
            ),
        )

    async def _run_chat_completions(
        self,
        body: OpenAIChatCompletionsCreateRequest | dict[str, Any],
        *,
        endpoint: Endpoint,
        resolved_provider: Provider,
        api_key: str | None,
        base_url: str | None,
        extra_body: Mapping[str, Any] | None,
        extra_headers: Mapping[str, str] | None,
        recorder: SpanRecorder,
    ) -> MoltoStream[OpenAIChatCompletionsStreamChunk] | OpenAIChatCompletionsResponse:
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        capabilities = PROVIDER_CAPABILITIES[resolved_provider]
        upstream_headers = _upstream_headers(extra_headers)
        if endpoint == "chat/completions":
            body = request_to_dict(body)
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
            adapter = OpenAIChatCompletionsAdapter(capabilities)
            return await _run_native(
                adapter,
                upstream,
                body,
                record_event=recorder.record_chat_completions_chunk,
                record_response=lambda response: recorder.set(
                    chat_completions_response_attributes(response)
                ),
            )

        request = OpenAIChatCompletionsCreateRequest.model_validate(body)
        adapter = OpenAIChatCompletionsAdapter(capabilities)
        normalized_request = adapter.normalize_request(request)
        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
        elif endpoint == "messages":
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}.")
        projected_events = adapter.project_events(
            observe(
                upstream.events(normalized_request.model_copy(update={"stream": True})),
                recorder.record_normalized,
            ),
            normalized_request,
        )
        if request.stream is True:
            return await _prefetch_stream(projected_events, upstream.response_headers)
        return await adapter.project_response(projected_events, normalized_request)

    async def messages(
        self,
        body: AnthropicMessagesCreateRequest | dict[str, Any],
        endpoint: Endpoint = "messages",
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        citation_policy: CitationPolicy = "drop",
    ) -> MoltoStream[AnthropicMessagesStreamEvent] | AnthropicMessagesResponse:
        """Consume an Anthropic Messages request using any supported upstream protocol.

        `endpoint` selects the wire protocol, `provider` selects the serving
        implementation and its capabilities, and `base_url` selects only the
        HTTP destination. With Responses upstream, `citation_policy="buffer"`
        withholds text until each part completes to preserve citation boundaries;
        `"auto"` buffers only when normalized tools or history signal possible
        citations, otherwise streaming text without citations. The default `"drop"`
        always streams text without citations. Transport extensions do not inform
        `"auto"`. The policy has no effect with Chat upstream or native Messages.
        """
        resolved_provider = self._resolve_provider(endpoint, provider)
        if not isinstance(body, (dict, AnthropicMessagesCreateRequest)):
            raise TypeError("body must be a dictionary or AnthropicMessagesCreateRequest.")
        return await _traced(
            "messages",
            endpoint,
            resolved_provider,
            base_url,
            body,
            lambda recorder: self._run_messages(
                body,
                endpoint=endpoint,
                resolved_provider=resolved_provider,
                api_key=api_key,
                base_url=base_url,
                extra_body=extra_body,
                extra_headers=extra_headers,
                citation_policy=citation_policy,
                recorder=recorder,
            ),
        )

    async def _run_messages(
        self,
        body: AnthropicMessagesCreateRequest | dict[str, Any],
        *,
        endpoint: Endpoint,
        resolved_provider: Provider,
        api_key: str | None,
        base_url: str | None,
        extra_body: Mapping[str, Any] | None,
        extra_headers: Mapping[str, str] | None,
        citation_policy: CitationPolicy,
        recorder: SpanRecorder,
    ) -> MoltoStream[AnthropicMessagesStreamEvent] | AnthropicMessagesResponse:
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        capabilities = PROVIDER_CAPABILITIES[resolved_provider]
        upstream_headers = _upstream_headers(extra_headers)
        if endpoint == "messages":
            body = request_to_dict(body)
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
            adapter = AnthropicMessagesAdapter(capabilities, citation_policy=citation_policy)
            return await _run_native(
                adapter,
                upstream,
                body,
                record_event=recorder.record_messages_event,
                record_response=lambda response: recorder.set(
                    messages_response_attributes(response)
                ),
            )

        request = AnthropicMessagesCreateRequest.model_validate(body)
        # Chat normalization drops annotations, so buffering cannot recover citations.
        adapter = AnthropicMessagesAdapter(
            capabilities,
            citation_policy="drop" if endpoint == "chat/completions" else citation_policy,
        )
        normalized_request = adapter.normalize_request(request)
        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
        elif endpoint == "chat/completions":
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
                extra_body=extra_body,
                extra_headers=upstream_headers,
            )
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}.")
        projected_events = adapter.project_events(
            observe(
                upstream.events(normalized_request.model_copy(update={"stream": True})),
                recorder.record_normalized,
            ),
            normalized_request,
        )
        if request.stream is True:
            return await _prefetch_stream(projected_events, upstream.response_headers)
        return await adapter.project_response(projected_events, normalized_request)
