from collections.abc import AsyncIterator, Mapping
from typing import Any, Self

from httpx import AsyncClient, Headers
from pydantic import BaseModel

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.configs import ENV_CONFIG
from molto.providers import (
    DEFAULT_PROVIDER_BY_ENDPOINT,
    PROVIDER_CAPABILITIES,
    PROVIDER_KEY_NAMES,
    Endpoint,
    Provider,
    ProviderCapabilities,
)
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesResponse,
    AnthropicMessagesStreamEvent,
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
)
from molto.upstream.registry import CLIENTS

VLLM_DUMMY_API_KEY = "EMPTY"

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
        self.response_headers = Headers(response_headers)

    def __aiter__(self) -> AsyncIterator[EventT]:
        return self

    async def __anext__(self) -> EventT:
        if self._closed:
            raise StopAsyncIteration
        if self._events:
            return self._events.pop()
        return await anext(self._stream)

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
        if close is not None:
            await close()


async def _prefetch_stream[EventT](
    stream: AsyncIterator[EventT], response_headers: Headers
) -> MoltoStream[EventT]:
    """Start a stream now so setup failures are raised by the coordinator await."""
    try:
        first_event = await anext(stream)
    except StopAsyncIteration:
        return MoltoStream([], stream, response_headers)
    return MoltoStream([first_event], stream, response_headers)


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
        if api_key is None and provider in PROVIDER_KEY_NAMES:
            return getattr(ENV_CONFIG, PROVIDER_KEY_NAMES[provider].lower()).get_secret_value()
        if api_key is None and provider == "vllm":
            return VLLM_DUMMY_API_KEY
        return api_key

    @staticmethod
    def _resolve_provider(endpoint: Endpoint, provider: Provider | None) -> Provider:
        # Validate the endpoint before any adapter normalization so an unknown
        # endpoint cannot surface as an unrelated translation error.
        if endpoint not in DEFAULT_PROVIDER_BY_ENDPOINT:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}.")
        if provider == "gemini" and endpoint != "chat/completions":
            raise ValueError("Gemini supports only the Chat Completions upstream.")
        if provider is not None and provider not in PROVIDER_CAPABILITIES:
            raise ValueError(f"Unsupported provider: {provider!r}.")
        return provider or DEFAULT_PROVIDER_BY_ENDPOINT[endpoint]

    @staticmethod
    def _resolve_provider_options(
        provider: Provider, endpoint: Endpoint, base_url: str | None
    ) -> tuple[str | None, ProviderCapabilities]:
        if provider == "togetherai" and endpoint != "chat/completions" and not base_url:
            raise ValueError(
                "Together AI requires an explicit base_url for an alternative upstream endpoint."
            )
        if provider == "bedrock" and not base_url:
            raise ValueError("Bedrock requires an explicit regional base_url and model/profile.")
        capabilities = CLIENTS[provider, endpoint].resolve_capabilities(
            PROVIDER_CAPABILITIES[provider]
        )
        return base_url, capabilities

    async def responses(
        self,
        body: BaseModel | dict[str, Any],
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
            body (BaseModel | dict[str, Any]): Request payload.
            endpoint: Upstream wire protocol. Defaults to "responses".
            provider: Implementation serving the protocol and its capability profile.
                Defaults to the canonical provider for the endpoint.
            api_key (str | None, optional): API key for the provider. Defaults to None (infer from provider).
            base_url (str | None, optional): HTTP destination. Defaults to the client's native URL.
            extra_body: Additional fields merged into the upstream request body.
            extra_headers: Additional upstream headers. Client credentials and transport
                headers are removed before the provider request.
        """
        request = OpenAIResponsesCreateRequest.model_validate(body)
        resolved_provider = self._resolve_provider(endpoint, provider)
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        base_url, capabilities = self._resolve_provider_options(
            resolved_provider, endpoint, base_url
        )
        upstream_headers = _upstream_headers(extra_headers)
        upstream = CLIENTS[resolved_provider, endpoint](
            self._http_client,
            api_key=resolved_api_key,
            base_url=base_url,
            capabilities=capabilities,
            extra_body=extra_body,
            extra_headers=upstream_headers,
        )

        if endpoint == "responses":
            adapter = OpenAIResponsesAdapter(capabilities)
            if request.stream is True:
                return await _prefetch_stream(
                    adapter.project_native_events(upstream.native_events(request)),
                    upstream.response_headers,
                )
            return await adapter.project_native_response(upstream.native_response(request))

        adapter = OpenAIResponsesAdapter(capabilities)
        normalized_request = adapter.normalize_request(request)
        projected_events = adapter.project_events(
            upstream.events(normalized_request.model_copy(update={"stream": True})),
            normalized_request,
        )
        if request.stream is True:
            return await _prefetch_stream(projected_events, upstream.response_headers)
        return await adapter.project_response(projected_events, normalized_request)

    async def chat_completions(
        self,
        body: BaseModel | dict[str, Any],
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
        request = OpenAIChatCompletionsCreateRequest.model_validate(body)
        resolved_provider = self._resolve_provider(endpoint, provider)
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        base_url, capabilities = self._resolve_provider_options(
            resolved_provider, endpoint, base_url
        )
        upstream_headers = _upstream_headers(extra_headers)
        upstream = CLIENTS[resolved_provider, endpoint](
            self._http_client,
            api_key=resolved_api_key,
            base_url=base_url,
            capabilities=capabilities,
            extra_body=extra_body,
            extra_headers=upstream_headers,
        )
        if endpoint == "chat/completions":
            adapter = OpenAIChatCompletionsAdapter(capabilities)
            if request.stream is True:
                return await _prefetch_stream(
                    adapter.project_native_events(upstream.native_events(request)),
                    upstream.response_headers,
                )
            return await adapter.project_native_response(upstream.native_response(request))

        adapter = OpenAIChatCompletionsAdapter(capabilities)
        normalized_request = adapter.normalize_request(request)
        projected_events = adapter.project_events(
            upstream.events(normalized_request.model_copy(update={"stream": True})),
            normalized_request,
        )
        if request.stream is True:
            return await _prefetch_stream(projected_events, upstream.response_headers)
        return await adapter.project_response(projected_events, normalized_request)

    async def messages(
        self,
        body: BaseModel | dict[str, Any],
        endpoint: Endpoint = "messages",
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> MoltoStream[AnthropicMessagesStreamEvent] | AnthropicMessagesResponse:
        """Consume an Anthropic Messages request using any supported upstream protocol.

        `endpoint` selects the wire protocol, `provider` selects the serving
        implementation and its capabilities, and `base_url` selects only the
        HTTP destination.
        """
        request = AnthropicMessagesCreateRequest.model_validate(body)
        resolved_provider = self._resolve_provider(endpoint, provider)
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        base_url, capabilities = self._resolve_provider_options(
            resolved_provider, endpoint, base_url
        )
        upstream_headers = _upstream_headers(extra_headers)
        upstream = CLIENTS[resolved_provider, endpoint](
            self._http_client,
            api_key=resolved_api_key,
            base_url=base_url,
            capabilities=capabilities,
            extra_body=extra_body,
            extra_headers=upstream_headers,
        )
        if endpoint == "messages":
            adapter = AnthropicMessagesAdapter(capabilities)
            if request.stream is True:
                return await _prefetch_stream(
                    adapter.project_native_events(upstream.native_events(request)),
                    upstream.response_headers,
                )
            return await adapter.project_native_response(upstream.native_response(request))

        adapter = AnthropicMessagesAdapter(capabilities)
        normalized_request = adapter.normalize_request(request)
        projected_events = adapter.project_events(
            upstream.events(normalized_request.model_copy(update={"stream": True})),
            normalized_request,
        )
        if request.stream is True:
            return await _prefetch_stream(projected_events, upstream.response_headers)
        return await adapter.project_response(projected_events, normalized_request)
