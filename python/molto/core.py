from collections.abc import AsyncIterator
from typing import Any

from httpx import AsyncClient
from pydantic import BaseModel

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.configs import ENV_CONFIG
from molto.providers import (
    DEFAULT_PROVIDER_BY_ENDPOINT,
    PROVIDER_CAPABILITIES,
    Endpoint,
    Provider,
)
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient

VLLM_DUMMY_API_KEY = "EMPTY"


async def _prefetch_stream[EventT](stream: AsyncIterator[EventT]) -> AsyncIterator[EventT]:
    """Start a stream now so setup failures are raised by the coordinator await."""
    try:
        first_event = await anext(stream)
    except StopAsyncIteration:
        return stream

    # return _PrefetchedStream(first_event, stream)
    async def agen():
        yield first_event
        async for event in stream:
            yield event

    return agen()


class Molto:
    def __init__(
        self,
        http_client: AsyncClient | None = None,
    ) -> None:
        self._http_client = http_client or AsyncClient()

    async def close(self) -> None:
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
        body: BaseModel | dict[str, Any],
        endpoint: Endpoint = "responses",
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[BaseModel]:
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
        """
        request = OpenAIResponsesCreateRequest.model_validate(body)
        if request.stream is not True:
            raise NotImplementedError("Only streaming Responses requests are currently supported.")

        resolved_provider = self._resolve_provider(endpoint, provider)
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        capabilities = PROVIDER_CAPABILITIES[resolved_provider]

        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
            return await _prefetch_stream(upstream.native_events(request))

        adapter = OpenAIResponsesAdapter(capabilities)
        normalized_request = adapter.normalize_request(request)
        if endpoint == "chat/completions":
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
        elif endpoint == "messages":
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}.")
        return await _prefetch_stream(
            adapter.project_events(upstream.events(normalized_request), normalized_request)
        )

    async def chat_completions(
        self,
        body: BaseModel | dict[str, Any],
        endpoint: Endpoint = "chat/completions",
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[BaseModel]:
        """Consume a Chat Completions request using any supported upstream protocol.

        ``endpoint`` selects the wire protocol, ``provider`` selects the serving
        implementation and its capabilities, and ``base_url`` selects only the
        HTTP destination.
        """
        request = OpenAIChatCompletionsCreateRequest.model_validate(body)
        if request.stream is not True:
            raise NotImplementedError(
                "Only streaming Chat Completions requests are currently supported."
            )

        resolved_provider = self._resolve_provider(endpoint, provider)
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        capabilities = PROVIDER_CAPABILITIES[resolved_provider]
        if endpoint == "chat/completions":
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
            return await _prefetch_stream(upstream.native_events(request))

        adapter = OpenAIChatCompletionsAdapter(capabilities)
        normalized_request = adapter.normalize_request(request)
        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
        elif endpoint == "messages":
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}.")
        return await _prefetch_stream(
            adapter.project_events(upstream.events(normalized_request), normalized_request)
        )

    async def messages(
        self,
        body: BaseModel | dict[str, Any],
        endpoint: Endpoint = "messages",
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[BaseModel]:
        """Consume an Anthropic Messages request using any supported upstream protocol.

        ``endpoint`` selects the wire protocol, ``provider`` selects the serving
        implementation and its capabilities, and ``base_url`` selects only the
        HTTP destination.
        """
        request = AnthropicMessagesCreateRequest.model_validate(body)
        if request.stream is not True:
            raise NotImplementedError("Only streaming Messages requests are currently supported.")

        resolved_provider = self._resolve_provider(endpoint, provider)
        resolved_api_key = self._resolve_api_key(resolved_provider, api_key)
        capabilities = PROVIDER_CAPABILITIES[resolved_provider]
        if endpoint == "messages":
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
            return await _prefetch_stream(upstream.native_events(request))

        adapter = AnthropicMessagesAdapter(capabilities)
        normalized_request = adapter.normalize_request(request)
        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
        elif endpoint == "chat/completions":
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
                capabilities=capabilities,
            )
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}.")
        return await _prefetch_stream(
            adapter.project_events(upstream.events(normalized_request), normalized_request)
        )
