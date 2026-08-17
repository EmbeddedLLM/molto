from collections.abc import AsyncIterator
from typing import Any, Literal

from httpx import AsyncClient
from pydantic import BaseModel

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.configs import ENV_CONFIG
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient

VLLM_DUMMY_API_KEY = "EMPTY"


class Molto:
    def __init__(
        self,
        http_client: AsyncClient | None = None,
    ) -> None:
        self._http_client = http_client or AsyncClient()

    async def close(self) -> None:
        await self._http_client.aclose()

    @staticmethod
    def _resolve_api_key(provider: str, api_key: str | None) -> str | None:
        if api_key is None and provider == "openai":
            return ENV_CONFIG.openai_api_key_plain
        if api_key is None and provider == "anthropic":
            return ENV_CONFIG.anthropic_api_key_plain
        if api_key is None and provider == "vllm":
            return VLLM_DUMMY_API_KEY
        return api_key

    async def responses(
        self,
        body: BaseModel | dict[str, Any],
        endpoint: Literal["responses", "chat/completions", "messages"] = "responses",
        provider: Literal["anthropic", "openai", "vllm"] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[BaseModel]:
        """
        Consume OpenAI Responses request and return Responses output.
        The upstream format can be OpenAI Responses, OpenAI Chat Completions,
        or Anthropic Messages.

        Args:
            body (BaseModel | dict[str, Any]): Request payload.
            endpoint: Upstream endpoint and protocol to use.
                Defaults to "responses".
            provider: Provider serving the requested model.
                Defaults to the provider implied by the endpoint.
            api_key (str | None, optional): API key for the provider. Defaults to None (infer from provider).
            base_url (str | None, optional): Base URL for the provider API. Defaults to None (infer from provider).
        """
        request = OpenAIResponsesCreateRequest.model_validate(body)
        if request.stream is not True:
            raise NotImplementedError("Only streaming Responses requests are currently supported.")

        resolved_api_key = self._resolve_api_key(
            provider or ("anthropic" if endpoint == "messages" else "openai"),
            api_key,
        )

        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
            return upstream.native_events(request)

        adapter = OpenAIResponsesAdapter()
        normalized_request = adapter.normalize_request(request)
        if endpoint == "chat/completions":
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
        else:
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
        return adapter.project_events(upstream.events(normalized_request), normalized_request)

    async def chat_completions(
        self,
        body: BaseModel | dict[str, Any],
        endpoint: Literal["responses", "chat/completions", "messages"] = "chat/completions",
        provider: Literal["anthropic", "openai", "vllm"] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[BaseModel]:
        """Consume a Chat Completions request using either OpenAI endpoint upstream."""
        request = OpenAIChatCompletionsCreateRequest.model_validate(body)
        if request.stream is not True:
            raise NotImplementedError(
                "Only streaming Chat Completions requests are currently supported."
            )

        resolved_api_key = self._resolve_api_key(
            provider or ("anthropic" if endpoint == "messages" else "openai"),
            api_key,
        )
        if endpoint == "chat/completions":
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
            return upstream.native_events(request)

        adapter = OpenAIChatCompletionsAdapter()
        normalized_request = adapter.normalize_request(request)
        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
        else:
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
        return adapter.project_events(upstream.events(normalized_request), normalized_request)

    async def messages(
        self,
        body: BaseModel | dict[str, Any],
        endpoint: Literal["responses", "chat/completions", "messages"] = "messages",
        provider: Literal["anthropic", "openai", "vllm"] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[BaseModel]:
        """Consume an Anthropic Messages request using any supported endpoint upstream."""
        request = AnthropicMessagesCreateRequest.model_validate(body)
        if request.stream is not True:
            raise NotImplementedError("Only streaming Messages requests are currently supported.")

        resolved_api_key = self._resolve_api_key(
            provider or ("anthropic" if endpoint == "messages" else "openai"),
            api_key,
        )
        if endpoint == "messages":
            upstream = AnthropicMessagesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
            return upstream.native_events(request)

        adapter = AnthropicMessagesAdapter()
        normalized_request = adapter.normalize_request(request)
        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
        else:
            upstream = OpenAIChatCompletionsClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
        return adapter.project_events(upstream.events(normalized_request), normalized_request)
