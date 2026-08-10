from collections.abc import AsyncIterator
from typing import Any, Literal

from httpx import AsyncClient
from pydantic import BaseModel

from molto.adapters.openai import OpenAIResponsesAdapter
from molto.configs import ENV_CONFIG
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


class Molto:
    def __init__(
        self,
        http_client: AsyncClient | None = None,
    ) -> None:
        self._http_client = http_client or AsyncClient()

    async def close(self) -> None:
        await self._http_client.aclose()

    async def responses(
        self,
        body: BaseModel | dict[str, Any],
        endpoint: Literal["responses", "chat/completions"] = "responses",
        provider: Literal["openai", "vllm"] = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[BaseModel]:
        """
        Consume OpenAI Responses request and return Responses output.
        The upstream format can be either OpenAI Responses or OpenAI Chat Completions.

        Args:
            body (BaseModel | dict[str, Any]): Request payload.
            endpoint (Literal["responses", "chat/completions"], optional): Upstream endpoint / format to use.
                Defaults to "responses".
            provider (Literal["openai", "vllm"], optional): Provider serving the requested model.
                Defaults to "openai".
            api_key (str | None, optional): API key for the provider. Defaults to None (infer from provider).
            base_url (str | None, optional): Base URL for the provider API. Defaults to None (infer from provider).
        """
        request = OpenAIResponsesCreateRequest.model_validate(body)
        if request.stream is not True:
            raise NotImplementedError("Only streaming Responses requests are currently supported.")

        resolved_api_key = api_key
        if resolved_api_key is None and provider == "openai":
            resolved_api_key = ENV_CONFIG.openai_api_key_plain

        if endpoint == "responses":
            upstream = OpenAIResponsesClient(
                self._http_client,
                api_key=resolved_api_key,
                base_url=base_url,
            )
            return upstream.native_events(request)

        adapter = OpenAIResponsesAdapter()
        normalized_request = adapter.normalize_request(request)
        upstream = OpenAIChatCompletionsClient(
            self._http_client,
            api_key=resolved_api_key,
            base_url=base_url,
        )
        return adapter.project_events(upstream.events(normalized_request), normalized_request)
