from typing import Any, Literal

from httpx import AsyncClient
from pydantic import BaseModel


class Molto:
    def __init__(
        self,
        http_client: AsyncClient | None = None,
    ) -> None:
        self._http_client = http_client

    async def close(self) -> None:
        await self._http_client.aclose()

    async def responses(
        self,
        body: BaseModel | dict[str, Any],
        endpoint: Literal["responses", "chat/completions"] = "responses",
        provider: Literal["openai", "vllm"] = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
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
        pass
