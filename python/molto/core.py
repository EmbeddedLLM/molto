from collections.abc import AsyncIterator

from httpx import AsyncClient

from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.types.openai.responses import (
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    _OpenAIStreamEvent,
)


class Molto:
    def __init__(
        self,
        http_client: AsyncClient,
    ) -> None:
        self._http_client = http_client

    async def close(self) -> None:
        await self._http_client.aclose()

    async def chat(
        self,
        body: OpenAIResponsesRequest | OpenAIChatCompletionRequest,
    ) -> OpenAIResponsesResponse | AsyncIterator[_OpenAIStreamEvent]:
        pass
        # TODO: Transform `body` into an intermediate representation, then dispatch to the appropriate provider(s)
