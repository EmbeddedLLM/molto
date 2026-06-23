from collections.abc import AsyncIterator
from typing import Literal

from httpx import AsyncClient

from molto.providers import (
    AnthropicMessagesProvider,
    OpenAIChatCompletionsProvider,
    OpenAIResponsesProvider,
)
from molto.runner import Runner
from molto.types.openai.chat_completions import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from molto.types.openai.responses import (
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    _OpenAIStreamEvent,
)


class Molto:
    def __init__(
        self,
        http_client: AsyncClient | None = None,
    ) -> None:
        self._http_client = http_client

    async def close(self) -> None:
        await self._http_client.aclose()

    async def chat(
        self,
        body: OpenAIResponsesRequest | OpenAIChatCompletionRequest,
        api_key: str | None = None,
        protocol: Literal["match_request", "responses", "chat_completions"] = "match_request",
    ) -> (
        OpenAIResponsesResponse
        | AsyncIterator[_OpenAIStreamEvent]
        | OpenAIChatCompletionResponse
        | AsyncIterator[OpenAIChatCompletionChunk]
    ):
        """Orchestrate the pipeline."""
        ir = body.to_ir()
        if isinstance(body, OpenAIChatCompletionRequest):
            request_provider = OpenAIChatCompletionsProvider(ir=ir)
        elif isinstance(body, OpenAIResponsesRequest):
            request_provider = OpenAIResponsesProvider(ir=ir)
        else:
            raise ValueError(f"Unsupported request type: {type(body)}")

        def openai_provider_for_endpoint():
            if protocol == "responses":
                return OpenAIResponsesProvider()
            if protocol == "chat_completions":
                return OpenAIChatCompletionsProvider()
            if protocol == "match_request":
                if isinstance(body, OpenAIChatCompletionRequest):
                    return OpenAIChatCompletionsProvider()
                return OpenAIResponsesProvider()
            raise ValueError(f"Unsupported protocol: {protocol}")

        if ir.model.startswith("openai/") or ir.model.startswith("vllm/"):
            model_provider = openai_provider_for_endpoint()
        elif ir.model.startswith("anthropic/"):
            # TODO: implement responses for Anthropic
            model_provider = AnthropicMessagesProvider()
        else:
            raise ValueError(f"Unsupported model: {ir.model}")

        agent_settings = await model_provider.parse_agent_settings(
            ir, self._http_client, api_key=api_key
        )
        run_settings = await model_provider.parse_run_settings(ir)
        runner = Runner(agent_settings, run_settings)
        if ir.stream:
            return request_provider.compose_outputs(runner.run_stream())
        else:
            return await request_provider.get_output(runner.run_stream())
