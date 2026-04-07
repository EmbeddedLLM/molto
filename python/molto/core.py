from collections.abc import AsyncIterator

from httpx import AsyncClient

from molto.providers import (
    AnthropicMessagesProvider,
    OpenAIChatCompletionsProvider,
    OpenAIResponsesProvider,
)
from molto.runner import Runner
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
        prefer_openai_responses: bool = False,
    ) -> OpenAIResponsesResponse | AsyncIterator[_OpenAIStreamEvent]:
        pass
        # TODO: Transform `body` into an intermediate representation, then dispatch to the appropriate provider(s)
        ir = body.to_ir()
        if ir.model.startswith("openai/"):
            provider = OpenAIResponsesProvider
        elif ir.model.startswith("anthropic/"):
            provider = AnthropicMessagesProvider
        elif ir.model.startswith("vllm/"):
            if prefer_openai_responses:
                provider = OpenAIResponsesProvider
            else:
                provider = OpenAIChatCompletionsProvider
        else:
            raise ValueError(f"Unsupported model: {ir.model}")
        agent_settings = await provider.parse_agent_settings(ir)
        run_settings = await provider.parse_run_settings(ir)
        runner = Runner(agent_settings, run_settings)
        if ir.stream:
            return await provider.compose_outputs(runner.run_stream())
        else:
            return await provider.get_output(runner.run_stream())
