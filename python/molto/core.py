from collections.abc import AsyncIterator

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
        prefer_openai_responses: bool = False,
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

        if ir.model.startswith("openai/"):
            model_provider = OpenAIResponsesProvider()
        elif ir.model.startswith("anthropic/"):
            model_provider = AnthropicMessagesProvider()
        elif ir.model.startswith("vllm/"):
            if prefer_openai_responses:
                model_provider = OpenAIResponsesProvider()
            else:
                model_provider = OpenAIChatCompletionsProvider()
        else:
            raise ValueError(f"Unsupported model: {ir.model}")

        agent_settings = await model_provider.parse_agent_settings(ir, self._http_client)
        run_settings = await model_provider.parse_run_settings(ir)
        runner = Runner(agent_settings, run_settings)
        if ir.stream:
            return request_provider.compose_outputs(runner.run_stream())
        else:
            return await request_provider.get_output(runner.run_stream())
