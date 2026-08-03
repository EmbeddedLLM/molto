from collections.abc import AsyncIterator, Callable
from typing import Any

from httpx import AsyncClient

from molto.configs import ENV_CONFIG
from molto.providers import (
    OpenAIProvider,
    create_provider_after_validation,
    provider_contract_for,
)
from molto.providers.openai.types import OpenAIEndpoint
from molto.types.ir.portable import InputIR, collect_output_ir
from molto.types.openai.chat_completions import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from molto.types.openai.chat_completions import (
    from_output_event_stream as chat_from_output_event_stream,
)
from molto.types.openai.chat_completions import from_output_ir as chat_from_output_ir
from molto.types.openai.cross_endpoint import (
    chat_from_responses_output_event_stream,
    chat_from_responses_output_ir,
    responses_from_chat_output_event_stream,
    responses_from_chat_output_ir,
)
from molto.types.openai.responses import (
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStreamEvent,
)
from molto.types.openai.responses import (
    from_output_event_stream as responses_from_output_event_stream,
)
from molto.types.openai.responses import (
    from_output_ir as responses_from_output_ir,
)
from molto.utils.exceptions import BadInputError


async def _execute(
    provider: Any,
    ir: Any,
    *,
    from_stream: Callable[[AsyncIterator[Any]], Any],
    from_output: Callable[[Any], Any],
) -> Any:
    """Run one provider and compose the public output.

    Streams start the provider before returning so immediate authentication and
    request-validation failures retain the public call-time behavior of the
    native SDK.
    """

    output_events = provider.run(ir)
    if ir.stream:
        first = await anext(output_events)

        async def chained() -> AsyncIterator[Any]:
            yield first
            async for event in output_events:
                yield event

        return from_stream(chained())
    output = await collect_output_ir(output_events)
    return from_output(output)


class Molto:
    """Public gateway entry point.

    Resource contract: caller-injected HTTP clients stay caller-owned and are
    never closed by Molto. Each provider closes its own SDK-created client
    after the response completes or the stream is exhausted, so `close` is a
    documented no-op kept for API symmetry.
    """

    def __init__(
        self,
        http_client: AsyncClient | None = None,
        *,
        base_url: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self._http_client = http_client
        self._base_url = base_url
        self._max_retries = max_retries

    async def close(self) -> None:
        """Release Molto-owned resources (see the class resource contract)."""

    def _check_model(self, model: str, *, allowed_prefixes: tuple[str, ...]) -> None:
        """Reject unsupported model prefixes and vLLM without an explicit base_url."""

        if not any(model.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(f"Unsupported model: {model}")
        if model.startswith("vllm/") and self._base_url is None:
            raise BadInputError("vLLM execution requires an explicit base_url.")

    def _create_provider(
        self,
        ir: InputIR,
        *,
        upstream: OpenAIEndpoint,
        api_key: str | None,
    ) -> OpenAIProvider:
        """Validate one provider/upstream route before constructing its SDK client."""

        self._check_model(ir.model, allowed_prefixes=("openai/", "vllm/", "anthropic/"))
        if upstream not in ("chat_completions", "responses"):
            raise BadInputError(f"Unsupported upstream endpoint: {upstream}")

        provider_api_key = api_key
        provider_base_url = self._base_url
        if ir.model.startswith("anthropic/"):
            provider_api_key = api_key or ENV_CONFIG.anthropic_api_key_plain
            provider_base_url = self._base_url or "https://api.anthropic.com/v1/"

        return create_provider_after_validation(
            provider_contract_for(ir.model, upstream),
            ir,
            lambda: OpenAIProvider(
                upstream,
                http_client=self._http_client,
                api_key=provider_api_key,
                base_url=provider_base_url,
                max_retries=self._max_retries,
            ),
        )

    async def responses(
        self,
        body: OpenAIResponsesRequest,
        api_key: str | None = None,
        *,
        upstream: OpenAIEndpoint = "responses",
    ) -> OpenAIResponsesResponse | AsyncIterator[OpenAIResponsesStreamEvent]:
        """Return Responses output using the selected OpenAI-compatible upstream."""

        if not isinstance(body, OpenAIResponsesRequest):
            raise BadInputError("Responses requests must use OpenAIResponsesRequest.")
        ir = body.to_ir()
        provider = self._create_provider(ir, upstream=upstream, api_key=api_key)

        if upstream == "chat_completions":
            return await _execute(
                provider,
                ir,
                from_stream=responses_from_chat_output_event_stream,
                from_output=responses_from_chat_output_ir,
            )
        return await _execute(
            provider,
            ir,
            from_stream=responses_from_output_event_stream,
            from_output=responses_from_output_ir,
        )

    async def chat_completions(
        self,
        body: OpenAIChatCompletionRequest,
        api_key: str | None = None,
        *,
        upstream: OpenAIEndpoint = "chat_completions",
    ) -> OpenAIChatCompletionResponse | AsyncIterator[OpenAIChatCompletionChunk]:
        """Return Chat output using the selected OpenAI-compatible upstream."""

        if not isinstance(body, OpenAIChatCompletionRequest):
            raise BadInputError("Chat Completions requests must use OpenAIChatCompletionRequest.")
        ir = body.to_ir()
        provider = self._create_provider(ir, upstream=upstream, api_key=api_key)

        if upstream == "responses":
            # The cross-endpoint Chat composer intentionally omits Responses reasoning items.
            return await _execute(
                provider,
                ir,
                from_stream=chat_from_responses_output_event_stream,
                from_output=chat_from_responses_output_ir,
            )
        return await _execute(
            provider,
            ir,
            from_stream=chat_from_output_event_stream,
            from_output=chat_from_output_ir,
        )
