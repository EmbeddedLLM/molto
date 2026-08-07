from collections.abc import AsyncIterator, Callable, Mapping
from functools import partial
from typing import Any

from httpx import AsyncClient

from molto.providers import (
    OpenAIProvider,
    create_provider_after_validation,
    provider_contract_for,
)
from molto.providers.openai.types import OpenAIEndpoint
from molto.types.ir.portable import InputIR, collect_output_ir
from molto.types.openai._common import OpenAIProviderName
from molto.types.openai.chat_completions import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from molto.types.openai.chat_completions import (
    from_output_event_stream as chat_from_output_event_stream,
)
from molto.types.openai.chat_completions import from_output_ir as chat_from_output_ir
from molto.types.openai.chat_completions.to_ir import prepare_chat_request
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
from molto.types.openai.responses.to_ir import prepare_responses_request
from molto.utils.exceptions import BadInputError


def default_endpoint_for(
    provider: OpenAIProviderName, public_endpoint: OpenAIEndpoint
) -> OpenAIEndpoint:
    """Resolve the provider-aware execution endpoint when the caller omits it."""

    return "chat_completions" if provider == "anthropic" else public_endpoint


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

    ``http_client`` and the default ``max_retries`` are configured on the
    gateway. Each call supplies its provider credential and can override
    ``base_url`` and ``max_retries``. ``strict`` controls capability
    enforcement: strict mode rejects unsupported request features, non-strict
    mode drops and ignores them.

    Resource contract: caller-injected HTTP clients stay caller-owned and are
    never closed by Molto. Each provider closes its own SDK-created client
    after the response completes or the stream is exhausted, so `close` is a
    documented no-op kept for API symmetry.
    """

    def __init__(
        self,
        *,
        strict: bool = False,
        max_retries: int = 2,
        http_client: AsyncClient | None = None,
    ) -> None:
        self._strict = strict
        self._max_retries = max_retries
        self._http_client = http_client

    async def close(self) -> None:
        """Release Molto-owned resources (see the class resource contract)."""

    def _check_base_url(
        self,
        model: str,
        *,
        base_url: str | None,
    ) -> None:
        """Check base_url combinations before provider construction."""

        if model.startswith("vllm/") and base_url is None:
            raise BadInputError("vLLM execution requires an explicit base_url.")

    def _create_provider(
        self,
        ir: InputIR,
        *,
        endpoint: OpenAIEndpoint,
        api_key: str,
        base_url: str | None,
        max_retries: int,
    ) -> OpenAIProvider:
        """Validate one provider/endpoint route before constructing its SDK client."""

        self._check_base_url(
            ir.model,
            base_url=base_url,
        )
        if endpoint not in ("chat_completions", "responses"):
            raise BadInputError(f"Unsupported endpoint: {endpoint}")

        provider_base_url = base_url
        if ir.model.startswith("anthropic/"):
            provider_base_url = base_url or "https://api.anthropic.com/v1/"

        return create_provider_after_validation(
            provider_contract_for(ir.model, endpoint),
            ir,
            lambda: OpenAIProvider(
                endpoint,
                http_client=self._http_client,
                api_key=api_key,
                base_url=provider_base_url,
                max_retries=max_retries,
            ),
            strict=self._strict,
        )

    async def responses(
        self,
        body: OpenAIResponsesRequest | Mapping[str, Any],
        api_key: str,
        *,
        endpoint: OpenAIEndpoint | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
    ) -> OpenAIResponsesResponse | AsyncIterator[OpenAIResponsesStreamEvent]:
        """Return Responses output using the selected OpenAI-compatible endpoint."""

        body = OpenAIResponsesRequest.from_body(body, model)
        resolved_provider = body.provider
        endpoint = endpoint or default_endpoint_for(resolved_provider, "responses")
        ir = prepare_responses_request(body, endpoint)
        provider = self._create_provider(
            ir,
            endpoint=endpoint,
            api_key=api_key,
            base_url=base_url,
            max_retries=self._max_retries if max_retries is None else max_retries,
        )

        if endpoint == "chat_completions":
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
        body: OpenAIChatCompletionRequest | Mapping[str, Any],
        api_key: str,
        *,
        endpoint: OpenAIEndpoint | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
    ) -> OpenAIChatCompletionResponse | AsyncIterator[OpenAIChatCompletionChunk]:
        """Return Chat output using the selected OpenAI-compatible endpoint."""

        body = OpenAIChatCompletionRequest.from_body(body, model)
        resolved_provider = body.provider
        endpoint = endpoint or default_endpoint_for(resolved_provider, "chat_completions")
        ir = prepare_chat_request(body, endpoint)
        provider = self._create_provider(
            ir,
            endpoint=endpoint,
            api_key=api_key,
            base_url=base_url,
            max_retries=self._max_retries if max_retries is None else max_retries,
        )

        if endpoint == "responses":
            # The cross-endpoint Chat composer intentionally omits Responses reasoning items.
            return await _execute(
                provider,
                ir,
                from_stream=partial(
                    chat_from_responses_output_event_stream,
                    include_usage=ir.include_usage,
                ),
                from_output=chat_from_responses_output_ir,
            )
        return await _execute(
            provider,
            ir,
            from_stream=chat_from_output_event_stream,
            from_output=chat_from_output_ir,
        )
