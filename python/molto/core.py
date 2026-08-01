from collections.abc import AsyncIterator, Callable
from typing import Any

from httpx import AsyncClient

from molto.configs import ENV_CONFIG
from molto.providers import (
    ANTHROPIC_OPENAI_CHAT_CAPABILITIES,
    OpenAIProvider,
    create_provider_after_validation,
    descriptor_for,
)
from molto.types.ir.portable import collect_output_ir
from molto.types.openai import OpenAIRequest
from molto.types.openai.chat_completions import (
    OpenAIChatCompletionChunk,
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
    OpenAIResponsesResponse,
    OpenAIResponsesStreamEvent,
)
from molto.types.openai.responses import (
    from_output_event_stream as responses_from_output_event_stream,
)
from molto.types.openai.responses import (
    from_output_ir as responses_from_output_ir,
)
from molto.types.openresponses import (
    OpenResponsesRequest,
    OpenResponsesResponse,
    OpenResponsesStreamEvent,
)
from molto.types.openresponses import (
    from_output_event_stream as openresponses_from_output_event_stream,
)
from molto.types.openresponses import from_output_ir as openresponses_from_output_ir
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

    async def responses(
        self,
        body: OpenAIRequest | OpenResponsesRequest,
        api_key: str | None = None,
    ) -> (
        OpenAIResponsesResponse
        | AsyncIterator[OpenAIResponsesStreamEvent]
        | OpenResponsesResponse
        | AsyncIterator[OpenResponsesStreamEvent]
    ):
        """Return Responses output; the request model selects the native endpoint.

        OpenAI Chat requests execute through Chat Completions and are translated
        semantically into Responses output. OpenAI Responses requests retain exact
        same-format parity. Anthropic returns the portable OpenResponses contract.
        """

        if not isinstance(body, (OpenAIRequest, OpenResponsesRequest)):
            raise BadInputError("OpenAI requests must use OpenAIRequest.")
        ir = body.to_ir()
        self._check_model(ir.model, allowed_prefixes=("openai/", "vllm/", "anthropic/"))

        if isinstance(body, OpenAIRequest) and body.native_endpoint == "chat_completions":
            payload = body.payload
            if ir.model.startswith("anthropic/"):
                raise BadInputError("Anthropic does not accept the OpenAI Chat request contract.")
            if (
                payload.audio is not None
                or (payload.modalities is not None and "audio" in payload.modalities)
                or any(item.audio_id is not None for item in ir.items if hasattr(item, "audio_id"))
            ):
                raise BadInputError("Responses output cannot represent generated Chat audio.")
            provider = create_provider_after_validation(
                descriptor_for(ir.model, "chat_completions"),
                ir,
                lambda: OpenAIProvider(
                    "chat_completions",
                    http_client=self._http_client,
                    api_key=api_key,
                    base_url=self._base_url,
                    max_retries=self._max_retries,
                ),
            )
            return await _execute(
                provider,
                ir,
                from_stream=responses_from_chat_output_event_stream,
                from_output=responses_from_chat_output_ir,
            )

        if ir.model.startswith("anthropic/"):
            provider = create_provider_after_validation(
                ANTHROPIC_OPENAI_CHAT_CAPABILITIES,
                ir,
                lambda: OpenAIProvider(
                    "chat_completions",
                    api_key=api_key or ENV_CONFIG.anthropic_api_key_plain,
                    base_url=self._base_url or "https://api.anthropic.com/v1/",
                    max_retries=self._max_retries,
                    http_client=self._http_client,
                ),
            )
            return await _execute(
                provider,
                ir,
                from_stream=openresponses_from_output_event_stream,
                from_output=openresponses_from_output_ir,
            )

        provider = create_provider_after_validation(
            descriptor_for(ir.model, "responses"),
            ir,
            lambda: OpenAIProvider(
                "responses",
                http_client=self._http_client,
                api_key=api_key,
                base_url=self._base_url,
                max_retries=self._max_retries,
            ),
        )
        return await _execute(
            provider,
            ir,
            from_stream=responses_from_output_event_stream,
            from_output=responses_from_output_ir,
        )

    async def chat_completions(
        self,
        body: OpenAIRequest,
        api_key: str | None = None,
    ) -> OpenAIChatCompletionResponse | AsyncIterator[OpenAIChatCompletionChunk]:
        """Return Chat output; the request model selects the native endpoint."""

        if not isinstance(body, OpenAIRequest):
            raise BadInputError("Chat Completions requests must use OpenAIRequest.")
        ir = body.to_ir()
        self._check_model(ir.model, allowed_prefixes=("openai/", "vllm/"))

        if body.native_endpoint == "responses":
            payload = body.payload
            unsupported_tools = [
                tool for tool in payload.tools or () if getattr(tool, "type", None) != "function"
            ]
            if unsupported_tools:
                raise BadInputError(
                    "Chat Completions output supports Responses function tools only."
                )
            if (
                payload.reasoning is not None
                or payload.conversation is not None
                or payload.previous_response_id is not None
                or payload.store is not None
            ):
                raise BadInputError(
                    "Chat Completions output cannot represent Responses reasoning or state."
                )
            provider = create_provider_after_validation(
                descriptor_for(ir.model, "responses"),
                ir,
                lambda: OpenAIProvider(
                    "responses",
                    http_client=self._http_client,
                    api_key=api_key,
                    base_url=self._base_url,
                    max_retries=self._max_retries,
                ),
            )
            return await _execute(
                provider,
                ir,
                from_stream=chat_from_responses_output_event_stream,
                from_output=chat_from_responses_output_ir,
            )

        provider = create_provider_after_validation(
            descriptor_for(ir.model, "chat_completions"),
            ir,
            lambda: OpenAIProvider(
                "chat_completions",
                http_client=self._http_client,
                api_key=api_key,
                base_url=self._base_url,
                max_retries=self._max_retries,
            ),
        )
        return await _execute(
            provider,
            ir,
            from_stream=chat_from_output_event_stream,
            from_output=chat_from_output_ir,
        )
