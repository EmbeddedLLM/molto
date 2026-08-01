from collections.abc import AsyncIterator
from typing import Any

from httpx import AsyncClient
from openai import AsyncOpenAI
from openai._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from molto.configs import ENV_CONFIG
from molto.providers.openai.output_translation import (
    chat_response_to_output_events,
    chat_stream_to_output_events,
    responses_response_to_output_events,
    responses_stream_to_output_events,
)
from molto.providers.openai.request_translation import build_openai_request
from molto.providers.openai.types import OpenAIEndpoint
from molto.providers.runtime import (
    execute_sdk_call,
    map_sdk_error,
    owns_sdk_client,
)
from molto.types.ir.events import OutputEvent
from molto.types.ir.portable import InputIR
from molto.utils.exceptions import MoltoException

_OPENAI_SDK_EXCEPTIONS = {
    "authentication": AuthenticationError,
    "permission_denied": PermissionDeniedError,
    "not_found": NotFoundError,
    "bad_request": BadRequestError,
    "rate_limit": RateLimitError,
    "timeout": APITimeoutError,
    "connection": APIConnectionError,
    "api_status": APIStatusError,
}

_CONTEXT_ERROR_CODES = frozenset({"context_length_exceeded"})


def map_openai_error(exc: Exception) -> MoltoException:
    """Map official OpenAI SDK failures to stable Molto exceptions."""

    return map_sdk_error(
        exc,
        sdk_exceptions=_OPENAI_SDK_EXCEPTIONS,
        context_error_codes=_CONTEXT_ERROR_CODES,
    )


class OpenAIProvider:
    """Execute requests through the official OpenAI SDK.

    Capability validation happens in `validate_provider_request` before this
    provider is constructed; `run` assumes a validated portable request.
    """

    def __init__(
        self,
        endpoint: OpenAIEndpoint,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        http_client: AsyncClient | None = None,
    ) -> None:
        self.endpoint = endpoint
        # Caller-supplied transports stay caller-owned: Molto never closes them.
        self._owns_http_client = owns_sdk_client(sdk_client=None, http_client=http_client)
        self._client = AsyncOpenAI(
            api_key=api_key or ENV_CONFIG.openai_api_key_plain or "missing-api-key",
            base_url=base_url,
            http_client=http_client,
            max_retries=max_retries,
        )

    async def close(self) -> None:
        if self._owns_http_client:
            await self._client.close()

    def run(
        self,
        request: InputIR,
    ) -> AsyncIterator[OutputEvent]:
        return self._run(request)

    async def _run(
        self,
        request: InputIR,
    ) -> AsyncIterator[OutputEvent]:
        if self.endpoint == "responses":
            stream_translate = responses_stream_to_output_events
            response_translate = responses_response_to_output_events
        else:
            stream_translate = chat_stream_to_output_events
            response_translate = chat_response_to_output_events
        async for event in execute_sdk_call(
            request=request,
            build=lambda portable: build_openai_request(portable, self.endpoint),
            create=self._create,
            stream_translate=stream_translate,
            response_translate=response_translate,
            map_error=map_openai_error,
            close=self.close,
        ):
            yield event

    async def _create(self, native_request: dict[str, Any]) -> Any:
        if self.endpoint == "responses":
            return await self._client.responses.create(**native_request)
        return await self._client.chat.completions.create(**native_request)
