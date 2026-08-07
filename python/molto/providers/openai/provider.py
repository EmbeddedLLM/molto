import json
from collections.abc import AsyncIterator
from typing import Any

from httpx import AsyncClient, Response
from openai import APIError, AsyncOpenAI
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


async def _sse_payload_stream(response: Response) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded JSON payloads from one streaming SSE response.

    Raw frames preserve the provider's exact key order and value types, which
    SDK-parsed events cannot (model dumps reorder keys and coerce numbers).
    """

    data_lines: list[str] = []
    try:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                data = line[5:]
                data_lines.append(data[1:] if data.startswith(" ") else data)
                continue
            # Only `data:` lines carry payloads; `event:`/`id:`/comment lines
            # are dropped, and a blank line dispatches the buffered event.
            if line or not data_lines:
                continue
            data = "\n".join(data_lines)
            data_lines = []
            if data == "[DONE]":
                return
            yield _stream_payload(data, response)
    finally:
        await response.aclose()


def _stream_payload(data: str, response: Response) -> dict[str, Any]:
    """Decode one SSE data payload, raising provider stream errors.

    Mirrors the official SDK stream contract: a frame whose payload carries an
    `error` object aborts iteration with `APIError`, which `map_openai_error`
    translates like any other SDK failure.
    """

    payload = json.loads(data)
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if not isinstance(message, str) or not message:
                message = "An error occurred during streaming"
            raise APIError(message=message, request=response.request, body=error)
    return payload


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
        api_key: str,
        base_url: str | None = None,
        max_retries: int = 2,
        http_client: AsyncClient | None = None,
    ) -> None:
        self.endpoint = endpoint
        # Caller-supplied transports stay caller-owned: Molto never closes them.
        self._owns_http_client = owns_sdk_client(sdk_client=None, http_client=http_client)
        self._client = AsyncOpenAI(
            api_key=api_key,
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
        resource = (
            self._client.responses
            if self.endpoint == "responses"
            else self._client.chat.completions
        )
        if native_request.get("stream"):
            # Stream raw SSE frames: SDK-parsed events lose the provider's exact
            # payload (key order, integer timestamps) that same-format replay
            # needs, so preserve frames from the raw HTTP response instead.
            raw = await resource.with_raw_response.create(**native_request)
            return _sse_payload_stream(raw.http_response)
        return await resource.create(**native_request)
