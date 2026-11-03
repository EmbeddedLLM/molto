from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from typing import Any, NoReturn

from httpx import URL, AsyncClient, Headers, Request
from openai import AsyncOpenAI, OpenAIError, omit
from pydantic import BaseModel

from molto.providers import ProviderCapabilities
from molto.types.normalized import (
    ModelResponseStreamEvent,
    NormalizedRequest,
    Thinking,
    ThinkingDelta,
)


class BaseClient[
    NativeRequestT: BaseModel,
    NativeEventT,
    NativeResponseT: BaseModel,
](ABC):
    DEFAULT_BASE_URL: str

    def __init__(
        self,
        http_client: AsyncClient,
        *,
        capabilities: ProviderCapabilities,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        required_api_key: str | None = None,
        bearer_auth: bool = False,
    ) -> None:
        self._http_client = http_client
        self._api_key = api_key
        self._base_url = base_url
        self._capabilities = capabilities
        self._extra_body = extra_body
        self._extra_headers = extra_headers
        self._required_api_key = required_api_key
        self._bearer_auth = bearer_auth
        self.response_headers = Headers()

    def _capture_response_headers(self, headers: Headers) -> None:
        self.response_headers.clear()
        self.response_headers.update(headers)

    def _make_sdk_client(self) -> AsyncOpenAI:
        if self._required_api_key and not self._api_key:
            raise OpenAIError(
                f"Missing credentials. Set {self._required_api_key} or pass api_key."
            )
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url or self.DEFAULT_BASE_URL,
            default_headers={
                # Preserve Molto's explicit credential boundary instead of merging
                # OpenAI account settings from the SDK's environment variables.
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Organization": omit,
                "OpenAI-Project": omit,
            },
            http_client=self._http_client,
        )

    def _make_authentication_request(self, path: str) -> Request:
        # Only for missing SDK credentials, which fail before an HTTP request is sent.
        # The downstream SDK AuthenticationError requires a response with a request;
        # this unsent request supplies its intended URL for our synthetic 401 response.
        # HTTP and stream errors retain their real requests instead. Match SDK path
        # appending here, including custom base prefixes and trailing slashes.
        base_url = URL(self._base_url or self.DEFAULT_BASE_URL)
        raw_path = base_url.raw_path
        if not raw_path.endswith(b"/"):
            raw_path += b"/"
        return Request("POST", base_url.copy_with(raw_path=raw_path + path.encode()))

    def _classify_thinking(self, text: str, *, signature: str | None = None) -> Thinking:
        """Place a protocol's ambiguous reasoning text in the provider's representation.

        Chat Completions `reasoning_content` and Messages `thinking_delta` do not
        say whether their text is raw reasoning or a summary; the provider
        capability profile classifies it. Responses labels its reasoning events
        explicitly and never uses this helper.
        """
        return Thinking(
            content=text if self._capabilities.thinking == "raw" else "",
            summary=text if self._capabilities.thinking == "summary" else "",
            signature=(signature or None) if self._capabilities.thinking == "summary" else None,
        )

    def _classify_thinking_delta(self, text: str) -> ThinkingDelta:
        """Classify one ambiguous reasoning text chunk as a normalized delta."""
        return ThinkingDelta(
            content_delta=text if self._capabilities.thinking == "raw" else None,
            summary_delta=text if self._capabilities.thinking == "summary" else None,
        )

    @abstractmethod
    def native_events(self, request: NativeRequestT) -> AsyncIterator[NativeEventT]:
        """Stream events using the provider's native protocol.

        Implementations must delegate failures to `_handle_error()` so the native
        path normalizes known errors to the same IR causes as `events()`.
        """
        raise NotImplementedError

    @abstractmethod
    async def native_response(self, request: NativeRequestT) -> NativeResponseT:
        """Return a native response, delegating failures to `_handle_error()`."""
        raise NotImplementedError

    async def events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream provider events translated to normalized events with error handling."""
        try:
            async with aclosing(self._events(request)) as events:
                async for event in events:
                    yield event
        except Exception as error:
            self._handle_error(error)

    @abstractmethod
    def _events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream provider events translated to normalized events."""
        raise NotImplementedError

    @abstractmethod
    def _handle_error(self, error: Exception) -> NoReturn:
        """Translate or re-raise an error raised while streaming normalized events."""
        raise NotImplementedError
