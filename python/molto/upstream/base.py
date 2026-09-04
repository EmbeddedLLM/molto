from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import NoReturn

from httpx import AsyncClient
from openai import AsyncOpenAI, omit
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
    ) -> None:
        self._http_client = http_client
        self._api_key = api_key
        self._base_url = base_url
        self._capabilities = capabilities

    def _make_sdk_client(self) -> AsyncOpenAI:
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
            async for event in self._events(request):
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
