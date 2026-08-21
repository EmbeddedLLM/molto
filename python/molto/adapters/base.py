from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable
from typing import NoReturn

from pydantic import BaseModel

from molto.providers import ProviderCapabilities
from molto.types.normalized import (
    ModelResponseStreamEvent,
    NormalizedAPIError,
    NormalizedRequest,
    Thinking,
)


class BaseAdapter[NativeRequestT: BaseModel, NativeEventT](ABC):
    """Translate the downstream protocol to and from normalized IR."""

    def __init__(
        self,
        capabilities: ProviderCapabilities,
    ) -> None:
        self._capabilities = capabilities
        self.native_request: NativeRequestT | None = None

    def normalize_request(self, request: NativeRequestT) -> NormalizedRequest:
        """Translate a downstream request into the normalized representation."""
        self.native_request = request
        return self._normalize_request(request)

    @abstractmethod
    def _normalize_request(self, request: NativeRequestT) -> NormalizedRequest:
        """Implement downstream request normalization."""
        raise NotImplementedError

    def project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[NativeEventT]:
        """Translate normalized response events into downstream events."""
        if self.native_request is None:
            raise RuntimeError("normalize_request() must be called before project_events().")
        return self._project_events(self._project_api_errors(events), request)

    async def open_projected_events(
        self,
        events: Awaitable[AsyncIterator[ModelResponseStreamEvent]],
        request: NormalizedRequest,
    ) -> AsyncIterator[NativeEventT]:
        """Open and project a translated stream without consuming an event."""
        try:
            opened_events = await events
        except NormalizedAPIError as error:
            self._raise_projected_api_error(error)
        return self.project_events(opened_events, request)

    async def _project_api_errors(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        try:
            async for event in events:
                yield event
        except NormalizedAPIError as error:
            self._raise_projected_api_error(error)

    def _raise_projected_api_error(self, error: NormalizedAPIError) -> NoReturn:
        projected_error = self._project_api_error(error)
        if error.__cause__ is None:
            raise projected_error from error
        raise projected_error from error.__cause__

    @abstractmethod
    def _project_api_error(self, error: NormalizedAPIError) -> Exception:
        """Construct the downstream SDK exception for a normalized API failure."""
        raise NotImplementedError

    @abstractmethod
    def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[NativeEventT]:
        """Implement downstream event projection."""
        raise NotImplementedError

    def _classify_thinking(self, text: str, *, signature: str | None = None) -> Thinking:
        """Place a protocol's ambiguous reasoning text in the provider's representation.

        Protocols with a single reasoning field do not say whether their text is
        raw reasoning or a summary; the provider capability profile classifies it.
        """
        return Thinking(
            content=text if self._capabilities.thinking == "raw" else "",
            summary=text if self._capabilities.thinking == "summary" else "",
            signature=(signature or None) if self._capabilities.thinking == "summary" else None,
        )
