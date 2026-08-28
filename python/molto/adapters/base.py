from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import NoReturn

from pydantic import BaseModel, ValidationError

from molto.providers import ProviderCapabilities
from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    ModelResponseStreamEvent,
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
        try:
            return self._normalize_request(request)
        except UnsupportedInput:
            raise
        except ValidationError as error:
            raise UnsupportedInput(str(error)) from error

    @classmethod
    def reject_unsupported_request_fields(
        cls,
        request: BaseModel,
        supported_fields: frozenset[str],
        *,
        protocol: str,
    ) -> None:
        """Reject populated native fields that would be lost during normalization."""
        unsupported_fields = sorted(
            field_name
            for field_name in request.model_fields_set - supported_fields
            if getattr(request, field_name, None) is not None
        )
        if unsupported_fields:
            raise UnsupportedInput(
                f"Unsupported {protocol} request fields for cross-protocol translation: "
                f"{unsupported_fields}."
            )

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
        # Keep this method synchronous so it returns an AsyncIterator directly and callers
        # can write `async for event in adapter.project_events(...)` without awaiting it.
        return self._stream_with_error_handling(self._project_events(events, request))

    def project_native_events(
        self,
        events: AsyncIterator[NativeEventT],
    ) -> AsyncIterator[NativeEventT]:
        """Pass same-protocol events through, translating known IR failures.

        Native passthrough skips request and event translation, but shares the
        translated path's error contract: failures already normalized to an IR
        error by the upstream client are projected to a downstream-native
        exception so every route exposes the same cause chain.
        """
        return self._stream_with_error_handling(events)

    async def _stream_with_error_handling(
        self,
        events: AsyncIterator[NativeEventT],
    ) -> AsyncIterator[NativeEventT]:
        try:
            async for event in events:
                yield event
        except Exception as error:
            self._handle_error(error)
        finally:
            # An `async for` loop does not close its iterator when this generator
            # is closed early; propagate closure so the SDK stream is released
            # deterministically instead of relying on garbage collection.
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

    @abstractmethod
    def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[NativeEventT]:
        """Implement downstream event projection."""
        raise NotImplementedError

    @abstractmethod
    def _handle_error(self, error: Exception) -> NoReturn:
        """Translate or re-raise an error raised while streaming downstream events."""
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
