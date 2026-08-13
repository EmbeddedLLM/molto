from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel

from molto.types.normalized import ModelResponseStreamEvent, NormalizedRequest


class BaseAdapter[NativeRequestT: BaseModel, NativeEventT](ABC):
    """Translate the downstream protocol to and from normalized IR."""

    def __init__(self) -> None:
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
        return self._project_events(events, request)

    @abstractmethod
    def _project_events(
        self,
        events: AsyncIterator[ModelResponseStreamEvent],
        request: NormalizedRequest,
    ) -> AsyncIterator[NativeEventT]:
        """Implement downstream event projection."""
        raise NotImplementedError
