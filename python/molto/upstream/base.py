from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from httpx import AsyncClient
from openai import AsyncOpenAI, omit
from pydantic import BaseModel

from molto.providers import ProviderCapabilities
from molto.types.normalized import ModelResponseStreamEvent, NormalizedRequest


class BaseClient[NativeRequestT: BaseModel, NativeEventT](ABC):
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

    @abstractmethod
    def native_events(self, request: NativeRequestT) -> AsyncIterator[NativeEventT]:
        """Stream events using the provider's native protocol."""
        raise NotImplementedError

    @abstractmethod
    def events(self, request: NormalizedRequest) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream provider events translated to normalized events."""
        raise NotImplementedError
