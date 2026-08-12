from httpx import AsyncClient
from openai import AsyncOpenAI, omit


class BaseClient:
    DEFAULT_BASE_URL: str

    def __init__(
        self,
        http_client: AsyncClient,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._http_client = http_client
        self._api_key = api_key
        self._base_url = base_url

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
