from typing import AsyncIterator

from httpx import AsyncClient, Response


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

    def _endpoint_url(self, endpoint: str) -> str:
        return f"{(self._base_url or self.DEFAULT_BASE_URL).rstrip('/')}/{endpoint}"

    def _authorization_headers(self) -> dict[str, str]:
        return {} if not self._api_key else {"authorization": f"Bearer {self._api_key}"}

    async def _iter_sse_records(self, response: Response) -> AsyncIterator[str]:
        lines: list[str] = []
        async for line in response.aiter_lines():
            if line:
                lines.append(line)
            elif lines:
                yield f"{'\n'.join(lines)}\n\n"
                lines.clear()
        if lines:
            yield f"{'\n'.join(lines)}\n\n"

    def _sse_data(self, record: str) -> str:
        return "\n".join(
            line.removeprefix("data:").removeprefix(" ")
            for line in record.splitlines()
            if line.startswith("data:")
        )
