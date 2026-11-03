from __future__ import annotations

import httpx
import pytest


@pytest.fixture(autouse=True)
def _block_httpx_network_in_playback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block default httpx network transport during every test run."""

    async def blocked_network(
        self: httpx.AsyncHTTPTransport,
        http_request: httpx.Request,
    ) -> httpx.Response:
        """Raise when default httpx transport would make a real network call.

        Example:
            # await blocked_network(transport, httpx.Request("GET", "https://example.test"))
            # raises AssertionError containing "Unexpected real network call"
        """
        raise AssertionError(
            "Unexpected real network call during cassette playback: "
            f"{http_request.method} {http_request.url}"
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_network)
