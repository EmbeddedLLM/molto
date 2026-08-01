from __future__ import annotations

import httpx
import pytest

from tests.support import OpenAICassetteManager


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the explicit cassette recording option.

    Example:
        pytest_addoption(parser)
        # parser accepts --record-cassettes as a boolean flag
    """
    parser.addoption(
        "--record-cassettes",
        action="store_true",
        default=False,
        help="Record OpenAI SDK calls to sanitized YAML cassettes.",
    )


@pytest.fixture
def cassette_manager(request: pytest.FixtureRequest) -> OpenAICassetteManager:
    """Return the per-test cassette manager fixture.

    Example:
        # In tests/test_demo.py::test_demo[stream], cassette_manager.safe_node_name is:
        # "test_demo-stream"
    """
    return OpenAICassetteManager.from_pytest_request(
        request,
        record=request.config.getoption("--record-cassettes"),
    )


@pytest.fixture(autouse=True)
def _block_httpx_network_in_playback(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block default httpx network transport during cassette playback.

    Example:
        # Without --record-cassettes, a raw httpx.AsyncHTTPTransport request raises:
        # AssertionError("Unexpected real network call during cassette playback: ...")
    """
    if request.config.getoption("--record-cassettes"):
        return

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
