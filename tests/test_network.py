import httpx
import pytest


async def test_default_transport_cannot_reach_the_network() -> None:
    """Ensures an unmocked HTTP request fails before any network access during tests."""
    async with httpx.AsyncClient() as client:
        with pytest.raises(AssertionError, match="Unexpected real network call"):
            await client.get("https://example.test/")
