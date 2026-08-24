import pytest
from httpx import AsyncClient

from molto import Molto


async def test_context_manager_closes_http_client_on_error() -> None:
    http_client = AsyncClient()
    molto = Molto(http_client=http_client)

    with pytest.raises(RuntimeError, match="stream failed"):
        async with molto as entered_molto:
            assert entered_molto is molto
            assert not http_client.is_closed
            raise RuntimeError("stream failed")

    assert http_client.is_closed
