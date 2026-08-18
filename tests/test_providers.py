from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError

import httpx
import pytest

from molto import Molto
from molto.providers import (
    DEFAULT_PROVIDER_BY_ENDPOINT,
    PROVIDER_CAPABILITIES,
    ProviderCapabilities,
)
from molto.types.normalized import ModelResponseStreamEvent, NormalizedRequest
from molto.upstream.anthropic import AnthropicMessagesClient


def test_provider_capabilities_are_explicit_and_immutable() -> None:
    assert PROVIDER_CAPABILITIES == {
        "anthropic": ProviderCapabilities(thinking="summary"),
        "openai": ProviderCapabilities(thinking="summary"),
        "vllm": ProviderCapabilities(thinking="raw"),
    }
    with pytest.raises(FrozenInstanceError):
        PROVIDER_CAPABILITIES["vllm"].thinking = "summary"  # type: ignore[misc]


def test_default_provider_is_separate_from_endpoint_selection() -> None:
    assert DEFAULT_PROVIDER_BY_ENDPOINT == {
        "responses": "openai",
        "chat/completions": "openai",
        "messages": "anthropic",
    }


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("openai", "https://vllm.example"),
        ("vllm", "https://api.anthropic.com"),
    ],
)
async def test_provider_selects_capabilities_independently_of_endpoint_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    base_url: str,
) -> None:
    captured: dict[str, object] = {}

    def stub_events(
        client: AnthropicMessagesClient,
        request: NormalizedRequest,
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        captured["capabilities"] = client._capabilities
        captured["base_url"] = client._base_url

        async def empty_stream() -> AsyncIterator[ModelResponseStreamEvent]:
            if False:
                yield

        return empty_stream()

    monkeypatch.setattr(AnthropicMessagesClient, "events", stub_events)
    client = Molto(http_client=httpx.AsyncClient())
    await client.responses(
        body={"input": "Question", "model": "m", "stream": True},
        endpoint="messages",
        provider=provider,
        base_url=base_url,
    )

    assert captured == {
        "capabilities": PROVIDER_CAPABILITIES[provider],
        "base_url": base_url,
    }
    await client.close()
