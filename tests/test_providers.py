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
    """Ensures each provider has an explicit immutable reasoning profile."""
    assert PROVIDER_CAPABILITIES == {
        "anthropic": ProviderCapabilities(thinking="summary"),
        "openai": ProviderCapabilities(thinking="summary"),
        "togetherai": ProviderCapabilities(thinking="raw"),
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
        ("togetherai", "https://together.example/v1"),
    ],
)
async def test_provider_selects_capabilities_independently_of_endpoint_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    base_url: str,
) -> None:
    """Ensures provider capabilities do not override the requested Messages endpoint."""
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
    await client.aclose()


_STREAMING_BODIES = {
    "responses": {"input": "Hi", "model": "m", "stream": True},
    "chat_completions": {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "m",
        "stream": True,
    },
    "messages": {
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "m",
        "stream": True,
    },
}


@pytest.mark.parametrize("provider", [None, "openai", "togetherai"])
@pytest.mark.parametrize("coordinator", ["responses", "chat_completions", "messages"])
async def test_coordinators_reject_unsupported_endpoints(
    provider: str | None,
    coordinator: str,
) -> None:
    """Ensures coordinators reject unknown endpoints before choosing a client."""
    # The endpoint Literal is not enforced at runtime; an unknown endpoint must
    # fail explicitly instead of falling through to a default client or provider.
    client = Molto(http_client=httpx.AsyncClient())
    with pytest.raises(ValueError, match="Unsupported endpoint: 'bogus'"):
        await getattr(client, coordinator)(
            body=_STREAMING_BODIES[coordinator],
            endpoint="bogus",
            provider=provider,
        )
    await client.aclose()


@pytest.mark.parametrize("provider", ["openai", "togetherai"])
async def test_coordinators_validate_endpoint_before_request_normalization(
    provider: str,
) -> None:
    """Ensures endpoint errors precede unrelated cross-protocol normalization failures."""
    # The body below fails adapter normalization (namespaced function call), but
    # the unsupported endpoint must be reported first, not the translation error.
    client = Molto(http_client=httpx.AsyncClient())
    with pytest.raises(ValueError, match="Unsupported endpoint: 'bogus'"):
        await client.responses(
            body={
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_123",
                        "name": "lookup",
                        "arguments": "{}",
                        "namespace": "tools",
                    }
                ],
                "model": "m",
                "stream": True,
            },
            endpoint="bogus",
            provider=provider,
        )
    await client.aclose()
