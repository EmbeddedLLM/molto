"""Provider/protocol selection and credential boundaries, exercised without live requests."""

from copy import deepcopy

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, OpenAIError
from pydantic import SecretStr

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.base import BaseClient
from molto.utils.io import json_loads
from tests.test_hosted_providers import DESTINATIONS, cassette, response

PROVIDERS = {
    "anthropic": ("Anthropic", "anthropic_api_key"),
    "bedrock": ("Bedrock", "aws_bearer_token_bedrock"),
    "fireworks": ("Fireworks", "fireworks_api_key"),
    "gemini": ("Gemini", "gemini_api_key"),
    "openai": ("OpenAI", "openai_api_key"),
    "openrouter": ("OpenRouter", "openrouter_api_key"),
    "togetherai": ("TogetherAI", "together_ai_api_key"),
    "vllm": ("VLLM", None),
}
PROTOCOLS = {
    "responses": ("Responses", "/responses"),
    "chat/completions": ("ChatCompletions", "/chat/completions"),
    "messages": ("Messages", "/v1/messages"),
}
ROUTES = [
    (provider, endpoint)
    for provider in PROVIDERS
    for endpoint in PROTOCOLS
    if provider != "gemini" or endpoint == "chat/completions"
]
REQUEST_TYPES = {
    "responses": OpenAIResponsesCreateRequest,
    "chat/completions": OpenAIChatCompletionsCreateRequest,
    "messages": AnthropicMessagesCreateRequest,
}
HOSTED_ROUTES = [route for route in ROUTES if route[0] not in {"openai", "anthropic", "vllm"}]
EXPLICIT_DESTINATIONS = [("bedrock", endpoint) for endpoint in PROTOCOLS] + [
    ("togetherai", "responses"),
    ("togetherai", "messages"),
]


@pytest.mark.parametrize(("provider", "endpoint"), ROUTES)
async def test_provider_protocol_selection_preserves_native_contract(
    provider, endpoint, monkeypatch
):
    """Ensures every existing route selects its provider class and preserves URL, key, and SDK output."""
    prefix, setting = PROVIDERS[provider]
    suffix, path = PROTOCOLS[endpoint]
    key = "configured-provider-key" if setting else "EMPTY"
    if setting:
        monkeypatch.setattr(ENV_CONFIG, setting, SecretStr(key))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-anthropic-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("OPENAI_ORG_ID", "unrelated-organization")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated-project")
    constructed = []
    original_init = BaseClient.__init__

    def observe_client(self, *args, **kwargs):
        constructed.append(type(self))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(BaseClient, "__init__", observe_client)
    recording = cassette(endpoint, False)
    body = deepcopy(recording["request"]["body"])
    body["stream"] = False
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://configured.example/prefix" + path
        assert json_loads(request.content) == body
        bearer = endpoint != "messages" or provider in {"bedrock", "openrouter"}
        assert request.headers.get("authorization") == (f"Bearer {key}" if bearer else None)
        assert request.headers.get("x-api-key") == (None if bearer else key)
        assert "openai-organization" not in request.headers
        assert "openai-project" not in request.headers
        return response(recording)

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        result = await getattr(molto, endpoint.replace("/", "_"))(
            body,
            provider=provider,
            endpoint=endpoint,
            base_url="https://configured.example/prefix",
            extra_headers={"Authorization": "inbound", "x-api-key": "inbound"},
        )
    assert [client.__name__ for client in constructed] == [f"{prefix}{suffix}Client"]
    assert len(requests) == 1
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response(recording))
    ) as transport:
        if endpoint == "messages":
            sdk = AsyncAnthropic(api_key="sdk-key", http_client=transport)
            expected = await sdk.messages.create(**body)
        else:
            sdk = AsyncOpenAI(api_key="sdk-key", http_client=transport)
            create = (
                sdk.responses.create if endpoint == "responses" else sdk.chat.completions.create
            )
            expected = await create(**body)
    assert result.model_dump() == expected.model_dump()


@pytest.mark.parametrize("endpoint", ["responses", "messages"])
async def test_gemini_rejects_other_protocols_before_http(endpoint):
    """Ensures explicit provider classes do not imply Gemini supports another upstream protocol."""
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Unsupported Gemini protocol reached HTTP")
            )
        )
    ) as molto:
        with pytest.raises(ValueError, match="Gemini supports only"):
            await molto.responses(
                {"model": "example", "input": "Hi"},
                provider="gemini",
                endpoint=endpoint,
                api_key="test-key",
                base_url="https://configured.example",
            )


@pytest.mark.parametrize(("provider", "endpoint"), EXPLICIT_DESTINATIONS)
async def test_provider_routes_still_require_explicit_destination(provider, endpoint):
    """Ensures alternate Together and regional Bedrock routes cannot fall back to another service."""
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Unconfigured provider destination reached HTTP")
            )
        )
    ) as molto:
        with pytest.raises(ValueError, match="explicit"):
            await molto.responses(
                {"model": "example", "input": "Hi"},
                provider=provider,
                endpoint=endpoint,
                api_key="test-key",
            )


@pytest.mark.parametrize(
    ("provider", "endpoint", "base", "path"),
    DESTINATIONS
    + [
        ("togetherai", "chat/completions", "https://api.together.ai/v1", "/chat/completions"),
    ],
)
async def test_direct_hosted_clients_own_destination_and_authentication(
    provider, endpoint, base, path, monkeypatch
):
    """Ensures directly constructed hosted clients retain their destination and isolated auth defaults."""
    from molto.upstream.registry import CLIENTS

    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-token")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS", "authorization: unrelated\nX-API-KEY: unrelated"
    )
    recording = cassette(endpoint, False)
    request = REQUEST_TYPES[endpoint].model_validate(recording["request"]["body"])
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == base + path
        bearer = endpoint != "messages" or provider in {"bedrock", "openrouter"}
        assert request.headers.get("authorization") == ("Bearer selected-key" if bearer else None)
        assert request.headers.get("x-api-key") == (None if bearer else "selected-key")
        return response(recording)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = CLIENTS[provider, endpoint](
            transport,
            api_key="selected-key",
            base_url=base if provider == "bedrock" else None,
            capabilities=PROVIDER_CAPABILITIES[provider],
        )
        await client.native_response(request)
    assert len(requests) == 1


@pytest.mark.parametrize(("provider", "endpoint"), HOSTED_ROUTES)
async def test_direct_hosted_clients_never_fall_back_to_other_credentials(
    provider, endpoint, monkeypatch
):
    """Ensures hosted subclasses enforce their credential boundary without facade-supplied flags."""
    from molto.upstream.registry import CLIENTS

    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-token")
    async with httpx.AsyncClient() as transport:
        client = CLIENTS[provider, endpoint](
            transport,
            base_url="https://configured.example",
            capabilities=PROVIDER_CAPABILITIES[provider],
        )
        error_type = TypeError if endpoint == "messages" else OpenAIError
        with pytest.raises(error_type, match=PROVIDERS[provider][1].upper()):
            if endpoint == "messages":
                client._make_anthropic_client()
            else:
                client._make_sdk_client()


@pytest.mark.parametrize(("provider", "endpoint"), EXPLICIT_DESTINATIONS)
async def test_direct_provider_clients_require_their_destination(provider, endpoint):
    """Ensures direct regional or alternative clients cannot send provider keys to canonical hosts."""
    from molto.upstream.registry import CLIENTS

    async with httpx.AsyncClient() as transport:
        with pytest.raises(ValueError, match="explicit"):
            CLIENTS[provider, endpoint](
                transport, api_key="test-key", capabilities=PROVIDER_CAPABILITIES[provider]
            )
