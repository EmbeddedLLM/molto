from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import pytest
from anthropic import APIError as AnthropicAPIError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic import NotFoundError as AnthropicNotFoundError
from httpx import AsyncClient, MockTransport, Request, Response
from openai import APIError as OpenAIAPIError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import AsyncOpenAI, OpenAIError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import BadRequestError as OpenAIBadRequestError
from openai import NotFoundError as OpenAINotFoundError
from pydantic import SecretStr

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.providers import DEFAULT_PROVIDER_BY_ENDPOINT, Provider
from molto.types.exceptions import (
    APIException,
    AuthenticationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ResourceNotFoundError,
)
from molto.utils.io import json_loads, read_yaml
from tests.core.cassette_helpers import _split_sdk_request_body

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes" / "errors"
Protocol = Literal["responses", "chat/completions", "messages"]
PROTOCOLS: tuple[Protocol, ...] = ("responses", "chat/completions", "messages")
ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(product(PROTOCOLS, PROTOCOLS))
ERROR_UPSTREAMS: tuple[tuple[Protocol, Provider], ...] = (
    ("responses", "openai"),
    ("chat/completions", "openai"),
    ("messages", "anthropic"),
    ("chat/completions", "togetherai"),
)
ERROR_ROUTES = tuple(
    (downstream, upstream, provider)
    for downstream, (upstream, provider) in product(PROTOCOLS, ERROR_UPSTREAMS)
)
DOWNSTREAM_ERROR_TYPES: dict[Protocol, type[OpenAIAPIStatusError | AnthropicAPIStatusError]] = {
    "responses": OpenAIAPIStatusError,
    "chat/completions": OpenAIAPIStatusError,
    "messages": AnthropicAPIStatusError,
}
DOWNSTREAM_STATUS_ERROR_TYPES: dict[
    Protocol,
    dict[int, type[OpenAIAPIStatusError | AnthropicAPIStatusError]],
] = {
    "responses": {
        400: OpenAIBadRequestError,
        401: OpenAIAuthenticationError,
        404: OpenAINotFoundError,
    },
    "chat/completions": {
        400: OpenAIBadRequestError,
        401: OpenAIAuthenticationError,
        404: OpenAINotFoundError,
    },
    "messages": {
        401: AnthropicAuthenticationError,
        404: AnthropicNotFoundError,
    },
}


@dataclass(frozen=True)
class ErrorScenario:
    """A normalized error and its concrete SDK type in each protocol."""

    name: str
    normalized_error: type[APIException]
    error_types: dict[Protocol, type[OpenAIAPIError | AnthropicAPIError]]
    # True when the error was triggered by a deliberately wrong request URL.
    # Current SDKs normalize base_url joining, so Molto cannot reproduce the
    # recorded path; replay asserts the host and endpoint instead.
    noncanonical_request_url: bool = False


# New scenarios: record cassettes under tests/cassettes/errors/<name>/ and append here.
ERROR_SCENARIOS: tuple[ErrorScenario, ...] = (
    ErrorScenario(
        name="model_not_found",
        normalized_error=ModelNotFoundError,
        error_types={
            "responses": OpenAINotFoundError,
            "chat/completions": OpenAINotFoundError,
            "messages": AnthropicNotFoundError,
        },
    ),
    ErrorScenario(
        name="invalid_api_key",
        normalized_error=AuthenticationError,
        error_types={
            "responses": OpenAIAuthenticationError,
            "chat/completions": OpenAIAuthenticationError,
            "messages": AnthropicAuthenticationError,
        },
    ),
    ErrorScenario(
        name="not_found",
        normalized_error=ResourceNotFoundError,
        error_types={
            "responses": OpenAINotFoundError,
            "chat/completions": OpenAINotFoundError,
            "messages": AnthropicNotFoundError,
        },
        noncanonical_request_url=True,
    ),
    ErrorScenario(
        name="context_length_exceeded",
        normalized_error=ContextLengthExceededError,
        error_types={
            "responses": OpenAIAPIError,
            "chat/completions": OpenAIBadRequestError,
            "messages": AnthropicBadRequestError,
        },
    ),
)


def scenario_ids(scenarios: Sequence[ErrorScenario]) -> list[str]:
    return [scenario.name for scenario in scenarios]


def _cassette_path(
    scenario: str,
    protocol: Protocol,
    *,
    stream: bool = True,
    provider: Provider | None = None,
) -> Path:
    provider = provider or DEFAULT_PROVIDER_BY_ENDPOINT[protocol]
    endpoint = protocol.replace("/", "_")
    filename = f"{provider}-{endpoint}-{'stream' if stream else 'non_stream'}-00.yaml"
    return CASSETTE_DIR / scenario / filename


def _read_cassette(
    scenario: str,
    protocol: Protocol,
    *,
    stream: bool = True,
    provider: Provider | None = None,
) -> dict[str, Any]:
    return read_yaml(_cassette_path(scenario, protocol, stream=stream, provider=provider))


def _get_http_client(cassette: dict[str, Any]) -> tuple[AsyncClient, list[Request]]:
    # This is useful to assert retries
    requests: list[Request] = []

    def cassette_handler(request: Request) -> Response:
        requests.append(request)
        cassette_response = cassette["response"]
        headers = {
            name: value
            for name, value in cassette_response["headers"].items()
            if name.lower() not in {"connection", "content-length"}
        }
        if "sse" in cassette_response:
            return Response(
                cassette_response["status_code"],
                content="".join(cassette_response["sse"]).encode(),
                headers=headers,
            )
        body = cassette_response["body"]
        if isinstance(body, str):
            return Response(
                cassette_response["status_code"],
                content=body.encode(),
                headers=headers,
            )
        return Response(
            cassette_response["status_code"],
            json=body,
            headers=headers,
        )

    return AsyncClient(transport=MockTransport(cassette_handler)), requests


async def _replay_sdk_error(
    cassette: dict[str, Any],
    protocol: Protocol,
    error_type: type[OpenAIAPIError | AnthropicAPIError],
    *,
    base_url: str | None = None,
    provider: Provider | None = None,
) -> tuple[OpenAIAPIError | AnthropicAPIError, list[Any]]:
    http_client, requests = _get_http_client(cassette)
    events = []
    async with http_client:
        if protocol == "messages":
            sdk = AsyncAnthropic(
                http_client=http_client, api_key="test-api-key", max_retries=0, base_url=base_url
            )
            create = sdk.messages.create
        else:
            sdk = AsyncOpenAI(
                http_client=http_client, api_key="test-api-key", max_retries=0, base_url=base_url
            )
            create = (
                sdk.responses.create if protocol == "responses" else sdk.chat.completions.create
            )
        body, extra_body = _split_sdk_request_body(
            deepcopy(cassette["request"]["body"]),
            protocol,
            provider or DEFAULT_PROVIDER_BY_ENDPOINT[protocol],
        )
        with pytest.raises(error_type) as exc_info:
            result = await create(**body, extra_body=extra_body)
            if body["stream"]:
                async with result:
                    async for event in result:
                        events.append(event)
        response = result.response if "sse" in cassette["response"] else exc_info.value.response
        assert response.is_closed
    assert len(requests) == 1
    assert type(exc_info.value) is error_type
    return exc_info.value, events


@pytest.mark.parametrize(("downstream", "upstream", "provider"), ERROR_ROUTES)
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("scenario", ERROR_SCENARIOS, ids=scenario_ids(ERROR_SCENARIOS))
async def test_error_scenario(
    scenario: ErrorScenario,
    downstream: Protocol,
    upstream: Protocol,
    provider: Provider,
    stream: bool,
):
    """Ensures every provider and mode preserves SDK errors, normalized causes, and transport."""
    upstream_cassette = _read_cassette(scenario.name, upstream, stream=stream, provider=provider)
    expected_request = upstream_cassette["request"]
    expected_body = deepcopy(expected_request["body"])
    translated = downstream != upstream
    body, extra_body = _split_sdk_request_body(deepcopy(expected_body), upstream, provider)
    if translated:
        body = deepcopy(
            _read_cassette(scenario.name, downstream, stream=stream)["request"]["body"]
        )
    assert body["stream"] is stream
    assert expected_body["stream"] is stream
    if translated and not stream:
        # Translated calls always stream upstream, even for non-streaming callers.
        # Streaming fixtures already match the outgoing body exactly.
        expected_body["stream"] = True
        if upstream == "chat/completions":
            expected_body.setdefault("stream_options", {})["include_usage"] = True
        if downstream == "messages":
            expected_body.setdefault("stream_options", {})["include_obfuscation"] = False

    expected_response = upstream_cassette["response"]
    stream_error = "sse" in expected_response
    status = expected_response["status_code"]
    upstream_error_type = (
        scenario.error_types[upstream]
        if stream_error
        else DOWNSTREAM_STATUS_ERROR_TYPES[upstream].get(status, scenario.error_types[upstream])
    )
    if not translated:
        expected_error_type = upstream_error_type
    elif stream_error:
        expected_error_type = AnthropicAPIError if downstream == "messages" else OpenAIAPIError
    else:
        expected_error_type = DOWNSTREAM_STATUS_ERROR_TYPES[downstream].get(
            status, DOWNSTREAM_ERROR_TYPES[downstream]
        )
    origin = {
        "openai": "https://api.openai.com",
        "anthropic": "https://api.anthropic.com",
        "togetherai": "https://api.together.ai",
    }[provider]
    sdk_error, sdk_events = await _replay_sdk_error(
        upstream_cassette,
        upstream,
        upstream_error_type,
        base_url=f"{origin}/v1" if upstream != "messages" else origin,
        provider=provider,
    )
    http_client, requests = _get_http_client(upstream_cassette)
    events = []
    async with http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        with pytest.raises(expected_error_type) as exc_info:
            result = await coordinator(
                body=body,
                endpoint=upstream,
                provider=provider,
                api_key="test-api-key",
                extra_body=extra_body,
            )
            if stream:
                async with result:
                    async for event in result:
                        events.append(event)
        assert exc_info.value.__cause__.response.is_closed
        assert not http_client.is_closed

    assert len(requests) == 1
    request = requests[0]
    assert request.method == expected_request["method"]
    if scenario.noncanonical_request_url:
        # Router 404s were recorded at deliberately wrong paths; replay targets
        # the canonical provider endpoint, as in the other scenarios.
        assert str(request.url) == f"{origin}/v1/{upstream}"
    else:
        assert str(request.url) == f"{origin}{expected_request['path']}"
    assert json_loads(request.content) == expected_body

    error = exc_info.value
    assert type(error) is expected_error_type
    normalized_error = error.__cause__
    assert type(normalized_error) is scenario.normalized_error
    original_error = normalized_error.__cause__
    assert type(original_error) is type(sdk_error) is upstream_error_type
    assert original_error.__cause__ is None
    assert error.message == normalized_error.message == original_error.message == sdk_error.message
    assert error.body is normalized_error.body is original_error.body
    assert error.body == sdk_error.body
    assert normalized_error.response.status_code == status
    assert error.request is normalized_error.request is original_error.request is request
    assert normalized_error.request is normalized_error.response.request
    # Reconstructing the outward exception avoids a cycle in the cause chain.
    assert error is not original_error

    if stream_error:
        assert not hasattr(error, "status_code")
        assert not hasattr(error, "response")
        assert not hasattr(original_error, "status_code")
    else:
        assert error.status_code == sdk_error.status_code == status
        assert error.response is normalized_error.response is original_error.response
        assert error.response.headers == sdk_error.response.headers
        # Exact bytes cover JSON, empty router errors, and provider HTML bodies.
        assert error.response.content == sdk_error.response.content

    if not translated:
        assert events == sdk_events
    elif sdk_events and upstream == "responses" and downstream == "messages":
        # Only the source-backed start is representable before this failure.
        assert [event.type for event in events] == ["message_start"]
        assert events[0].message.id == sdk_events[0].response.id
        assert events[0].message.model == sdk_events[0].response.model
        assert events[0].message.content == []
        assert events[0].message.stop_reason is None
    else:
        assert events == []


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("api_key", [None, ""])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "https://gateway.example/api",
        "https://gateway.example/api/",
        "https://gateway.example/api//",
    ],
)
async def test_missing_credentials(downstream, upstream, api_key, stream, base_url, monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_ADMIN_KEY",
        "OPENAI_CUSTOM_HEADERS",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_CUSTOM_HEADERS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ENV_CONFIG, "openai_api_key", SecretStr(""))
    monkeypatch.setattr(ENV_CONFIG, "anthropic_api_key", SecretStr(""))
    upstream_cassette = _read_cassette("invalid_api_key", upstream)
    downstream_cassette = _read_cassette("invalid_api_key", downstream)
    request_body = deepcopy(downstream_cassette["request"]["body"])
    request_body["stream"] = stream
    http_client, requests = _get_http_client(upstream_cassette)
    sdk_error_type = TypeError if upstream == "messages" else OpenAIError
    expected_error_type = (
        AnthropicAuthenticationError if downstream == "messages" else OpenAIAuthenticationError
    )
    origin = base_url or (
        "https://api.anthropic.com" if upstream == "messages" else "https://api.openai.com/v1"
    )
    reference_error, _ = await _replay_sdk_error(
        upstream_cassette,
        upstream,
        AnthropicAuthenticationError if upstream == "messages" else OpenAIAuthenticationError,
        base_url=origin,
    )
    async with http_client:
        # Missing credentials fail locally; no response cassette should be consumed.
        with pytest.raises(sdk_error_type) as sdk_exc_info:
            if upstream == "messages":
                sdk = AsyncAnthropic(http_client=http_client, api_key="", max_retries=0)
                create = sdk.messages.create
            else:
                sdk = AsyncOpenAI(http_client=http_client, api_key="", max_retries=0)
                create = (
                    sdk.responses.create
                    if upstream == "responses"
                    else sdk.chat.completions.create
                )
            sdk_body = deepcopy(upstream_cassette["request"]["body"])
            sdk_body["stream"] = stream
            await create(**sdk_body)
        assert requests == []

        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        with pytest.raises(expected_error_type) as exc_info:
            await coordinator(
                body=request_body,
                endpoint=upstream,
                api_key=api_key,
                base_url=base_url,
            )
        assert not http_client.is_closed

    error = exc_info.value
    assert type(error) is expected_error_type
    normalized_error = error.__cause__
    assert type(normalized_error) is AuthenticationError
    original_error = normalized_error.__cause__
    assert type(original_error) is type(sdk_exc_info.value) is sdk_error_type
    assert (
        error.message == normalized_error.message == str(original_error) == str(sdk_exc_info.value)
    )
    assert error.body is normalized_error.body is None
    assert normalized_error.response is None
    assert error.status_code == error.response.status_code == 401
    assert error.request is error.response.request is normalized_error.request
    assert error.request.method == "POST"
    assert error.request.url == reference_error.request.url
    assert "authorization" not in error.request.headers
    assert "x-api-key" not in error.request.headers
    assert error.request_id is None
    assert original_error.__cause__ is None
    assert requests == []
