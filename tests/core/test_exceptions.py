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
from molto.types.exceptions import (
    APIException,
    AuthenticationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ResourceNotFoundError,
)
from molto.utils.io import json_loads, read_yaml

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes" / "errors"
Protocol = Literal["responses", "chat/completions", "messages"]
PROTOCOLS: tuple[Protocol, ...] = ("responses", "chat/completions", "messages")
ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(product(PROTOCOLS, PROTOCOLS))
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
            "responses": OpenAIBadRequestError,
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


def _cassette_path(scenario: str, protocol: Protocol) -> Path:
    filename = {
        "responses": "openai-responses-stream-00.yaml",
        "chat/completions": "openai-chat_completions-stream-00.yaml",
        "messages": "anthropic-messages-stream-00.yaml",
    }[protocol]
    return CASSETTE_DIR / scenario / filename


def _read_cassette(scenario: str, protocol: Protocol) -> dict[str, Any]:
    return read_yaml(_cassette_path(scenario, protocol))


def _response_body(cassette: dict[str, Any]) -> dict[str, Any]:
    body = cassette["response"]["body"]
    if isinstance(body, str):
        # Router 404s from a deliberately wrong URL legitimately carry no body.
        return json_loads(body) if body else {}
    return body


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
    stream: bool = True,
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
        body = deepcopy(cassette["request"]["body"])
        body["stream"] = stream
        with pytest.raises(error_type) as exc_info:
            result = await create(**body)
            if stream:
                async with result:
                    async for event in result:
                        events.append(event)
        response = result.response if "sse" in cassette["response"] else exc_info.value.response
        assert response.is_closed
    assert len(requests) == 1
    assert type(exc_info.value) is error_type
    return exc_info.value, events


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("scenario", ERROR_SCENARIOS, ids=scenario_ids(ERROR_SCENARIOS))
async def test_error_scenario(scenario: ErrorScenario, downstream: Protocol, upstream: Protocol):
    downstream_cassette = _read_cassette(scenario.name, downstream)
    upstream_cassette = _read_cassette(scenario.name, upstream)
    expected_response = upstream_cassette["response"]
    stream_error = "sse" in expected_response
    sdk_error, sdk_events = await _replay_sdk_error(
        upstream_cassette, upstream, scenario.error_types[upstream]
    )
    # Native routes preserve the SDK subclass; translated routes preserve whether
    # the provider reported an HTTP failure or an error inside a successful stream.
    if downstream == upstream:
        expected_error_type = scenario.error_types[downstream]
    elif stream_error:
        expected_error_type = AnthropicAPIError if downstream == "messages" else OpenAIAPIError
    else:
        expected_error_type = DOWNSTREAM_STATUS_ERROR_TYPES[downstream].get(
            expected_response["status_code"], DOWNSTREAM_ERROR_TYPES[downstream]
        )

    body = deepcopy(downstream_cassette["request"]["body"])
    expected_request = upstream_cassette["request"]
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
            stream = await coordinator(
                body=body,
                endpoint=upstream,
                provider="anthropic" if upstream == "messages" else "openai",
                api_key="test-api-key",
            )
            async with stream:
                async for event in stream:
                    events.append(event)
        assert exc_info.value.__cause__.response.is_closed
        assert not http_client.is_closed

    assert len(requests) == 1
    request = requests[0]
    origin = "https://api.anthropic.com" if upstream == "messages" else "https://api.openai.com"
    assert request.method == expected_request["method"]
    if scenario.noncanonical_request_url:
        # The recorded path is the deliberate wrong-URL trigger for a router 404.
        # Molto always builds canonical provider URLs, so assert the request
        # targets the upstream host and endpoint instead of the recorded path.
        assert str(request.url).startswith(origin)
        assert request.url.path.endswith(f"/{upstream}")
    else:
        assert str(request.url) == f"{origin}{expected_request['path']}"
    assert json_loads(request.content) == expected_request["body"]

    error = exc_info.value
    assert type(error) is expected_error_type
    normalized_error = error.__cause__
    assert type(normalized_error) is scenario.normalized_error
    original_error = normalized_error.__cause__
    assert type(original_error) is type(sdk_error) is scenario.error_types[upstream]
    assert error.message == normalized_error.message == original_error.message == sdk_error.message
    assert error.body is normalized_error.body is original_error.body
    assert error.body == sdk_error.body
    assert normalized_error.response.status_code == expected_response["status_code"]
    assert normalized_error.request is normalized_error.response.request
    # Reconstructing the outward exception avoids a cycle in the cause chain.
    assert error is not original_error

    if stream_error:
        assert not hasattr(error, "status_code")
        assert not hasattr(error, "response")
        assert not hasattr(original_error, "status_code")
        assert error.request is normalized_error.response.request is original_error.request
    else:
        assert error.status_code == sdk_error.status_code == expected_response["status_code"]
        assert error.response is normalized_error.response is original_error.response
        expected_body = _response_body(upstream_cassette)
        if expected_body:
            assert expected_body["error"]["message"] in error.message
            assert error.response.json() == sdk_error.response.json() == expected_body
        else:
            # Router 404s legitimately carry no body.
            assert error.response.content == sdk_error.response.content == b""

    if downstream == upstream:
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


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("scenario", ERROR_SCENARIOS, ids=scenario_ids(ERROR_SCENARIOS))
async def test_non_streaming_error_scenario(
    scenario: ErrorScenario,
    downstream: Protocol,
    upstream: Protocol,
):
    translated = downstream != upstream
    downstream_cassette = _read_cassette(scenario.name, downstream)
    upstream_cassette = _read_cassette(scenario.name, upstream)
    upstream_error_type = scenario.error_types[upstream]
    if downstream == upstream == "responses" and scenario.name == "context_length_exceeded":
        # A native non-streaming request needs the recorded HTTP error, while
        # translated requests still consume the upstream streaming failure.
        upstream_cassette = read_yaml(
            CASSETTE_DIR.parent / "openai" / "exceptions" / "context_length_exceeded.yaml"
        )
        downstream_cassette = upstream_cassette
        upstream_error_type = OpenAIBadRequestError
    request_body = deepcopy(downstream_cassette["request"]["body"])
    request_body["stream"] = False
    http_client, requests = _get_http_client(upstream_cassette)
    expected_response = upstream_cassette["response"]
    stream_error = "sse" in expected_response
    sdk_error, _ = await _replay_sdk_error(
        upstream_cassette, upstream, upstream_error_type, stream=translated
    )
    if not translated:
        expected_error_type = upstream_error_type
    elif stream_error:
        expected_error_type = AnthropicAPIError if downstream == "messages" else OpenAIAPIError
    else:
        expected_error_type = DOWNSTREAM_STATUS_ERROR_TYPES[downstream].get(
            expected_response["status_code"], DOWNSTREAM_ERROR_TYPES[downstream]
        )

    async with http_client:
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
                provider="anthropic" if upstream == "messages" else "openai",
                api_key="test-api-key",
            )
        assert exc_info.value.__cause__.response.is_closed
        assert not http_client.is_closed

    assert len(requests) == 1
    expected_request_body = upstream_cassette["request"]["body"] if translated else request_body
    assert json_loads(requests[0].content) == expected_request_body
    error = exc_info.value
    assert type(error) is expected_error_type
    normalized_error = error.__cause__
    assert type(normalized_error) is scenario.normalized_error
    original_error = normalized_error.__cause__
    assert type(original_error) is type(sdk_error) is upstream_error_type
    assert error.message == normalized_error.message == original_error.message == sdk_error.message
    assert error.body is normalized_error.body is original_error.body
    assert error.body == sdk_error.body
    assert normalized_error.response.status_code == expected_response["status_code"]
    assert normalized_error.request is normalized_error.response.request
    assert error is not original_error
    if stream_error:
        assert not hasattr(error, "status_code")
        assert not hasattr(error, "response")
        assert not hasattr(original_error, "status_code")
        assert error.request is normalized_error.response.request is original_error.request
    else:
        assert error.status_code == sdk_error.status_code == expected_response["status_code"]
        assert error.response is normalized_error.response is original_error.response
        assert error.response.content == sdk_error.response.content
