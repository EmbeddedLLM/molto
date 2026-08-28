from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import pytest
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import NotFoundError as AnthropicNotFoundError
from httpx import AsyncClient, MockTransport, Request, Response
from openai import APIStatusError as OpenAIAPIStatusError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import BadRequestError as OpenAIBadRequestError
from openai import NotFoundError as OpenAINotFoundError

from molto import Molto
from molto.types.exceptions import AuthenticationError, ModelNotFoundError, MoltoException
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


@dataclass(frozen=True)
class ErrorScenario:
    """A normalized error and its concrete SDK type in each protocol."""

    name: str
    normalized_error: type[MoltoException]
    error_types: dict[Protocol, type[OpenAIAPIStatusError | AnthropicAPIStatusError]]


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
    return json_loads(body) if isinstance(body, str) else body


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


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
@pytest.mark.parametrize("scenario", ERROR_SCENARIOS, ids=scenario_ids(ERROR_SCENARIOS))
async def test_error_scenario(scenario: ErrorScenario, downstream: Protocol, upstream: Protocol):
    downstream_cassette = _read_cassette(scenario.name, downstream)
    upstream_cassette = _read_cassette(scenario.name, upstream)
    expected_response = upstream_cassette["response"]
    expected_body = _response_body(upstream_cassette)
    # Native passthrough preserves the original SDK subclass. Translated routes
    # project the normalized semantic error into the downstream SDK namespace.
    expected_error_type = (
        scenario.error_types[upstream]
        if downstream == upstream
        else DOWNSTREAM_ERROR_TYPES[downstream]
    )
    http_client, requests = _get_http_client(upstream_cassette)

    async with http_client:
        client = Molto(http_client=http_client)
        coordinator = {
            "responses": client.responses,
            "chat/completions": client.chat_completions,
            "messages": client.messages,
        }[downstream]
        with pytest.raises(expected_error_type) as exc_info:
            await coordinator(
                body=deepcopy(downstream_cassette["request"]["body"]),
                endpoint=upstream,
                provider="anthropic" if upstream == "messages" else "openai",
                api_key="test-api-key",
            )

    assert len(requests) == 1
    request = requests[0]
    expected_request = upstream_cassette["request"]
    origin = "https://api.anthropic.com" if upstream == "messages" else "https://api.openai.com"
    assert request.method == expected_request["method"]
    assert str(request.url) == f"{origin}{expected_request['path']}"
    assert json_loads(request.content) == expected_request["body"]

    error = exc_info.value
    assert error.status_code == expected_response["status_code"]
    assert expected_body["error"]["message"] in error.message
    assert error.response.json() == expected_body

    # Native and translated routes share one error contract: the outward
    # exception is the downstream SDK namespace, caused by the normalized IR
    # error, which in turn preserves the original upstream SDK exception.
    assert type(error) is expected_error_type
    normalized_error = error.__cause__
    assert isinstance(normalized_error, scenario.normalized_error)
    assert normalized_error.message == error.message
    assert normalized_error.response is error.response
    assert normalized_error.body == error.body
    original_error = normalized_error.__cause__
    assert type(original_error) is scenario.error_types[upstream]
    # The outward exception is reconstructed, never the original instance:
    # chaining the original `from` the normalized error would make the two
    # exceptions each other's cause.
    assert error is not original_error
