"""Public embeddings parity against independently recorded native SDK requests."""

import asyncio
import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from openai import APIStatusError, AsyncOpenAI, AuthenticationError
from openai.resources.embeddings import AsyncEmbeddings
from openai.types import CreateEmbeddingResponse
from pydantic import SecretStr, ValidationError

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.types.embeddings import CohereEmbeddingOptions
from molto.types.openai_embeddings import (
    OpenAIEmbeddingsCreateRequest,
    OpenAIEmbeddingsResponse,
)
from molto.utils.io import read_yaml

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes/embeddings/recorded/openai"
MANIFEST = read_yaml(CASSETTE_DIR / "manifest.yaml")
SUCCESS_CASES = [case for case in MANIFEST["scenarios"] if case["result"] == "success"]
ERROR_CASES = [case for case in MANIFEST["scenarios"] if case["result"] == "upstream_error"]


def _recording(name: str) -> dict[str, Any]:
    return read_yaml(CASSETTE_DIR / f"{name}.yaml")


def _client(cassette: dict[str, Any]) -> tuple[AsyncClient, list[Request]]:
    requests = []

    def replay(request: Request) -> Response:
        requests.append(request)
        assert request.method == cassette["request"]["method"]
        assert request.url.path == cassette["request"]["path"]
        assert json.loads(request.content) == cassette["request"]["body"]
        recorded = cassette["response"]
        return Response(
            recorded["status_code"],
            headers=recorded["headers"],
            content=recorded["body_bytes"],
            request=request,
        )

    return AsyncClient(transport=MockTransport(replay)), requests


@pytest.mark.parametrize("case", MANIFEST["scenarios"], ids=lambda case: case["scenario"])
def test_recorded_embedding_evidence_integrity(case: dict[str, Any]) -> None:
    """Ensures promoted HTTP bytes still match the independent recording manifest."""
    source = (CASSETTE_DIR / case["file"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == case["sha256"]
    cassette = read_yaml(CASSETTE_DIR / case["file"])
    assert json.loads(cassette["response"]["body_bytes"]) == cassette["response"]["body"]


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=lambda case: case["scenario"])
async def test_embeddings_native_recorded_sdk_parity(case: dict[str, Any]) -> None:
    """Ensures exact requests and parsed SDK vectors, indices, models, and usage survive."""
    cassette = read_yaml(CASSETTE_DIR / case["file"])
    body = deepcopy(case["sdk_arguments"])
    reference_client, reference_requests = _client(cassette)
    async with reference_client:
        reference = await AsyncOpenAI(
            api_key="reference-key", http_client=reference_client, max_retries=0
        ).embeddings.create(**body)

    http_client, requests = _client(cassette)
    async with Molto(http_client=http_client) as molto:
        actual = await molto.embeddings(body, api_key="molto-key")
        assert not http_client.is_closed
    assert http_client.is_closed
    assert isinstance(actual, CreateEmbeddingResponse)
    assert actual.model_dump(mode="python", warnings=False) == reference.model_dump(
        mode="python", warnings=False
    )
    assert body == case["sdk_arguments"]
    assert len(requests) == len(reference_requests) == 1
    assert requests[0].url == reference_requests[0].url
    assert actual.model == case["returned_model"]
    assert [row.index for row in actual.data] == case["indices"]
    expected_type = str if body.get("encoding_format") == "base64" else list
    assert all(isinstance(row.embedding, expected_type) for row in actual.data)
    if expected_type is list:
        assert all(len(row.embedding) == body.get("dimensions", 1536) for row in actual.data)


@pytest.mark.parametrize("case", ERROR_CASES, ids=lambda case: case["scenario"])
async def test_embeddings_recorded_error_sdk_parity(case: dict[str, Any]) -> None:
    """Ensures native provider failures keep the SDK exception and original response."""
    cassette = read_yaml(CASSETTE_DIR / case["file"])
    reference_client, _ = _client(cassette)
    async with reference_client:
        with pytest.raises(APIStatusError) as reference:
            await AsyncOpenAI(
                api_key="reference-key", http_client=reference_client, max_retries=0
            ).embeddings.create(**case["sdk_arguments"])

    http_client, requests = _client(cassette)
    async with Molto(http_client=http_client) as molto:
        with pytest.raises(APIStatusError) as actual:
            await molto.embeddings(case["sdk_arguments"], api_key="molto-key")
    assert len(requests) == 1
    assert type(actual.value) is type(reference.value)
    assert actual.value.status_code == reference.value.status_code
    assert actual.value.body == reference.value.body
    assert actual.value.message == reference.value.message
    assert actual.value.response.content == reference.value.response.content


def test_embeddings_public_schema_tracks_sdk() -> None:
    """Ensures the public request fields and native response alias match the SDK contract."""
    parameters = set(inspect.signature(AsyncEmbeddings.create).parameters)
    transport = {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
    assert set(OpenAIEmbeddingsCreateRequest.model_fields) == parameters - transport
    assert OpenAIEmbeddingsResponse is CreateEmbeddingResponse
    request = OpenAIEmbeddingsCreateRequest(model="example", input="text")
    assert request.model_dump() == {"model": "example", "input": "text"}


@pytest.mark.parametrize(
    "input_value",
    [True, 1, 1.5, [True], [1.5], [[False]], [[1.5]], [1, "text"], {"text": "value"}],
)
async def test_embeddings_rejects_coerced_inputs_before_http(input_value: object) -> None:
    """Ensures invalid token and text forms cannot be silently coerced into different inputs."""
    calls = []

    def unexpected(request: Request) -> Response:
        calls.append(request)
        raise AssertionError("Invalid input reached HTTP")

    async with Molto(http_client=AsyncClient(transport=MockTransport(unexpected))) as molto:
        with pytest.raises(ValidationError):
            await molto.embeddings({"model": "example", "input": input_value}, api_key="key")
    assert calls == []


@pytest.mark.parametrize("input_value", [[True], [1.5], [1, "text"], {"text": "value"}])
async def test_embeddings_rejects_invalid_extra_body_inputs_before_http(
    input_value: object,
) -> None:
    """Ensures transport overrides cannot bypass strict input semantics with a valid typed body."""
    async with Molto(
        http_client=AsyncClient(transport=MockTransport(lambda _: pytest.fail()))
    ) as molto:
        with pytest.raises(ValidationError):
            await molto.embeddings(
                {"model": "example", "input": "valid text"},
                api_key="key",
                extra_body={"input": input_value},
            )


@pytest.mark.parametrize("field", ["stream", "input_type", "truncate", "title", "unknown"])
async def test_embeddings_rejects_unmodeled_body_fields(field: str) -> None:
    """Ensures embedding options cannot disappear through a permissive request model."""
    async with Molto(
        http_client=AsyncClient(transport=MockTransport(lambda _: pytest.fail()))
    ) as molto:
        with pytest.raises(ValidationError):
            await molto.embeddings({"model": "example", "input": "text", field: True})


@pytest.mark.parametrize(
    ("endpoint", "provider"),
    [
        ("chat/completions", "openai"),
        ("embeddings", "cohere"),
        ("embeddings", "voyageai"),
        ("cohere/embed", "openai"),
        ("voyage/embeddings", "openai"),
        ("embeddings", "unknown"),
    ],
)
async def test_embeddings_rejects_incompatible_routes_before_http(
    endpoint: str, provider: str
) -> None:
    """Ensures provider selection never silently switches the declared wire protocol."""
    async with Molto(
        http_client=AsyncClient(transport=MockTransport(lambda _: pytest.fail()))
    ) as molto:
        with pytest.raises(ValueError):
            await molto.embeddings(
                {"model": "example", "input": "text"},
                endpoint=endpoint,
                provider=provider,
                api_key="key",
            )


@pytest.mark.parametrize(
    ("name", "encoding", "override"),
    [
        ("batch-float", "base64", "float"),
        ("batch-base64", "float", "base64"),
        ("batch-base64", None, "base64"),
    ],
)
async def test_embeddings_extra_body_retains_sdk_encoding_precedence(
    name: str, encoding: str | None, override: str
) -> None:
    """Ensures wire overrides and the SDK's explicit-versus-omitted decoding remain identical."""
    cassette = _recording(name)
    arguments = deepcopy(cassette["request"]["body"])
    arguments.pop("encoding_format")
    arguments["input"] = "This input is deliberately overridden."
    arguments["model"] = "deliberately-overridden-model"
    arguments["dimensions"] = 64
    if encoding is not None:
        arguments["encoding_format"] = encoding
    extra_body = {
        field: cassette["request"]["body"][field] for field in ("input", "model", "dimensions")
    }
    extra_body["encoding_format"] = override
    reference_client, _ = _client(cassette)
    async with reference_client:
        reference = await AsyncOpenAI(
            api_key="key", http_client=reference_client, max_retries=0
        ).embeddings.create(**arguments, extra_body=extra_body)
    http_client, _ = _client(cassette)
    async with Molto(http_client=http_client) as molto:
        actual = await molto.embeddings(arguments, api_key="key", extra_body=extra_body)
    assert actual.model_dump(mode="python", warnings=False) == reference.model_dump(
        mode="python", warnings=False
    )


async def test_embeddings_rejects_native_provider_options_on_openai_route() -> None:
    """Ensures a provider task option cannot be accepted on an incompatible native route."""
    async with Molto(
        http_client=AsyncClient(transport=MockTransport(lambda _: pytest.fail()))
    ) as molto:
        with pytest.raises(ValueError, match="options"):
            await molto.embeddings(
                {"model": "example", "input": "text"},
                api_key="key",
                options=CohereEmbeddingOptions(input_type="search_query"),
            )


async def test_embeddings_extra_body_token_tuple_is_one_input() -> None:
    """Ensures accepted token sequences remain one input when serialized by the SDK."""
    cassette = _recording("token-array-float")
    body = {**cassette["request"]["body"], "input": "overridden text"}
    extra_body = {"input": tuple(cassette["request"]["body"]["input"])}
    reference_client, _ = _client(cassette)
    async with reference_client:
        reference = await AsyncOpenAI(
            api_key="key", http_client=reference_client, max_retries=0
        ).embeddings.create(**body, extra_body=extra_body)
    http_client, _ = _client(cassette)
    async with Molto(http_client=http_client) as molto:
        actual = await molto.embeddings(body, api_key="key", extra_body=extra_body)
    assert actual.model_dump() == reference.model_dump()
    assert len(actual.data) == 1


async def test_embeddings_native_preserves_reordered_results_and_extensions() -> None:
    """Ensures a synthetic reordered native response retains input indices and opaque metadata."""
    cassette = _recording("batch-float")
    payload = deepcopy(cassette["response"]["body"])
    payload["data"].reverse()
    payload["data"][0]["provider_metadata"] = {"source": "second input"}
    payload["usage"]["billable_units"] = 7
    payload["provider_request_id"] = "synthetic-request"
    cassette["response"]["body_bytes"] = json.dumps(payload).encode()
    reference_client, _ = _client(cassette)
    async with reference_client:
        reference = await AsyncOpenAI(
            api_key="key", http_client=reference_client, max_retries=0
        ).embeddings.create(**cassette["request"]["body"])
    http_client, _ = _client(cassette)
    async with Molto(http_client=http_client) as molto:
        actual = await molto.embeddings(cassette["request"]["body"], api_key="key")
    assert actual.model_dump() == reference.model_dump()
    assert [item.index for item in actual.data] == [1, 0]
    assert actual.data[0].model_extra == {"provider_metadata": {"source": "second input"}}
    assert actual.usage.model_extra == {"billable_units": 7}


async def test_embeddings_credentials_headers_base_url_and_shared_lifecycle(monkeypatch) -> None:
    """Ensures requests reuse Molto's client while credentials, URL, timeout, and extras stay scoped."""
    monkeypatch.setattr(ENV_CONFIG, "openai_api_key", SecretStr("configured-key"))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.invalid/")
    monkeypatch.setenv("OPENAI_ORG_ID", "unrelated-organization")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated-project")
    cassette = _recording("batch-float")
    requests = []

    def replay(request: Request) -> Response:
        requests.append(request)
        assert request.extensions["timeout"] == {
            name: 7.5 for name in ("connect", "read", "write", "pool")
        }
        assert request.url == "https://example.test/custom/v1/embeddings"
        assert "openai-organization" not in request.headers
        assert "openai-project" not in request.headers
        for excluded in ("x-api-key", "cookie", "x-connection-token", "x-forwarded-for"):
            assert excluded not in request.headers
        return Response(200, json=cassette["response"]["body"], request=request)

    http_client = AsyncClient(transport=MockTransport(replay), timeout=7.5)
    async with Molto(http_client=http_client) as molto:
        await molto.embeddings(
            cassette["request"]["body"],
            base_url="https://example.test/custom/v1",
            extra_headers={
                "Authorization": "Bearer client-secret",
                "X-Api-Key": "client-secret",
                "Cookie": "session=client-secret",
                "Connection": "X-Connection-Token",
                "X-Connection-Token": "client-secret",
                "OpenAI-Organization": "client-organization",
                "X-Forwarded-For": "192.0.2.1",
                "X-Application": "first-call",
            },
        )
        assert not http_client.is_closed
        await molto.embeddings(
            cassette["request"]["body"],
            api_key="explicit-key",
            base_url="https://example.test/custom/v1/",
        )
        assert not http_client.is_closed
    assert http_client.is_closed
    assert len(requests) == 2
    assert requests[0].headers["authorization"] == "Bearer configured-key"
    assert requests[0].headers["x-application"] == "first-call"
    assert requests[1].headers["authorization"] == "Bearer explicit-key"
    assert "x-application" not in requests[1].headers


async def test_embeddings_missing_credentials_do_not_fall_back_to_sdk_environment(
    monkeypatch,
) -> None:
    """Ensures a missing resolved key fails locally even when the SDK environment contains a key."""
    monkeypatch.setattr(ENV_CONFIG, "openai_api_key", SecretStr(""))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-environment-key")
    async with Molto(
        http_client=AsyncClient(transport=MockTransport(lambda _: pytest.fail()))
    ) as molto:
        with pytest.raises(AuthenticationError):
            await molto.embeddings({"model": "example", "input": "text"})


async def test_embeddings_cancellation_does_not_close_shared_client() -> None:
    """Ensures cancelling one in-flight embedding request releases execution but preserves the client."""
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_for_cancellation(request: Request) -> Response:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("Unreachable")

    http_client = AsyncClient(transport=MockTransport(wait_for_cancellation))
    async with Molto(http_client=http_client) as molto:
        task = asyncio.create_task(
            molto.embeddings({"model": "example", "input": "text"}, api_key="key")
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert not http_client.is_closed
    assert http_client.is_closed
