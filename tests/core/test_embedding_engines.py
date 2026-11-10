"""Schema-based engine replay; these fixtures are not live engine recordings."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI, BadRequestError
from pydantic import ValidationError

from molto import Molto
from molto.types.exceptions import UnsupportedInput
from molto.upstream.embeddings import InfinityEmbeddingsClient, VLLMEmbeddingsClient

FIXTURES = Path(__file__).parents[1] / "cassettes" / "embeddings" / "synthetic"
ENGINES = ("infinity", "vllm")
CASES = [
    (engine, case)
    for engine in ENGINES
    for case in json.loads((FIXTURES / engine / "manifest.json").read_text())["cases"]
]


@pytest.mark.parametrize("engine,case", CASES, ids=[f"{e}-{c['file']}" for e, c in CASES])
@pytest.mark.parametrize("api_key", [None, "engine-test-key"])
async def test_engine_sdk_parity(engine, case, api_key):
    """Ensures engine requests and parsed responses match the SDK for each input form."""
    cassette = json.loads((FIXTURES / engine / case["file"]).read_text())
    requests = []
    resolved_key = api_key or "EMPTY"

    def handle(request: Request) -> Response:
        requests.append(request)
        assert request.method == cassette["request"]["method"]
        assert str(request.url) == "https://engine.example/v1/embeddings"
        assert request.headers["authorization"] == f"Bearer {resolved_key}"
        assert json.loads(request.content) == cassette["request"]["body"]
        response = cassette["response"]
        return Response(
            response["status_code"], headers=response["headers"], json=response["body"]
        )

    async with AsyncClient(transport=MockTransport(handle)) as reference_http:
        sdk = AsyncOpenAI(
            api_key=resolved_key,
            base_url="https://engine.example/v1",
            http_client=reference_http,
            max_retries=0,
        )
        if cassette["response"]["status_code"] >= 400:
            with pytest.raises(BadRequestError) as reference_error:
                await sdk.embeddings.create(**case["body"], extra_body=case["extra_body"])
        else:
            expected = await sdk.embeddings.create(**case["body"], extra_body=case["extra_body"])

    original_body = deepcopy(case["body"])
    async with Molto(http_client=AsyncClient(transport=MockTransport(handle))) as molto:
        kwargs = dict(
            provider=engine,
            api_key=api_key,
            base_url="https://engine.example/v1",
            extra_body=case["extra_body"],
        )
        if cassette["response"]["status_code"] >= 400:
            with pytest.raises(BadRequestError) as actual_error:
                await molto.embeddings(case["body"], **kwargs)
            assert actual_error.value.status_code == reference_error.value.status_code
            assert actual_error.value.body == reference_error.value.body
            assert actual_error.value.response.content == reference_error.value.response.content
        else:
            actual = await molto.embeddings(case["body"], **kwargs)
            assert type(actual) is type(expected)
            assert actual.model_dump() == expected.model_dump()
            assert [item.index for item in actual.data] == list(range(len(actual.data)))
    assert case["body"] == original_body
    assert len(requests) == 2


@pytest.mark.parametrize("engine", ENGINES)
async def test_engine_requires_explicit_destination(engine):
    """Ensures an engine credential cannot be sent to a hosted default destination."""

    def unexpected_request(request: Request) -> Response:
        pytest.fail("Missing engine URL must fail before HTTP")

    async with Molto(
        http_client=AsyncClient(transport=MockTransport(unexpected_request))
    ) as molto:
        with pytest.raises(ValueError, match="base_url"):
            await molto.embeddings({"model": "test", "input": "text"}, provider=engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "extra_body",
    [
        {"encoding_format": "tensor"},
        {"encoding_format": "bytes"},
        {"encoding_format": "bytes_only"},
        {"embed_dtype": "float16"},
        {"endianness": "big"},
        {"messages": [{"role": "user", "content": "text"}]},
        {"late_chunking_params": {"chunk_size": 2}},
        {"input": [{"role": "user", "content": "text"}]},
    ],
)
async def test_engine_rejects_non_dense_protocol_modes(engine, extra_body):
    """Ensures unsupported tensor, multimodal and contextual modes fail without HTTP."""

    def unexpected_request(request: Request) -> Response:
        pytest.fail("Unsupported modes must fail before HTTP")

    async with Molto(
        http_client=AsyncClient(transport=MockTransport(unexpected_request))
    ) as molto:
        with pytest.raises((UnsupportedInput, ValidationError)):
            await molto.embeddings(
                {"model": "test", "input": "text"},
                provider=engine,
                base_url="https://engine.example/v1",
                extra_body=extra_body,
            )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "embedding",
    [
        [[0.25, 0.5], [0.75, 1.0]],
        {"shape": [2], "dtype": "float32", "data_encoding": "base64", "data": "AACAPwAAAEA="},
        {"indices": [1, 3], "values": [0.25, 0.75]},
    ],
)
async def test_engine_rejects_non_dense_vectors(engine, embedding):
    """Ensures server token matrices, tensors and sparse data are never flattened to dense vectors."""

    def handle(request: Request) -> Response:
        return Response(
            200,
            json={
                "object": "list",
                "model": "test",
                "data": [{"index": 0, "object": "embedding", "embedding": embedding}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    async with Molto(http_client=AsyncClient(transport=MockTransport(handle))) as molto:
        with pytest.raises(ValueError):
            await molto.embeddings(
                {"model": "test", "input": "text", "encoding_format": "float"},
                provider=engine,
                base_url="https://engine.example/v1",
            )


@pytest.mark.parametrize("engine", ENGINES)
async def test_engine_preserves_response_order_and_indices(engine):
    """Ensures reordered native results retain their indices and extensions without reassociation."""
    cassette = json.loads((FIXTURES / engine / "batch-float.json").read_text())
    response_body = cassette["response"]["body"]
    response_body["data"].reverse()
    response_body["server_extension"] = {"revision": "test"}
    response_body["data"][0]["item_extension"] = "retained"

    def handle(request: Request) -> Response:
        assert json.loads(request.content) == cassette["request"]["body"]
        return Response(200, json=response_body)

    async with Molto(http_client=AsyncClient(transport=MockTransport(handle))) as molto:
        result = await molto.embeddings(
            cassette["request"]["body"],
            provider=engine,
            base_url="https://engine.example/v1",
        )
    assert [item.index for item in result.data] == [1, 0]
    assert result.model_dump() == response_body


@pytest.mark.parametrize("client_type", [InfinityEmbeddingsClient, VLLMEmbeddingsClient])
async def test_engine_client_requires_explicit_destination(client_type):
    """Ensures self-hosted destination policy belongs to each client, even outside Molto."""
    async with AsyncClient() as http_client:
        with pytest.raises(UnsupportedInput, match="explicit base_url"):
            client_type(http_client, api_key="engine-test-key")
