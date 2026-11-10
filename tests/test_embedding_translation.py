"""Public embedding translation against faithful recordings and explicit synthetic cases."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import yaml
from openai import AuthenticationError, NotFoundError
from pydantic import SecretStr

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.types.embeddings import CohereEmbeddingOptions, VoyageEmbeddingOptions
from molto.types.exceptions import APIException, UnsupportedInput

RECORDED = Path(__file__).parent / "cassettes" / "embeddings" / "recorded"


def _recording(provider, scenario):
    return yaml.safe_load((RECORDED / provider / f"{scenario}.yaml").read_text())


def _replay(recording, seen):
    def respond(request):
        seen.append(request)
        assert str(request.url) == recording["request"]["url"]
        assert json.loads(request.content) == recording["request"]["body"]
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            recording["response"]["status_code"],
            headers=recording["response"]["headers"],
            content=recording["response"]["body_bytes"],
        )

    return httpx.MockTransport(respond)


@pytest.mark.parametrize("provider", ["cohere", "voyageai"])
def test_native_recordings_unchanged(provider):
    """Ensures replay evidence remains byte-for-byte identical to its reviewed manifest."""
    manifest = yaml.safe_load((RECORDED / provider / "manifest.yaml").read_text())
    for entry in manifest["fixtures"]:
        path = RECORDED / provider / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        record = yaml.safe_load(path.read_text())
        assert json.loads(record["response"]["body_bytes"]) == record["response"]["body"]


@pytest.mark.parametrize(
    ("scenario", "encoding"),
    [
        (scenario, encoding)
        for scenario in ["single-query", "batch-document", "classification", "clustering"]
        for encoding in [None, "float"]
    ]
    + [("batch-base64", "base64")],
)
async def test_cohere_recorded_public_translation(scenario, encoding):
    """Ensures Cohere's native v2 schema preserves task, vectors, order, dimensions, and metadata."""
    cassette = _recording("cohere", scenario)
    wire = cassette["request"]["body"]
    actual_encoding = wire["embedding_types"][0]
    body = {
        "model": wire["model"],
        "input": wire["texts"][0] if scenario == "single-query" else wire["texts"],
        "dimensions": wire["output_dimension"],
    }
    if encoding is not None:
        body["encoding_format"] = encoding
    seen = []
    client = httpx.AsyncClient(transport=_replay(cassette, seen))
    async with Molto(client) as molto:
        result = await molto.embeddings(
            body,
            endpoint="cohere/embed",
            api_key="test-key",
            options=CohereEmbeddingOptions(
                input_type=wire["input_type"], truncate=wire["truncate"]
            ),
        )
        assert not client.is_closed
    assert client.is_closed
    native = cassette["response"]["body"]
    assert result.object == "list"
    assert result.model == wire["model"]
    assert [item.index for item in result.data] == list(range(len(wire["texts"])))
    assert [item.embedding for item in result.data] == native["embeddings"][actual_encoding]
    assert result.usage is None
    assert result.provider_metadata == {
        key: value for key, value in native.items() if key != "embeddings"
    }
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("provider", "scenario", "error_type"),
    [("cohere", "invalid-model", NotFoundError)]
    + [
        ("voyageai", name, AuthenticationError)
        for name in (
            "single-default",
            "single-query",
            "batch-document",
            "batch-base64",
            "invalid-model",
        )
    ],
)
async def test_recorded_native_errors(provider, scenario, error_type):
    """Ensures actual Cohere 404 and Voyage 401 bodies survive the shared error cause chain."""
    cassette = _recording(provider, scenario)
    wire = cassette["request"]["body"]
    cohere = provider == "cohere"
    body = {"model": wire["model"], "input": wire["texts"] if cohere else wire["input"]}
    if "output_dimension" in wire:
        body["dimensions"] = wire["output_dimension"]
    if "encoding_format" in wire:
        body["encoding_format"] = wire["encoding_format"]
    options = (
        CohereEmbeddingOptions(input_type=wire["input_type"])
        if cohere
        else VoyageEmbeddingOptions.model_validate(
            {key: wire[key] for key in ("input_type", "truncation") if key in wire}
        )
    )
    seen = []
    async with Molto(httpx.AsyncClient(transport=_replay(cassette, seen))) as molto:
        with pytest.raises(error_type) as caught:
            await molto.embeddings(
                body,
                endpoint="cohere/embed" if cohere else "voyage/embeddings",
                api_key="test-key",
                options=options,
            )
    assert caught.value.response.json() == cassette["response"]["body"]
    assert isinstance(caught.value.__cause__, APIException)
    assert len(seen) == 1


@pytest.mark.parametrize("encoding", [None, "float", "base64"])
@pytest.mark.parametrize("input_type", [None, "query", "document"])
async def test_voyage_schema_translation_synthetic(encoding, input_type):
    """Ensures Voyage's documented schema preserves reordered vectors without invented prompt usage.

    This is deliberate synthetic schema coverage, not a successful live Voyage
    capture: https://docs.voyageai.com/reference/embeddings-api.
    """
    name = f"{input_type or 'default'}-{'base64' if encoding == 'base64' else 'float'}"
    fixture = yaml.safe_load(
        (RECORDED.parent / "synthetic" / "voyageai" / f"{name}.yaml").read_text()
    )
    native = fixture["response"]["body"]
    expected = fixture["request"]["body"]
    seen = []

    def respond(request):
        seen.append(request)
        assert request.url == "https://example.test/prefix/embeddings"
        assert json.loads(request.content) == expected
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["x-request-note"] == "trace"
        assert "openai-organization" not in request.headers
        return httpx.Response(200, json=native)

    body = {
        "model": expected["model"],
        "input": expected["input"],
        "dimensions": expected["output_dimension"],
    }
    if encoding is not None:
        body["encoding_format"] = encoding
    async with Molto(httpx.AsyncClient(transport=httpx.MockTransport(respond))) as molto:
        result = await molto.embeddings(
            body,
            endpoint="voyage/embeddings",
            api_key="test-key",
            base_url="https://example.test/prefix",
            options=VoyageEmbeddingOptions(input_type=input_type, truncation=False),
            extra_headers={
                "Authorization": "unsafe",
                "OpenAI-Organization": "unsafe",
                "x-request-note": "trace",
            },
        )
    assert [item.index for item in result.data] == [1, 0]
    assert [item.embedding for item in result.data] == [
        item["embedding"] for item in native["data"]
    ]
    assert [item.provider_metadata for item in result.data] == [{"trace": 1}, {"trace": 0}]
    assert result.usage.prompt_tokens is None
    assert result.usage.total_tokens == 7
    assert result.provider_metadata == {
        key: value for key, value in native.items() if key != "data"
    }
    assert len(seen) == 1


@pytest.mark.parametrize(
    "endpoint,field", [("cohere/embed", "cohere_api_key"), ("voyage/embeddings", "voyage_api_key")]
)
async def test_hosted_missing_credentials_do_not_use_openai(endpoint, field, monkeypatch):
    """Ensures an absent native key never leaks an ambient OpenAI key to another service."""
    monkeypatch.setattr(ENV_CONFIG, field, SecretStr(""))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    def respond(request):
        pytest.fail("No request may be made without the selected provider credential.")

    async with Molto(httpx.AsyncClient(transport=httpx.MockTransport(respond))) as molto:
        with pytest.raises(AuthenticationError):
            await molto.embeddings(
                {"model": "model", "input": "hello"},
                endpoint=endpoint,
                options=CohereEmbeddingOptions(input_type="search_query")
                if endpoint == "cohere/embed"
                else None,
            )


@pytest.mark.parametrize("endpoint", ["cohere/embed", "voyage/embeddings"])
@pytest.mark.parametrize("change", [{"input": [1, 2]}, {"input": [[1], [2]]}, {"user": "user"}])
async def test_unsupported_translation_is_local(endpoint, change):
    """Ensures token IDs and OpenAI user metadata cannot silently disappear in native translation."""

    def respond(request):
        pytest.fail("Unsupported translation must be rejected before HTTP.")

    async with Molto(httpx.AsyncClient(transport=httpx.MockTransport(respond))) as molto:
        with pytest.raises(UnsupportedInput):
            await molto.embeddings(
                {"model": "model", "input": "hello", **change},
                endpoint=endpoint,
                api_key="test-key",
                options=CohereEmbeddingOptions(input_type="search_query")
                if endpoint == "cohere/embed"
                else None,
            )


@pytest.mark.parametrize(
    "endpoint,options",
    [
        ("cohere/embed", None),
        ("cohere/embed", VoyageEmbeddingOptions()),
        ("voyage/embeddings", CohereEmbeddingOptions(input_type="search_query")),
    ],
)
async def test_native_task_options_are_not_interchangeable(endpoint, options):
    """Ensures Cohere requires intent and options from another protocol cannot be silently reused."""
    async with Molto(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("HTTP")))
    ) as molto:
        with pytest.raises(UnsupportedInput):
            await molto.embeddings(
                {"model": "model", "input": "hello"},
                endpoint=endpoint,
                api_key="test-key",
                options=options,
            )


@pytest.mark.parametrize("provider", ["cohere", "voyageai"])
async def test_native_extra_body_overrides_are_applied_before_output_projection(provider):
    """Ensures upstream overrides control model, inputs, dimensions, encoding, and task together."""
    cohere = provider == "cohere"
    fixture = (
        _recording(provider, "batch-base64")
        if cohere
        else yaml.safe_load(
            (RECORDED.parent / "synthetic" / provider / "document-base64.yaml").read_text()
        )
    )
    wire = fixture["request"]["body"]

    def respond(request):
        assert json.loads(request.content) == wire
        return httpx.Response(200, json=fixture["response"]["body"])

    async with Molto(httpx.AsyncClient(transport=httpx.MockTransport(respond))) as molto:
        result = await molto.embeddings(
            {
                "model": "overridden",
                "input": "overridden",
                "dimensions": 1,
                "encoding_format": "float",
            },
            endpoint="cohere/embed" if cohere else "voyage/embeddings",
            api_key="test-key",
            options=CohereEmbeddingOptions(input_type="search_query")
            if cohere
            else VoyageEmbeddingOptions(input_type="query"),
            extra_body=wire,
        )
    assert result.model == wire["model"]
    assert len(result.data) == 2
    assert all(isinstance(item.embedding, str) for item in result.data)


@pytest.mark.parametrize("corruption", ["missing", "nested", "dimensions", "multiple"])
async def test_cohere_malformed_output_rejects_without_coercion(corruption):
    """Ensures incomplete, multi-vector, wrong-sized, or quantized responses cannot look valid."""
    fixture = _recording("cohere", "batch-document")
    native = fixture["response"]["body"]
    if corruption == "missing":
        native["embeddings"]["float"].pop()
    elif corruption == "nested":
        native["embeddings"]["float"][0] = [[0.5], [0.25]]
    elif corruption == "dimensions":
        native["embeddings"]["float"][0].pop()
    else:
        native["embeddings"]["int8"] = [[1, 2], [3, 4]]
    wire = fixture["request"]["body"]
    async with Molto(
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=native))
        )
    ) as molto:
        with pytest.raises(ValueError):
            await molto.embeddings(
                {
                    "model": wire["model"],
                    "input": wire["texts"],
                    "dimensions": wire["output_dimension"],
                },
                endpoint="cohere/embed",
                api_key="test-key",
                options=CohereEmbeddingOptions(input_type=wire["input_type"]),
            )


async def test_cohere_preserves_nested_embedding_metadata():
    """Ensures unfamiliar metadata alongside embeddings is retained without accepting another dtype."""
    fixture = _recording("cohere", "single-query")
    native = fixture["response"]["body"]
    native["embeddings"]["provider_revision"] = "synthetic-extension"
    native["embeddings"]["int8"] = None
    wire = fixture["request"]["body"]
    async with Molto(
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=native))
        )
    ) as molto:
        result = await molto.embeddings(
            {"model": wire["model"], "input": wire["texts"]},
            endpoint="cohere/embed",
            api_key="test-key",
            options=CohereEmbeddingOptions(input_type=wire["input_type"]),
        )
    assert result.provider_metadata["embeddings"] == {
        "provider_revision": "synthetic-extension",
        "int8": None,
    }


@pytest.mark.parametrize("indices", [[0, 0], [1, 2], [False, 1], ["0", 1]])
async def test_voyage_invalid_indices_cannot_misassociate_inputs(indices):
    """Ensures duplicate, missing, or non-integer native indices never associate vectors incorrectly."""
    native = {"data": [{"index": index, "embedding": [0.5]} for index in indices]}
    async with Molto(
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=native))
        )
    ) as molto:
        with pytest.raises(ValueError):
            await molto.embeddings(
                {"model": "voyage-3.5-lite", "input": ["first", "second"]},
                endpoint="voyage/embeddings",
                api_key="test-key",
            )


@pytest.mark.parametrize("endpoint", ["cohere/embed", "voyage/embeddings"])
@pytest.mark.parametrize(
    "override",
    [
        {"output_dtype": "int8"},
        {"embedding_types": ["float", "binary"]},
        {"inputs": [{"content": [{"type": "text", "text": "grouped"}]}]},
        {"images": ["data:image/png;base64,AAAA"]},
        {"stream": True},
    ],
)
async def test_unsupported_native_overrides_fail_before_http(endpoint, override):
    """Ensures raw transport options cannot turn dense text output into an unsupported mode."""
    async with Molto(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("HTTP")))
    ) as molto:
        with pytest.raises(UnsupportedInput):
            await molto.embeddings(
                {"model": "model", "input": "text"},
                endpoint=endpoint,
                api_key="test-key",
                options=CohereEmbeddingOptions(input_type="search_query")
                if endpoint == "cohere/embed"
                else None,
                extra_body=override,
            )
