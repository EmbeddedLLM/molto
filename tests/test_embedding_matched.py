"""Public embedding routes against matched cassettes, with OpenAI Embeddings as the reference.

Each scenario holds an OpenAI cassette and aligned Cohere and VoyageAI cassettes.
The OpenAI request drives every route. The expected upstream request and the expected
result both come from the cassettes, independent of Molto's translation code.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI

from molto import Molto
from molto.types.embeddings import CohereEmbeddingOptions, VoyageEmbeddingOptions
from molto.utils.io import read_yaml

MATCHED = Path(__file__).parent / "cassettes" / "embeddings" / "matched"
SCENARIOS = sorted(path.name for path in MATCHED.iterdir())
ROUTES = {
    "cohere-cohere_embed.yaml": "cohere/embed",
    "voyageai-voyage_embeddings.yaml": "voyage/embeddings",
}


def _client(cassette: dict[str, Any], seen: list[Request]) -> AsyncClient:
    def replay(request: Request) -> Response:
        seen.append(request)
        assert request.url.path == cassette["request"]["path"]
        assert json.loads(request.content) == cassette["request"]["body"]
        return Response(
            cassette["response"]["status_code"],
            headers={"content-type": cassette["response"]["headers"]["content-type"]},
            json=cassette["response"]["body"],
        )

    return AsyncClient(transport=MockTransport(replay))


async def _reference(openai: dict[str, Any]) -> dict[str, Any]:
    async with _client(openai, []) as http_client:
        parsed = await AsyncOpenAI(
            api_key="key", http_client=http_client, max_retries=0
        ).embeddings.create(**openai["request"]["body"])
    return parsed.model_dump()


def _options(endpoint: str, wire: dict[str, Any]):
    if endpoint == "cohere/embed":
        return CohereEmbeddingOptions(input_type=wire["input_type"], truncate=wire["truncate"])
    return VoyageEmbeddingOptions(input_type=wire["input_type"], truncation=wire["truncation"])


def _without_provider_metadata(result: dict[str, Any]) -> dict[str, Any]:
    result.pop("provider_metadata", None)
    for item in result["data"]:
        item.pop("provider_metadata", None)
    return result


@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_matched_openai_embeddings_native_parity(scenario):
    """Ensures the native OpenAI route sends the recorded request and returns the SDK result."""
    openai = read_yaml(str(MATCHED / scenario / "openai-embeddings.yaml"))
    seen = []
    async with Molto(http_client=_client(openai, seen)) as molto:
        actual = await molto.embeddings(openai["request"]["body"], api_key="key")
    assert actual.model_dump() == await _reference(openai)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "scenario,filename",
    [
        (scenario, filename)
        for scenario in SCENARIOS
        for filename in ROUTES
        if (MATCHED / scenario / filename).exists()
    ],
)
async def test_matched_embeddings_translation(scenario, filename):
    """Ensures an OpenAI request reaches Cohere or Voyage exactly and returns the OpenAI result.

    The aligned provider cassette fixes the expected upstream request and response values.
    Token usage maps the provider's input token count to both prompt and total tokens.
    """
    openai = read_yaml(str(MATCHED / scenario / "openai-embeddings.yaml"))
    cassette = read_yaml(str(MATCHED / scenario / filename))
    endpoint = ROUTES[filename]
    seen = []
    async with Molto(http_client=_client(cassette, seen)) as molto:
        actual = await molto.embeddings(
            openai["request"]["body"],
            endpoint=endpoint,
            api_key="key",
            options=_options(endpoint, cassette["request"]["body"]),
        )
    assert _without_provider_metadata(actual.model_dump()) == await _reference(openai)
    assert len(seen) == 1
