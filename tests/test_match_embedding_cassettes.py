from pathlib import Path

import pytest

from molto.utils.io import dump_yaml, read_yaml
from scripts.match_embedding_cassettes import main, match_embedding_cassette

MATCHED = Path(__file__).parent / "cassettes" / "embeddings" / "matched"
OPENAI = "openai-embeddings.yaml"
COHERE = "cohere-cohere_embed.yaml"
VOYAGE = "voyageai-voyage_embeddings.yaml"
PAIRS = [
    (path, path.parent / OPENAI)
    for path in sorted(MATCHED.glob("*/*.yaml"))
    if path.name != OPENAI
]


@pytest.mark.parametrize("path,openai_path", PAIRS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_committed_matched_embedding_cassettes_are_aligned(path, openai_path):
    """Ensures every committed Cohere and Voyage fixture already matches its OpenAI reference."""
    cassette = read_yaml(str(path))
    assert match_embedding_cassette(cassette, read_yaml(str(openai_path))) == cassette


def test_match_embedding_cassette_cli_restores_aligned_values(tmp_path):
    """Ensures the CLI rewrites a provider's model, vectors, and tokens to the OpenAI values."""
    scenario = MATCHED / "documents-float"
    raw = read_yaml(str(scenario / COHERE))
    raw["request"]["body"]["model"] = "embed-v4.0"
    vectors = raw["response"]["body"]["embeddings"]["float"]
    vectors[0] = list(reversed(vectors[0]))
    raw["response"]["body"]["meta"]["billed_units"]["input_tokens"] = 17
    source = tmp_path / "raw.yaml"
    output = tmp_path / "matched.yaml"
    dump_yaml(raw, str(source), sort_keys=True)
    assert main([str(source), str(scenario / OPENAI), "--output", str(output)]) == 0
    assert read_yaml(str(output)) == read_yaml(str(scenario / COHERE))


@pytest.mark.parametrize(
    "filename,field,value",
    [
        (COHERE, "texts", ["A different input."]),
        (COHERE, "output_dimension", 512),
        (COHERE, "embedding_types", ["float"]),
        (VOYAGE, "input", ["A different input."]),
        (VOYAGE, "output_dimension", 512),
        (VOYAGE, "encoding_format", None),
    ],
)
def test_match_embedding_cassette_rejects_non_equivalent_requests(filename, field, value):
    """Asserts that a pair with different inputs, dimensions, or encoding is never aligned."""
    scenario = MATCHED / "documents-base64"
    cassette = read_yaml(str(scenario / filename))
    cassette["request"]["body"][field] = value
    with pytest.raises(ValueError):
        match_embedding_cassette(cassette, read_yaml(str(scenario / OPENAI)))
