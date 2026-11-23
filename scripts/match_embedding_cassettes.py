#!/usr/bin/env python3
"""Align a Cohere or VoyageAI embedding cassette with its OpenAI Embeddings counterpart.

The OpenAI cassette is canonical. Independently generated models never return the same
vectors or token counts, so this script copies the representable values from the OpenAI
cassette: the requested model, the vectors, the response model, and the input token count.
Every other field, including provider IDs and transport metadata, stays as recorded.

The pair must already describe the same request: the same inputs in the same order,
the same output dimensions, and the same vector encoding. A mismatch is rejected,
because copying values between different requests would hide a translation error.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from molto.utils.io import dump_yaml, read_yaml


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def match_embedding_cassette(cassette: dict[str, Any], openai: dict[str, Any]) -> dict[str, Any]:
    """Return `cassette` with model, vectors, and token usage aligned to `openai`."""
    reference_request = openai["request"]["body"]
    reference = openai["response"]["body"]
    _check(openai["response"]["status_code"] == 200, "The OpenAI cassette must succeed.")
    _check(cassette["response"]["status_code"] == 200, "The aligned cassette must succeed.")
    vectors = [item["embedding"] for item in sorted(reference["data"], key=lambda i: i["index"])]
    usage = reference["usage"]
    _check(
        usage["prompt_tokens"] == usage["total_tokens"],
        "Embedding usage must have equal prompt and total tokens.",
    )
    tokens = usage["total_tokens"]

    matched = deepcopy(cassette)
    request = matched["request"]["body"]
    response = matched["response"]["body"]
    request["model"] = reference_request["model"]
    if "texts" in request:
        # Cohere v2 Embed returns one list per requested `embedding_types` entry.
        _check(request["texts"] == reference_request["input"], "Cohere texts differ.")
        _check(
            request.get("output_dimension") == reference_request.get("dimensions"),
            "Cohere output_dimension differs.",
        )
        (encoding,) = request["embedding_types"]
        _check(encoding == reference_request["encoding_format"], "Cohere encoding differs.")
        _check(
            len(response["embeddings"][encoding]) == len(vectors), "Cohere vector count differs."
        )
        response["embeddings"][encoding] = vectors
        billed = response["meta"]["billed_units"]
        billed["input_tokens"] = tokens
        if "text_tokens" in billed:
            billed["text_tokens"] = tokens
    else:
        _check(request["input"] == reference_request["input"], "Voyage input differs.")
        _check(
            request.get("output_dimension") == reference_request.get("dimensions"),
            "Voyage output_dimension differs.",
        )
        _check(
            request.get("encoding_format") == reference_request["encoding_format"],
            "Voyage encoding differs.",
        )
        items = sorted(response["data"], key=lambda item: item["index"])
        for item, vector in zip(items, vectors, strict=True):
            item["embedding"] = vector
        response["model"] = reference["model"]
        response["usage"]["total_tokens"] = tokens
    return matched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cassette", type=Path, help="Cohere or VoyageAI cassette to align.")
    parser.add_argument("openai_cassette", type=Path, help="Canonical OpenAI cassette.")
    parser.add_argument("--output", type=Path, help="Write to a different YAML file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matched = match_embedding_cassette(
        read_yaml(str(args.cassette)), read_yaml(str(args.openai_cassette))
    )
    dump_yaml(matched, str(args.output or args.cassette), sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
