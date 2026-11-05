"""Embedding-specific options and normalized dense output.

Task names and truncation controls remain protocol-specific because their
semantics are not interchangeable between providers.
"""

from typing import Any, Literal

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr

from molto.types.base import StrictModel


class CohereEmbeddingOptions(StrictModel):
    input_type: Literal["search_query", "search_document", "classification", "clustering"]
    truncate: Literal["NONE", "START", "END"] | None = None


class VoyageEmbeddingOptions(StrictModel):
    input_type: Literal["query", "document"] | None = None
    truncation: StrictBool | None = None


class EmbeddingVector(StrictModel):
    index: StrictInt
    embedding: list[StrictFloat] | StrictStr
    provider_metadata: dict[str, Any] = {}


class EmbeddingUsage(StrictModel):
    prompt_tokens: StrictInt | None = None
    total_tokens: StrictInt | None = None


class NormalizedEmbeddingResponse(StrictModel):
    model: StrictStr
    data: list[EmbeddingVector]
    usage: EmbeddingUsage | None = None
    provider_metadata: dict[str, Any] = {}
