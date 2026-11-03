"""Validated requests and SDK-native results for OpenAI embeddings."""

from typing import Literal

from openai.types import CreateEmbeddingResponse
from pydantic import StrictInt, StrictStr

from molto.types.base import ExcludeNoneModel


class OpenAIEmbeddingsCreateRequest(ExcludeNoneModel):
    """The body parameters accepted by `AsyncOpenAI.embeddings.create`."""

    input: StrictStr | list[StrictStr] | list[StrictInt] | list[list[StrictInt]]
    model: StrictStr
    dimensions: StrictInt | None = None
    encoding_format: Literal["float", "base64"] | None = None
    user: StrictStr | None = None


OpenAIEmbeddingsResponse = CreateEmbeddingResponse
