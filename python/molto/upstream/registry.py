"""Explicit client selection for supported embedding provider/protocol pairs."""

from molto.providers import EmbeddingEndpoint, EmbeddingProvider
from molto.upstream.embeddings import (
    CohereEmbeddingsClient,
    InfinityEmbeddingsClient,
    OpenAIEmbeddingsClient,
    VLLMEmbeddingsClient,
    VoyageAIEmbeddingsClient,
)

EMBEDDING_CLIENTS: dict[
    tuple[EmbeddingProvider, EmbeddingEndpoint],
    type[OpenAIEmbeddingsClient | CohereEmbeddingsClient | VoyageAIEmbeddingsClient],
] = {
    ("openai", "embeddings"): OpenAIEmbeddingsClient,
    ("infinity", "embeddings"): InfinityEmbeddingsClient,
    ("vllm", "embeddings"): VLLMEmbeddingsClient,
    ("cohere", "cohere/embed"): CohereEmbeddingsClient,
    ("voyageai", "voyage/embeddings"): VoyageAIEmbeddingsClient,
}
