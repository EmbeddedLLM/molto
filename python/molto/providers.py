from dataclasses import dataclass
from typing import Literal, TypeAlias

Endpoint: TypeAlias = Literal["responses", "chat/completions", "messages"]
Provider: TypeAlias = Literal["anthropic", "openai", "vllm"]
EmbeddingEndpoint: TypeAlias = Literal["embeddings", "cohere/embed", "voyage/embeddings"]
EmbeddingProvider: TypeAlias = Literal["openai", "cohere", "voyageai", "infinity", "vllm"]
ReasoningRepresentation: TypeAlias = Literal["raw", "summary"]

DEFAULT_EMBEDDING_PROVIDER: dict[EmbeddingEndpoint, EmbeddingProvider] = {
    "embeddings": "openai",
    "cohere/embed": "cohere",
    "voyage/embeddings": "voyageai",
}
EMBEDDING_PROVIDERS: dict[EmbeddingEndpoint, frozenset[EmbeddingProvider]] = {
    "embeddings": frozenset({"openai", "infinity", "vllm"}),
    "cohere/embed": frozenset({"cohere"}),
    "voyage/embeddings": frozenset({"voyageai"}),
}


@dataclass(frozen=True)
class ProviderCapabilities:
    thinking: ReasoningRepresentation


PROVIDER_CAPABILITIES: dict[Provider, ProviderCapabilities] = {
    "anthropic": ProviderCapabilities(thinking="summary"),
    "openai": ProviderCapabilities(thinking="summary"),
    "vllm": ProviderCapabilities(thinking="raw"),
}

DEFAULT_PROVIDER_BY_ENDPOINT: dict[Endpoint, Provider] = {
    "responses": "openai",
    "chat/completions": "openai",
    "messages": "anthropic",
}
