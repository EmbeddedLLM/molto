"""Project dense embedding output without inventing missing provider usage."""

from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from molto.types.embeddings import NormalizedEmbeddingResponse


def project_embeddings_response(response: NormalizedEmbeddingResponse) -> CreateEmbeddingResponse:
    """Return SDK models, retaining unavailable counts as None and native metadata."""
    return CreateEmbeddingResponse.model_construct(
        object="list",
        model=response.model,
        data=[
            Embedding.model_construct(
                object="embedding",
                index=item.index,
                embedding=item.embedding,
                **(
                    {"provider_metadata": item.provider_metadata} if item.provider_metadata else {}
                ),
            )
            for item in response.data
        ],
        usage=(
            Usage.model_construct(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            )
            if response.usage is not None
            else None
        ),
        provider_metadata=response.provider_metadata,
    )
