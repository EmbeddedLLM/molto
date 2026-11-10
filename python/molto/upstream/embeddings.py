"""Native OpenAI transport and text-only Cohere/Voyage embedding protocols."""

import base64
import binascii
import sys
from typing import Any, NoReturn

from openai import APIStatusError, AsyncOpenAI, OpenAIError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import NotFoundError as OpenAINotFoundError

from molto.types.embeddings import (
    CohereEmbeddingOptions,
    EmbeddingUsage,
    EmbeddingVector,
    NormalizedEmbeddingResponse,
    VoyageEmbeddingOptions,
)
from molto.types.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ResourceNotFoundError,
    UnsupportedInput,
)
from molto.types.openai_embeddings import (
    OpenAIEmbeddingsCreateRequest,
    OpenAIEmbeddingsResponse,
)
from molto.upstream.base import BaseHTTPClient


def _validate_dense_options(body: dict[str, Any]) -> None:
    """Reject known alternate representations that are not dense float32 vectors."""
    for field, allowed in (
        ("encoding_format", (None, "float", "base64")),
        ("output_dtype", (None, "float")),
        ("embed_dtype", (None, "float32")),
        ("endianness", (None, "native", sys.byteorder)),
    ):
        if body.get(field) not in allowed:
            raise UnsupportedInput(f"Unsupported embedding {field}: {body[field]!r}.")
    if body.get("embedding_types") not in (None, ["float"], ["base64"]):
        raise UnsupportedInput("Only one float or base64 embedding representation is supported.")
    if body.get("task") == "token_embed":
        raise UnsupportedInput("Token embeddings are not one dense vector per input.")
    if body.get("stream"):
        raise UnsupportedInput("Streaming embeddings are not supported.")
    for field in ("messages", "images", "inputs", "late_chunking_params"):
        if body.get(field) is not None:
            raise UnsupportedInput(f"Embedding {field} is not supported by the text contract.")


def _input_count(value: Any) -> int:
    if isinstance(value, str) or (isinstance(value, list) and value and type(value[0]) is int):
        return 1
    return len(value)


def _validate_vectors(
    vectors: list[Any], *, count: int, dimensions: int | None, encoding: str | None
) -> None:
    """Check correspondence and reject multi-vector output without reordering it."""
    if (
        len(vectors) != count
        or any(type(item.index) is not int for item in vectors)
        or {item.index for item in vectors} != set(range(count))
    ):
        raise ValueError("Embedding response indices do not match the requested inputs.")
    for item in vectors:
        vector = item.embedding
        if isinstance(vector, str):
            if encoding != "base64":
                raise ValueError("Expected a dense float embedding, received encoded output.")
            try:
                raw = base64.b64decode(vector, validate=True)
            except (ValueError, binascii.Error) as error:
                raise ValueError("Invalid base64 embedding.") from error
            if len(raw) % 4:
                raise ValueError("Base64 embedding is not a float32 vector.")
            size = len(raw) // 4
        elif isinstance(vector, list) and all(type(value) in (float, int) for value in vector):
            if encoding == "base64":
                raise ValueError("Expected a base64 embedding, received float output.")
            size = len(vector)
        else:
            raise ValueError("Embedding output must contain one dense vector per input.")
        if dimensions is not None and size != dimensions:
            raise ValueError("Embedding output dimensions do not match the request.")


class _EmbeddingsClient(BaseHTTPClient):
    PATH: str

    def _sdk_client(self) -> AsyncOpenAI:
        if not self._api_key:
            error = OpenAIError("Missing credentials. Supply an embedding provider API key.")
            raise AuthenticationError(
                str(error),
                response=None,
                body=None,
                request=self._make_authentication_request(self.PATH.lstrip("/")),
            ) from error
        return self._make_sdk_client()

    @staticmethod
    def _handle_error(error: Exception) -> NoReturn:
        if isinstance(error, APIStatusError):
            error_type = None
            if error.code == "context_length_exceeded":
                error_type = ContextLengthExceededError
            elif error.code == "model_not_found":
                error_type = ModelNotFoundError
            elif isinstance(error, OpenAIAuthenticationError):
                error_type = AuthenticationError
            elif isinstance(error, OpenAINotFoundError):
                error_type = ResourceNotFoundError
            if error_type is not None:
                raise error_type(
                    error.message, response=error.response, body=error.body
                ) from error
        raise error

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._sdk_client().post(
                self.PATH,
                body=body,
                cast_to=dict[str, Any],
                options={"headers": self._extra_headers or {}},
            )
        except Exception as error:
            self._handle_error(error)


class OpenAIEmbeddingsClient(_EmbeddingsClient):
    """Reuse the SDK's omitted-encoding decoder and native response models."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    PATH = "/embeddings"

    async def native_response(
        self, request: OpenAIEmbeddingsCreateRequest
    ) -> OpenAIEmbeddingsResponse:
        body = request.model_dump()
        effective_body = {**body, **(self._extra_body or {})}
        _validate_dense_options(effective_body)
        effective_request = OpenAIEmbeddingsCreateRequest.model_validate(
            {
                key: effective_body[key]
                for key in OpenAIEmbeddingsCreateRequest.model_fields
                if key in effective_body
            }
        )
        try:
            response = await self._sdk_client().embeddings.create(
                **body, extra_body=self._extra_body, extra_headers=self._extra_headers
            )
        except Exception as error:
            self._handle_error(error)
        # The SDK decodes omitted encoding on the caller's behalf. Validate the
        # parsed representation, preserving the native SDK object and extensions.
        _validate_vectors(
            response.data,
            count=_input_count(effective_request.input),
            dimensions=effective_request.dimensions,
            encoding=(
                effective_request.encoding_format if request.encoding_format is not None else None
            ),
        )
        return response


class InfinityEmbeddingsClient(OpenAIEmbeddingsClient):
    """Infinity embeddings using the shared OpenAI wire implementation."""

    REQUIRES_BASE_URL = True


class VLLMEmbeddingsClient(OpenAIEmbeddingsClient):
    """vLLM embeddings using the shared OpenAI wire implementation."""

    REQUIRES_BASE_URL = True


def _text_input(request: OpenAIEmbeddingsCreateRequest) -> str | list[str]:
    value = request.input
    if not isinstance(value, str) and not (
        isinstance(value, list) and all(isinstance(item, str) for item in value)
    ):
        raise UnsupportedInput("This embedding protocol accepts text, not token IDs.")
    if request.user is not None:
        raise UnsupportedInput("This embedding protocol cannot preserve the OpenAI user field.")
    return value


class CohereEmbeddingsClient(_EmbeddingsClient):
    DEFAULT_BASE_URL = "https://api.cohere.com/v2"
    PATH = "/embed"

    async def response(
        self, request: OpenAIEmbeddingsCreateRequest, options: CohereEmbeddingOptions | None
    ) -> NormalizedEmbeddingResponse:
        value = _text_input(request)
        if options is None:
            raise UnsupportedInput("Cohere requires explicit CohereEmbeddingOptions.input_type.")
        body: dict[str, Any] = {
            "model": request.model,
            "texts": [value] if isinstance(value, str) else value,
            "embedding_types": [request.encoding_format or "float"],
            **options.model_dump(),
        }
        if request.dimensions is not None:
            body["output_dimension"] = request.dimensions
        body.update(self._extra_body or {})
        _validate_dense_options(body)
        texts = body["texts"]
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise UnsupportedInput("Cohere texts must remain a list of strings.")
        encoding = (body.get("embedding_types") or ["float"])[0]
        result = await self._post(body)
        representations = {"float", "base64", "int8", "uint8", "binary", "ubinary"}
        if any(
            key in representations and key != encoding and value is not None
            for key, value in result["embeddings"].items()
        ):
            raise ValueError("Cohere returned multiple embedding representations.")
        data = [
            EmbeddingVector(index=index, embedding=vector)
            for index, vector in enumerate(result["embeddings"][encoding])
        ]
        _validate_vectors(
            data, count=len(texts), dimensions=body.get("output_dimension"), encoding=encoding
        )
        # Billed units describe billing, not actual prompt/total token usage.
        metadata = {key: value for key, value in result.items() if key != "embeddings"}
        embedding_metadata = {
            key: value for key, value in result["embeddings"].items() if key != encoding
        }
        if embedding_metadata:
            metadata["embeddings"] = embedding_metadata
        return NormalizedEmbeddingResponse(
            model=body["model"],
            data=data,
            provider_metadata=metadata,
        )


class VoyageAIEmbeddingsClient(_EmbeddingsClient):
    DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
    PATH = "/embeddings"

    async def response(
        self, request: OpenAIEmbeddingsCreateRequest, options: VoyageEmbeddingOptions | None
    ) -> NormalizedEmbeddingResponse:
        body: dict[str, Any] = {
            "model": request.model,
            "input": _text_input(request),
            "output_dtype": "float",
            **(options.model_dump() if options is not None else {}),
        }
        if request.dimensions is not None:
            body["output_dimension"] = request.dimensions
        # Voyage uses an omitted/null encoding for JSON floats, not "float".
        if request.encoding_format == "base64":
            body["encoding_format"] = "base64"
        body.update(self._extra_body or {})
        _validate_dense_options(body)
        if body.get("encoding_format") == "float":
            raise UnsupportedInput("Voyage encoding_format uses null for float output.")
        value = body["input"]
        if not isinstance(value, str) and not (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        ):
            raise UnsupportedInput("Voyage input must remain text or a list of texts.")
        result = await self._post(body)
        data = [
            EmbeddingVector(
                index=item["index"],
                embedding=item["embedding"],
                provider_metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {"index", "embedding", "object"}
                },
            )
            for item in result["data"]
        ]
        _validate_vectors(
            data,
            count=_input_count(value),
            dimensions=body.get("output_dimension"),
            encoding=body.get("encoding_format"),
        )
        usage = result.get("usage")
        return NormalizedEmbeddingResponse(
            model=result.get("model", body["model"]),
            data=data,
            usage=(
                EmbeddingUsage(
                    prompt_tokens=usage.get("prompt_tokens"),
                    total_tokens=usage.get("total_tokens"),
                )
                if usage is not None
                else None
            ),
            provider_metadata={key: value for key, value in result.items() if key != "data"},
        )
