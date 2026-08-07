"""Concepts shared verbatim across the OpenAI Chat Completions and Responses API models.

Only genuinely identical definitions belong here. The two endpoints deliberately
differ in wire shape (nested vs flat formats, tool nesting, optionality), so most
of their models stay endpoint-specific; this module holds the rare exact twins so
a supported-value change is one edit, not two.
"""

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self, TypeAlias, cast, get_args

from pydantic import BaseModel, Field, field_validator

from molto.utils.exceptions import BadInputError

OpenAIProviderName: TypeAlias = Literal["openai", "vllm", "anthropic"]


class OpenAIRequestBase(BaseModel):
    """Shared Pydantic behavior for complete OpenAI-compatible requests."""

    model: Annotated[
        str,
        Field(description="The provider-qualified model used to generate the response."),
    ]

    @staticmethod
    def provider_from_model(model: str) -> OpenAIProviderName:
        """Return the provider encoded in a provider-qualified model name."""

        provider, separator, local_model = model.partition("/")
        if not separator or not local_model:
            raise ValueError("model must include a provider prefix.")
        if provider not in get_args(OpenAIProviderName):
            raise ValueError("Request model has an unsupported provider prefix.")
        return cast(OpenAIProviderName, provider)

    @field_validator("model", check_fields=False)
    @classmethod
    def validate_model(cls, value: str) -> str:
        cls.provider_from_model(value)
        return value

    @property
    def provider(self) -> OpenAIProviderName:
        return self.provider_from_model(self.model)

    @classmethod
    def from_body(
        cls,
        body: Any,
        model: str | None,
    ) -> Self:
        if isinstance(body, Mapping):
            if model is None and "model" not in body:
                raise BadInputError("model is required when body has no model.")
            if model is not None and "model" in body and body["model"] != model:
                raise BadInputError("The body model must match the model argument.")
            request_data = {**body, "model": model} if model is not None else body
            return cls.model_validate(request_data)

        if isinstance(body, cls):
            if model is not None and body.model != model:
                raise BadInputError("The body model must match the model argument.")
            return body

        if isinstance(body, BaseModel):
            raise BadInputError(f"{cls.__name__} body must use {cls.__name__}.")
        raise BadInputError(f"{cls.__name__} body must be its request model or a mapping.")


# Reasoning effort levels accepted by both endpoints' request models.
OpenAIReasoningEffortLevel: TypeAlias = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh"
]
