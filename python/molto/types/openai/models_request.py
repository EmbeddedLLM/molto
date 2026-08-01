"""Unified public OpenAI request model.

The two OpenAI endpoints retain distinct wire schemas, but callers use one
request envelope. Its explicit ``native_endpoint`` selects the endpoint while
the typed payload preserves the exact endpoint contract before conversion to
the portable request IR.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from molto.types.ir.portable import InputIR

from .chat_completions.models_request import OpenAIChatCompletionRequest
from .responses.models_request import OpenAIResponsesRequest

OpenAIRequestPayload = OpenAIChatCompletionRequest | OpenAIResponsesRequest


class OpenAIRequest(BaseModel):
    """One public request envelope for Chat Completions and Responses.

    ``native_endpoint`` is deliberately explicit: a public destination method
    controls its returned protocol, whereas this field controls the endpoint
    that executes the request. The payload remains endpoint-specific so we do
    not accept or silently discard invalid OpenAI wire data.
    """

    model_config = ConfigDict(extra="forbid")

    native_endpoint: Literal["chat_completions", "responses"]
    payload: OpenAIRequestPayload

    @model_validator(mode="after")
    def _validate_endpoint_matches_payload(self) -> OpenAIRequest:
        expected_type = (
            OpenAIChatCompletionRequest
            if self.native_endpoint == "chat_completions"
            else OpenAIResponsesRequest
        )
        if not isinstance(self.payload, expected_type):
            raise ValueError(
                f"native_endpoint {self.native_endpoint!r} does not match "
                f"{type(self.payload).__name__}."
            )
        return self

    @classmethod
    def chat(cls, **payload: object) -> OpenAIRequest:
        """Build a Chat Completions request envelope."""

        return cls(native_endpoint="chat_completions", payload=payload)

    @classmethod
    def responses(cls, **payload: object) -> OpenAIRequest:
        """Build a Responses request envelope."""

        return cls(native_endpoint="responses", payload=payload)

    def to_ir(self) -> InputIR:
        """Convert the selected endpoint payload into the mandatory IR."""

        return self.payload.to_ir()
