from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from molto.types.ir.base import IRModel

TokenCount = Annotated[int, Field(ge=0)]


class Usage(IRModel):
    input_tokens: TokenCount = 0
    output_tokens: TokenCount = 0
    total_tokens: TokenCount = 0
    cached_input_tokens: TokenCount = 0
    cache_write_input_tokens: TokenCount | None = None
    reasoning_output_tokens: TokenCount = 0
    audio_input_tokens: TokenCount | None = None
    image_input_tokens: TokenCount | None = None
    text_input_tokens: TokenCount | None = None
    audio_output_tokens: TokenCount | None = None
    image_output_tokens: TokenCount | None = None
    text_output_tokens: TokenCount | None = None
    accepted_prediction_output_tokens: TokenCount | None = None
    rejected_prediction_output_tokens: TokenCount | None = None

    @model_validator(mode="after")
    def _validate_breakdowns(self) -> Usage:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens.")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens.")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning_output_tokens cannot exceed output_tokens.")
        return self
