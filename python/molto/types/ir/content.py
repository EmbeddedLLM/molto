from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

from pydantic import Field

from .base import ContentIndex, IRModel, ProviderName


class InputTextPart(IRModel):
    type: Literal["input_text"] = "input_text"
    text: str


class InputImagePart(IRModel):
    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: Literal["auto", "low", "high"] = "auto"


class InputAudioPart(IRModel):
    type: Literal["input_audio"] = "input_audio"
    data: str
    format: Literal["wav", "mp3"]


class UrlCitation(IRModel):
    type: Literal["url_citation"] = "url_citation"
    url: str
    title: str
    start_index: ContentIndex
    end_index: ContentIndex


class OutputTextPart(IRModel):
    value_field: ClassVar[str] = "text"
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: tuple[UrlCitation, ...] | None = None


class RefusalPart(IRModel):
    value_field: ClassVar[str] = "refusal"
    type: Literal["refusal"] = "refusal"
    refusal: str


class ReasoningSummaryPart(IRModel):
    value_field: ClassVar[str] = "text"
    type: Literal["summary_text"] = "summary_text"
    text: str


class ReasoningContentPart(IRModel):
    """Raw reasoning text only when a provider explicitly exposes it."""

    value_field: ClassVar[str] = "text"
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str
    provider: ProviderName


class EncryptedReasoningPart(IRModel):
    """Opaque provider state; it is never interpreted as assistant text."""

    value_field: ClassVar[str] = "data"
    type: Literal["encrypted_reasoning"] = "encrypted_reasoning"
    data: str
    provider: ProviderName


InputContentPart = Annotated[
    Union[InputTextPart, InputImagePart, InputAudioPart],
    Field(discriminator="type"),
]
AssistantContentPart = Annotated[
    Union[OutputTextPart, RefusalPart],
    Field(discriminator="type"),
]
ReasoningPart = Annotated[
    Union[ReasoningSummaryPart, ReasoningContentPart, EncryptedReasoningPart],
    Field(discriminator="type"),
]
RequestContentPart = Annotated[
    Union[
        InputTextPart,
        InputImagePart,
        InputAudioPart,
        OutputTextPart,
        RefusalPart,
    ],
    Field(discriminator="type"),
]
