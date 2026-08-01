from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from .base import AssistantPhase, IRModel, ItemStatus, StableId
from .content import AssistantContentPart, ReasoningPart, UrlCitation
from .tools import FunctionName


class GeneratedAudio(IRModel):
    id: StableId
    data: str
    expires_at: int
    transcript: str


class AssistantMessageItem(IRModel):
    type: Literal["message"] = "message"
    id: StableId
    status: ItemStatus
    role: Literal["assistant"] = "assistant"
    content: tuple[AssistantContentPart, ...]
    phase: AssistantPhase | None = None
    audio: GeneratedAudio | None = None
    annotations: tuple[UrlCitation, ...] | None = None


class ReasoningItem(IRModel):
    type: Literal["reasoning"] = "reasoning"
    id: StableId
    status: ItemStatus
    parts: tuple[ReasoningPart, ...]


class FunctionCallItem(IRModel):
    type: Literal["function_call"] = "function_call"
    id: StableId
    status: ItemStatus
    call_id: StableId
    name: FunctionName
    arguments: str


class OpenAIHostedToolItem(IRModel):
    type: Literal["openai_hosted_tool"] = "openai_hosted_tool"
    id: StableId
    status: ItemStatus
    item_type: str
    openai_status: str | None = None
    payload_json: str


OutputItem = Annotated[
    Union[
        AssistantMessageItem,
        ReasoningItem,
        FunctionCallItem,
        OpenAIHostedToolItem,
    ],
    Field(discriminator="type"),
]


class OutputItems(IRModel):
    values: tuple[OutputItem, ...]

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> OutputItems:
        ids = [item.id for item in self.values]
        if len(ids) != len(set(ids)):
            raise ValueError("Output item IDs must be unique.")
        return self
