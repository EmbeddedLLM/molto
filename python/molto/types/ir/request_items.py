from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from .base import AssistantPhase, IRModel, ItemStatus, StableId
from .content import InputAudioPart, InputImagePart, InputTextPart, RequestContentPart
from .tools import FunctionName


class RequestMessageItem(IRModel):
    type: Literal["message"] = "message"
    id: StableId | None = None
    role: Literal["system", "developer", "user", "assistant"]
    content: tuple[RequestContentPart, ...]
    audio_id: StableId | None = None
    status: ItemStatus | None = None
    phase: AssistantPhase | None = None
    input_form: (
        Literal[
            "responses_string",
            "responses_item",
            "responses_output_item",
        ]
        | None
    ) = None
    type_present: bool = False

    @model_validator(mode="after")
    def _validate_content_form(self) -> RequestMessageItem:
        output_parts = tuple(
            part
            for part in self.content
            if not isinstance(part, (InputTextPart, InputImagePart, InputAudioPart))
        )
        if self.input_form == "responses_output_item":
            if self.role != "assistant" or self.id is None or self.status is None:
                raise ValueError(
                    "Responses output-message history requires assistant role, id, and status."
                )
            if len(output_parts) != len(self.content):
                raise ValueError("Responses output-message history requires output content parts.")
        elif output_parts:
            raise ValueError("Output content parts require Responses output-message history.")
        return self


class ReasoningRequestItem(IRModel):
    """A prior OpenAI Responses reasoning item supplied as conversation history."""

    type: Literal["reasoning"] = "reasoning"
    id: StableId
    summary: tuple[str, ...]
    content: tuple[str, ...] = ()
    encrypted_content: str | None = None
    status: ItemStatus | None = None


class WebSearchSource(IRModel):
    type: Literal["url"] = "url"
    url: str


class WebSearchActionSearch(IRModel):
    type: Literal["search"] = "search"
    query: str | None = None
    queries: tuple[str, ...] | None = None
    sources: tuple[WebSearchSource, ...] | None = None


class WebSearchActionOpenPage(IRModel):
    type: Literal["open_page"] = "open_page"
    url: str | None = None


class WebSearchActionFindInPage(IRModel):
    type: Literal["find_in_page"] = "find_in_page"
    pattern: str
    url: str


WebSearchAction = Annotated[
    Union[
        WebSearchActionSearch,
        WebSearchActionOpenPage,
        WebSearchActionFindInPage,
    ],
    Field(discriminator="type"),
]


class WebSearchCallRequestItem(IRModel):
    """A prior OpenAI Responses web-search call supplied as conversation history."""

    type: Literal["web_search_call"] = "web_search_call"
    id: StableId
    action: WebSearchAction
    status: Literal["in_progress", "searching", "completed", "failed"]


class CodeInterpreterOutputLog(IRModel):
    type: Literal["logs"] = "logs"
    logs: str


class CodeInterpreterOutputImage(IRModel):
    type: Literal["image"] = "image"
    url: str


CodeInterpreterOutput = Annotated[
    Union[CodeInterpreterOutputLog, CodeInterpreterOutputImage],
    Field(discriminator="type"),
]


class CodeInterpreterCallRequestItem(IRModel):
    """A prior OpenAI Responses code-interpreter call supplied as history."""

    type: Literal["code_interpreter_call"] = "code_interpreter_call"
    id: StableId
    code: str | None
    container_id: StableId
    outputs: tuple[CodeInterpreterOutput, ...] | None
    status: Literal[
        "in_progress",
        "completed",
        "incomplete",
        "interpreting",
        "failed",
    ]


class ImageGenerationCallRequestItem(IRModel):
    """A prior OpenAI Responses image-generation call supplied as history."""

    type: Literal["image_generation_call"] = "image_generation_call"
    id: StableId
    result: str | None = None
    status: Literal["in_progress", "completed", "generating", "failed"] | None = None
    fields_present: frozenset[Literal["result", "status"]] = frozenset()


class FunctionCallResultItem(IRModel):
    type: Literal["function_call_output"] = "function_call_output"
    id: StableId | None = None
    call_id: StableId
    output: str


class FunctionCallRequestItem(IRModel):
    """A prior developer-hosted function call supplied as conversation history."""

    type: Literal["function_call"] = "function_call"
    id: StableId | None = None
    status: ItemStatus | None = None
    call_id: StableId
    name: FunctionName
    arguments: str


RequestItem = Annotated[
    Union[
        RequestMessageItem,
        ReasoningRequestItem,
        FunctionCallRequestItem,
        FunctionCallResultItem,
        WebSearchCallRequestItem,
        CodeInterpreterCallRequestItem,
        ImageGenerationCallRequestItem,
    ],
    Field(discriminator="type"),
]
