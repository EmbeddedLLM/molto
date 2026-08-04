from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from .base import IRModel, StableId

FunctionName = StableId
OPENAI_WEB_SEARCH_TOOL_TYPES = frozenset(
    {
        "web_search",
        "web_search_2025_08_26",
        "web_search_preview",
        "web_search_preview_2025_03_11",
    }
)


class FunctionToolDefinition(IRModel):
    type: Literal["function"] = "function"
    name: FunctionName
    parameters: dict[str, object] | None = None
    description: str | None = None
    strict: bool | None = None


class OpenAIHostedToolDefinition(IRModel):
    type: Literal["openai_hosted_tool"] = "openai_hosted_tool"
    tool_type: str
    payload_json: str


ToolDefinition = Annotated[
    Union[FunctionToolDefinition, OpenAIHostedToolDefinition],
    Field(discriminator="type"),
]


class FunctionToolChoice(IRModel):
    type: Literal["function"] = "function"
    name: FunctionName


class AllowedFunctionTools(IRModel):
    type: Literal["allowed_tools"] = "allowed_tools"
    mode: Literal["auto", "required"]
    tools: tuple[FunctionToolChoice, ...]

    @model_validator(mode="after")
    def _validate_tools(self) -> AllowedFunctionTools:
        if not self.tools:
            raise ValueError("Allowed function tools cannot be empty.")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Allowed function tool names must be unique.")
        return self


class OpenAIHostedToolChoice(IRModel):
    type: Literal["openai_hosted_tool"] = "openai_hosted_tool"
    choice_type: str
    payload_json: str


ToolChoice = (
    Literal["none", "auto", "required"]
    | FunctionToolChoice
    | AllowedFunctionTools
    | OpenAIHostedToolChoice
)
