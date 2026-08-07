from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from molto.types.ir.base import IRModel, ResponseStatus, StableId
from molto.types.ir.config import GenerationConfig, OutputFormat, StreamOptions, TextOutputFormat
from molto.types.ir.errors import ErrorDetail
from molto.types.ir.events import (
    OutputEventSequence,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseMetadata,
)
from molto.types.ir.output_items import OutputItems
from molto.types.ir.request_items import (
    FunctionCallRequestItem,
    FunctionCallResultItem,
    RequestItem,
)
from molto.types.ir.state import StateReferences
from molto.types.ir.tools import (
    AllowedFunctionTools,
    FunctionToolChoice,
    FunctionToolDefinition,
    OpenAIHostedToolChoice,
    ToolChoice,
    ToolDefinition,
)
from molto.types.ir.usage import Usage


class InputIR(IRModel):
    """Complete request contract consumed by OpenAI-compatible adapters."""

    model: str
    stream: bool | None
    stream_options: StreamOptions | None = None
    # Whether the public output stream should include usage. Distinct from
    # `stream_options`: that value is forwarded to the provider, while this one
    # controls output emission and survives cross-endpoint routing (e.g. a
    # Chat request executed through the Responses endpoint).
    include_usage: bool | None = None
    store: bool | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    safety_identifier: str | None = None
    user: str | None = None
    instructions: str | None = None
    include: tuple[str, ...] | None = None
    items: tuple[RequestItem, ...]
    output_format: OutputFormat = TextOutputFormat()
    generation: GenerationConfig = GenerationConfig()
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice | None = None
    state: StateReferences | None = None

    @model_validator(mode="after")
    def _validate_request_relationships(self) -> InputIR:
        item_ids: set[str] = set()
        call_ids: set[str] = set()
        result_call_ids: set[str] = set()
        function_tools = tuple(
            tool for tool in self.tools if isinstance(tool, FunctionToolDefinition)
        )
        tool_names = {tool.name for tool in function_tools}

        if len(tool_names) != len(function_tools):
            raise ValueError("Function tool names must be unique.")

        for item in self.items:
            if item.id is not None:
                if item.id in item_ids:
                    raise ValueError(f"Request item ID {item.id!r} is duplicated.")
                item_ids.add(item.id)
            if isinstance(item, FunctionCallRequestItem):
                if item.call_id in call_ids:
                    raise ValueError(f"Function call ID {item.call_id!r} is duplicated.")
                call_ids.add(item.call_id)
            elif isinstance(item, FunctionCallResultItem):
                if item.call_id not in call_ids:
                    raise ValueError(
                        f"Function result {item.call_id!r} must follow its function call."
                    )
                if item.call_id in result_call_ids:
                    raise ValueError(f"Function call {item.call_id!r} has more than one result.")
                result_call_ids.add(item.call_id)

        selected_names: set[str] = set()
        if isinstance(self.tool_choice, FunctionToolChoice):
            selected_names.add(self.tool_choice.name)
        elif isinstance(self.tool_choice, AllowedFunctionTools):
            selected_names.update(tool.name for tool in self.tool_choice.tools)
        elif isinstance(self.tool_choice, OpenAIHostedToolChoice):
            selected_names.clear()
        unknown_names = selected_names - tool_names
        if unknown_names:
            names = ", ".join(sorted(unknown_names))
            raise ValueError(f"Function tool choice references undeclared tools: {names}.")
        return self


class OutputIR(IRModel):
    """Terminal response snapshot and its validated OpenAI-compatible events."""

    id: StableId
    status: ResponseStatus
    output: OutputItems
    usage: Usage
    metadata: ResponseMetadata = ResponseMetadata()
    error: ErrorDetail | None = None
    incomplete_reason: str | None = None
    state: StateReferences | None = None
    event_sequence: OutputEventSequence

    @model_validator(mode="after")
    def _validate_snapshot_against_events(self) -> OutputIR:
        if not self.status.terminal:
            raise ValueError("OutputIR requires a terminal response status.")

        terminal = self.event_sequence.terminal
        if terminal.response_id != self.id:
            raise ValueError("OutputIR ID must match the output event response ID.")
        if terminal.status is not self.status:
            raise ValueError("OutputIR status must match the terminal output event status.")
        if terminal.usage != self.usage:
            raise ValueError("OutputIR usage must match terminal output event usage.")
        if self.event_sequence.metadata != self.metadata:
            raise ValueError("OutputIR metadata must match the started output event.")
        if self.event_sequence.done_items != self.output.values:
            raise ValueError("OutputIR items must match ordered output item events.")

        if isinstance(terminal, ResponseCompletedEvent):
            if self.error is not None or self.incomplete_reason is not None:
                raise ValueError(
                    "A completed response cannot carry an error or incomplete reason."
                )
        elif isinstance(terminal, ResponseIncompleteEvent):
            if self.incomplete_reason != terminal.reason:
                raise ValueError(
                    "OutputIR incomplete reason must match the terminal output event reason."
                )
            if self.error is not None:
                raise ValueError("An incomplete response cannot carry an error.")
        elif isinstance(terminal, ResponseFailedEvent):
            if self.error != terminal.error:
                raise ValueError("OutputIR error must match the terminal output event error.")
            if self.incomplete_reason is not None:
                raise ValueError("A failed response cannot carry an incomplete reason.")
        return self

    @classmethod
    def from_output_events(
        cls,
        event_sequence: OutputEventSequence,
        *,
        state: StateReferences | None = None,
    ) -> OutputIR:
        """Build the terminal snapshot from a validated event sequence."""

        terminal = event_sequence.terminal
        error = terminal.error if isinstance(terminal, ResponseFailedEvent) else None
        incomplete_reason = (
            terminal.reason if isinstance(terminal, ResponseIncompleteEvent) else None
        )
        return cls(
            id=terminal.response_id,
            status=terminal.status,
            output=OutputItems(values=event_sequence.done_items),
            usage=terminal.usage,
            metadata=event_sequence.metadata,
            error=error,
            incomplete_reason=incomplete_reason,
            state=state,
            event_sequence=event_sequence,
        )
