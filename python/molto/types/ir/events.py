from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from .base import (
    ContentIndex,
    IRModel,
    ItemIndex,
    ItemStatus,
    ResponseStatus,
    SequenceNumber,
    StableId,
)
from .content import (
    AssistantContentPart,
    EncryptedReasoningPart,
    OutputTextPart,
    ReasoningContentPart,
    ReasoningPart,
    ReasoningSummaryPart,
    RefusalPart,
)
from .errors import ErrorDetail
from .output_items import (
    AssistantMessageItem,
    FunctionCallItem,
    OpenAIHostedToolItem,
    OutputItem,
    ReasoningItem,
)
from .usage import Usage


class ResponseMetadata(IRModel):
    """OpenAI-compatible response fields needed by public output adapters."""

    model: str = ""
    created_at: float | None = None
    object: str | None = None
    choice_index: int = 0
    finish_reason: str | None = None
    service_tier: str | None = None
    service_tier_present: bool = False
    system_fingerprint: str | None = None
    logprobs_present: bool = False


class _Event(IRModel):
    response_id: StableId
    sequence_number: SequenceNumber


class ResponseStartedEvent(_Event):
    type: Literal["response.started"] = "response.started"
    status: Literal[ResponseStatus.IN_PROGRESS] = ResponseStatus.IN_PROGRESS
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class OutputItemAddedEvent(_Event):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    item_index: ItemIndex
    item: OutputItem

    @model_validator(mode="after")
    def _validate_item_started(self) -> OutputItemAddedEvent:
        if self.item.status is not ItemStatus.IN_PROGRESS:
            raise ValueError("An added output item must be in progress.")
        return self


class ContentPartAddedEvent(_Event):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: StableId
    item_index: ItemIndex
    content_index: ContentIndex
    part: AssistantContentPart | ReasoningPart


class ContentPartDeltaEvent(_Event):
    type: Literal["response.content_part.delta"] = "response.content_part.delta"
    item_id: StableId
    item_index: ItemIndex
    content_index: ContentIndex
    delta: str


class ContentPartDoneEvent(_Event):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: StableId
    item_index: ItemIndex
    content_index: ContentIndex
    part: AssistantContentPart | ReasoningPart


class FunctionArgumentsDeltaEvent(_Event):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    item_id: StableId
    item_index: ItemIndex
    delta: str


class FunctionArgumentsDoneEvent(_Event):
    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    item_id: StableId
    item_index: ItemIndex
    arguments: str


class PreservedOpenAIStreamEvent(_Event):
    """One validated frame from an OpenAI-compatible streaming endpoint."""

    type: Literal["openai.stream_frame"] = "openai.stream_frame"
    endpoint: Literal["chat_completions", "responses"]
    event_type: str
    payload_json: str


class PreservedOpenAIResponseEvent(_Event):
    """One complete validated response from an OpenAI-compatible endpoint."""

    type: Literal["openai.response_snapshot"] = "openai.response_snapshot"
    endpoint: Literal["chat_completions", "responses"]
    payload_json: str


class OutputItemDoneEvent(_Event):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    item_index: ItemIndex
    item: OutputItem

    @model_validator(mode="after")
    def _validate_item_finished(self) -> OutputItemDoneEvent:
        if not self.item.status.terminal:
            raise ValueError("A done output item must have a terminal status.")
        return self


class ResponseCompletedEvent(_Event):
    type: Literal["response.completed"] = "response.completed"
    status: Literal[ResponseStatus.COMPLETED] = ResponseStatus.COMPLETED
    usage: Usage


class ResponseIncompleteEvent(_Event):
    type: Literal["response.incomplete"] = "response.incomplete"
    status: Literal[ResponseStatus.INCOMPLETE] = ResponseStatus.INCOMPLETE
    reason: str
    usage: Usage


class ResponseFailedEvent(_Event):
    type: Literal["response.failed"] = "response.failed"
    status: Literal[ResponseStatus.FAILED] = ResponseStatus.FAILED
    error: ErrorDetail
    usage: Usage


OutputEvent = Annotated[
    Union[
        ResponseStartedEvent,
        OutputItemAddedEvent,
        ContentPartAddedEvent,
        ContentPartDeltaEvent,
        ContentPartDoneEvent,
        FunctionArgumentsDeltaEvent,
        FunctionArgumentsDoneEvent,
        PreservedOpenAIStreamEvent,
        PreservedOpenAIResponseEvent,
        OutputItemDoneEvent,
        ResponseCompletedEvent,
        ResponseIncompleteEvent,
        ResponseFailedEvent,
    ],
    Field(discriminator="type"),
]
OutputStreamEvent = OutputEvent

_TerminalEvent = ResponseCompletedEvent | ResponseIncompleteEvent | ResponseFailedEvent
_ContentEvent = ContentPartAddedEvent | ContentPartDeltaEvent | ContentPartDoneEvent
_FunctionArgumentsEvent = FunctionArgumentsDeltaEvent | FunctionArgumentsDoneEvent


@dataclass
class _OpenPart:
    started: AssistantContentPart | ReasoningPart
    deltas: list[str] = field(default_factory=list)

    @property
    def expected_value(self) -> str:
        return getattr(self.started, self.started.value_field) + "".join(self.deltas)


class OutputEventSequence(IRModel):
    """A complete OpenAI-compatible response stream with stable ordering."""

    events: tuple[OutputEvent, ...]

    @property
    def terminal(self) -> _TerminalEvent:
        return self.events[-1]

    @property
    def metadata(self) -> ResponseMetadata:
        return self.events[0].metadata

    @property
    def done_items(self) -> tuple[OutputItem, ...]:
        return tuple(
            event.item
            for event in sorted(
                (event for event in self.events if isinstance(event, OutputItemDoneEvent)),
                key=lambda event: event.item_index,
            )
        )

    @model_validator(mode="after")
    def _validate_sequence(self) -> OutputEventSequence:
        if not self.events:
            raise ValueError("An output event sequence cannot be empty.")
        if not isinstance(self.events[0], ResponseStartedEvent):
            raise ValueError("An output event sequence must start with response.started.")
        if not isinstance(self.events[-1], _TerminalEvent):
            raise ValueError("An output event sequence must end with a terminal response event.")

        response_id = self.events[0].response_id
        previous_sequence_number = -1
        open_items: dict[int, OutputItem] = {}
        open_content: dict[tuple[int, int], _OpenPart] = {}
        completed_content: dict[int, dict[int, AssistantContentPart | ReasoningPart]] = {}
        function_argument_deltas: dict[int, list[str]] = {}
        completed_function_arguments: dict[int, str] = {}
        seen_item_ids: set[str] = set()
        seen_item_indices: set[int] = set()
        done_items: dict[int, OutputItem] = {}

        for position, event in enumerate(self.events):
            if event.response_id != response_id:
                raise ValueError("Every output event must use the same response ID.")
            if event.sequence_number <= previous_sequence_number:
                raise ValueError("Output event sequence numbers must increase monotonically.")
            previous_sequence_number = event.sequence_number
            if isinstance(event, _TerminalEvent) and position != len(self.events) - 1:
                raise ValueError("A terminal response event must be last.")
            if isinstance(event, ResponseStartedEvent) and position != 0:
                raise ValueError("response.started can appear only once, at the beginning.")

            if isinstance(event, OutputItemAddedEvent):
                if event.item_index in seen_item_indices:
                    raise ValueError(f"Output item index {event.item_index} was already used.")
                if event.item_index != len(seen_item_indices):
                    raise ValueError("Output item indices must be contiguous and start at zero.")
                if event.item.id in seen_item_ids:
                    raise ValueError(f"Output item ID {event.item.id!r} was already used.")
                seen_item_indices.add(event.item_index)
                seen_item_ids.add(event.item.id)
                open_items[event.item_index] = event.item
            elif isinstance(event, _ContentEvent):
                item = self._require_open_item(open_items, event.item_index, event.item_id)
                if isinstance(item, FunctionCallItem):
                    raise ValueError("Function-call items cannot emit content-part events.")
                if isinstance(event, (ContentPartAddedEvent, ContentPartDoneEvent)):
                    if isinstance(item, AssistantMessageItem) and not isinstance(
                        event.part, (OutputTextPart, RefusalPart)
                    ):
                        raise ValueError("Assistant messages require assistant content parts.")
                    if isinstance(item, ReasoningItem) and not isinstance(
                        event.part,
                        (ReasoningSummaryPart, ReasoningContentPart, EncryptedReasoningPart),
                    ):
                        raise ValueError("Reasoning items require reasoning content parts.")
                key = (event.item_index, event.content_index)
                if isinstance(event, ContentPartAddedEvent):
                    if key in open_content:
                        raise ValueError(f"Content part {key} is already open.")
                    expected_index = len(completed_content.get(event.item_index, {}))
                    if event.content_index != expected_index:
                        raise ValueError(
                            "Content-part indices must be contiguous and start at zero."
                        )
                    open_content[key] = _OpenPart(started=event.part)
                elif key not in open_content:
                    raise ValueError(f"Content part {key} has not been added.")
                elif isinstance(event, ContentPartDeltaEvent):
                    open_content[key].deltas.append(event.delta)
                elif isinstance(event, ContentPartDoneEvent):
                    open_part = open_content[key]
                    if type(open_part.started) is not type(event.part):
                        raise ValueError(f"Content part {key} changed type before completion.")
                    if self._part_metadata(open_part.started) != self._part_metadata(event.part):
                        raise ValueError(f"Content part {key} changed metadata before completion.")
                    if getattr(event.part, event.part.value_field) != open_part.expected_value:
                        raise ValueError(f"Content part {key} does not match its ordered deltas.")
                    completed_content.setdefault(event.item_index, {})[event.content_index] = (
                        event.part
                    )
                    del open_content[key]
            elif isinstance(event, _FunctionArgumentsEvent):
                item = self._require_open_item(open_items, event.item_index, event.item_id)
                if not isinstance(item, FunctionCallItem):
                    raise ValueError(
                        "Function argument events require an open function-call item."
                    )
                if isinstance(event, FunctionArgumentsDeltaEvent):
                    if event.item_index in completed_function_arguments:
                        raise ValueError(
                            "Function arguments cannot change after their done event."
                        )
                    function_argument_deltas.setdefault(event.item_index, []).append(event.delta)
                elif event.item_index in completed_function_arguments:
                    raise ValueError("Function arguments can have only one done event.")
                else:
                    arguments = item.arguments + "".join(
                        function_argument_deltas.get(event.item_index, [])
                    )
                    if event.arguments != arguments:
                        raise ValueError(
                            "Completed function arguments do not match their ordered deltas."
                        )
                    completed_function_arguments[event.item_index] = event.arguments
            elif isinstance(event, OutputItemDoneEvent):
                item = self._require_open_item(open_items, event.item_index, event.item.id)
                if any(key[0] == event.item_index for key in open_content):
                    raise ValueError("An output item cannot finish with open content parts.")
                if type(item) is not type(event.item):
                    raise ValueError("An output item cannot change type before completion.")
                if isinstance(event.item, FunctionCallItem):
                    if (event.item.call_id, event.item.name) != (item.call_id, item.name):
                        raise ValueError("A function-call item cannot change call ID or name.")
                    if completed_function_arguments.get(event.item_index) != event.item.arguments:
                        raise ValueError(
                            "A function-call item must finish with its completed arguments."
                        )
                elif isinstance(event.item, AssistantMessageItem):
                    if item.audio != event.item.audio:
                        raise ValueError("Generated audio cannot change before item completion.")
                    if item.annotations != event.item.annotations:
                        raise ValueError("Annotations cannot change before item completion.")
                    parts = completed_content.get(event.item_index, {})
                    if tuple(parts[index] for index in sorted(parts)) != event.item.content:
                        raise ValueError(
                            "A completed output item must contain its completed content parts."
                        )
                elif isinstance(event.item, OpenAIHostedToolItem):
                    if (
                        item.item_type,
                        item.openai_status,
                        item.payload_json,
                    ) != (
                        event.item.item_type,
                        event.item.openai_status,
                        event.item.payload_json,
                    ):
                        raise ValueError(
                            "An OpenAI hosted-tool item cannot change before completion."
                        )
                else:
                    parts = completed_content.get(event.item_index, {})
                    if tuple(parts[index] for index in sorted(parts)) != event.item.parts:
                        raise ValueError(
                            "A completed output item must contain its completed content parts."
                        )
                done_items[event.item_index] = event.item
                del open_items[event.item_index]

        if open_content:
            raise ValueError("Every content part must be completed.")
        if open_items:
            raise ValueError("Every output item must be completed.")
        terminal = self.events[-1]
        if isinstance(terminal, ResponseCompletedEvent) and any(
            item.status is not ItemStatus.COMPLETED for item in done_items.values()
        ):
            raise ValueError("A completed response requires completed output items.")
        return self

    @staticmethod
    def _part_metadata(part: AssistantContentPart | ReasoningPart) -> tuple[object, ...]:
        return tuple(
            getattr(part, name)
            for name in type(part).model_fields
            if name not in {"text", "refusal", "data"}
        )

    @staticmethod
    def _require_open_item(
        open_items: dict[int, OutputItem], item_index: int, item_id: str
    ) -> OutputItem:
        item = open_items.get(item_index)
        if item is None:
            raise ValueError(f"Output item index {item_index} has not been added.")
        if item.id != item_id:
            raise ValueError(f"Output item index {item_index} changed identity.")
        return item
