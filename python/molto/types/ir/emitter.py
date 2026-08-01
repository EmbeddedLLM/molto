"""Helpers for constructing ordered OpenAI-compatible output events."""

from __future__ import annotations

from collections.abc import Iterator

from molto.types.ir.content import AssistantContentPart, ReasoningPart
from molto.types.ir.errors import ErrorDetail
from molto.types.ir.events import (
    ContentPartAddedEvent,
    ContentPartDeltaEvent,
    ContentPartDoneEvent,
    FunctionArgumentsDeltaEvent,
    FunctionArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseMetadata,
    ResponseStartedEvent,
)
from molto.types.ir.output_items import OutputItem
from molto.types.ir.usage import Usage


class OutputEventEmitter:
    """Stamp response identity and a monotonic sequence number on each event."""

    def __init__(
        self,
        response_id: str,
        *,
        metadata: ResponseMetadata | None = None,
        per_event: bool = True,
    ) -> None:
        self._response_id = response_id
        self._metadata = metadata or ResponseMetadata()
        self._per_event = per_event
        self._sequence_number = 0

    @property
    def sequence_number(self) -> int:
        return self._sequence_number

    def advance(self) -> None:
        """Advance the counter for providers that number one frame as a group."""

        self._sequence_number += 1

    def _stamp(self) -> dict[str, int | str]:
        sequence_number = self._sequence_number
        if self._per_event:
            self._sequence_number += 1
        return {"response_id": self._response_id, "sequence_number": sequence_number}

    def started(self) -> ResponseStartedEvent:
        return ResponseStartedEvent(**self._stamp(), metadata=self._metadata)

    def item_added(self, item_index: int, item: OutputItem) -> OutputItemAddedEvent:
        return OutputItemAddedEvent(**self._stamp(), item_index=item_index, item=item)

    def item_done(self, item_index: int, item: OutputItem) -> OutputItemDoneEvent:
        return OutputItemDoneEvent(**self._stamp(), item_index=item_index, item=item)

    def part_added(
        self,
        item_id: str,
        item_index: int,
        content_index: int,
        part: AssistantContentPart | ReasoningPart,
    ) -> ContentPartAddedEvent:
        return ContentPartAddedEvent(
            **self._stamp(),
            item_id=item_id,
            item_index=item_index,
            content_index=content_index,
            part=part,
        )

    def part_delta(
        self, item_id: str, item_index: int, content_index: int, delta: str
    ) -> ContentPartDeltaEvent:
        return ContentPartDeltaEvent(
            **self._stamp(),
            item_id=item_id,
            item_index=item_index,
            content_index=content_index,
            delta=delta,
        )

    def part_done(
        self,
        item_id: str,
        item_index: int,
        content_index: int,
        part: AssistantContentPart | ReasoningPart,
    ) -> ContentPartDoneEvent:
        return ContentPartDoneEvent(
            **self._stamp(),
            item_id=item_id,
            item_index=item_index,
            content_index=content_index,
            part=part,
        )

    def function_arguments_delta(
        self, item_id: str, item_index: int, delta: str
    ) -> FunctionArgumentsDeltaEvent:
        return FunctionArgumentsDeltaEvent(
            **self._stamp(), item_id=item_id, item_index=item_index, delta=delta
        )

    def function_arguments_done(
        self, item_id: str, item_index: int, arguments: str
    ) -> FunctionArgumentsDoneEvent:
        return FunctionArgumentsDoneEvent(
            **self._stamp(), item_id=item_id, item_index=item_index, arguments=arguments
        )

    def completed(self, usage: Usage) -> ResponseCompletedEvent:
        return ResponseCompletedEvent(**self._stamp(), usage=usage)

    def incomplete(self, reason: str, usage: Usage) -> ResponseIncompleteEvent:
        return ResponseIncompleteEvent(**self._stamp(), reason=reason, usage=usage)

    def failed(self, error: ErrorDetail, usage: Usage) -> ResponseFailedEvent:
        return ResponseFailedEvent(**self._stamp(), error=error, usage=usage)

    def replay_part(
        self,
        item_id: str,
        item_index: int,
        content_index: int,
        part: AssistantContentPart | ReasoningPart,
        delta: str,
        *,
        empty_part: AssistantContentPart | ReasoningPart | None = None,
    ) -> Iterator[ContentPartAddedEvent | ContentPartDeltaEvent | ContentPartDoneEvent]:
        """Replay one content part's added, delta, and done choreography."""

        started_part = empty_part or part.model_copy(update={part.value_field: ""})
        yield self.part_added(item_id, item_index, content_index, started_part)
        yield self.part_delta(item_id, item_index, content_index, delta)
        yield self.part_done(item_id, item_index, content_index, part)
