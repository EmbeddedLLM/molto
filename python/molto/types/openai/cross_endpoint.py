"""OpenAI-compatible cross-endpoint output adapters."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from molto.types.ir.base import ResponseStatus
from molto.types.ir.events import (
    OutputEvent,
    OutputItemDoneEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
)
from molto.types.ir.output_items import AssistantMessageItem, OpenAIHostedToolItem
from molto.types.ir.portable import OutputIR
from molto.utils.exceptions import BadInputError

from .chat_completions.from_ir import (
    from_output_event_stream as chat_from_output_event_stream,
)
from .chat_completions.from_ir import (
    from_output_ir as chat_from_output_ir,
)
from .chat_completions.models_response import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionResponse,
)
from .responses.from_ir import (
    from_output_event_stream as responses_from_output_event_stream,
)
from .responses.from_ir import (
    from_output_ir as responses_from_output_ir,
)
from .responses.models_response import OpenAIResponsesResponse, OpenAIResponsesStreamEvent


def responses_from_chat_output_ir(output: OutputIR) -> OpenAIResponsesResponse:
    """Render Chat-native semantic output as an OpenAI Responses response."""

    if any(
        isinstance(item, AssistantMessageItem) and item.audio is not None
        for item in output.output.values
    ):
        raise BadInputError("Responses output cannot represent generated Chat audio.")
    return responses_from_output_ir(output)


def chat_from_responses_output_ir(output: OutputIR) -> OpenAIChatCompletionResponse:
    """Render Responses-native semantic output as a Chat response."""

    if output.status is ResponseStatus.FAILED:
        raise BadInputError("Chat Completions output cannot represent a failed Response.")
    if output.status is ResponseStatus.INCOMPLETE and output.incomplete_reason not in {
        "max_output_tokens",
        "content_filter",
    }:
        raise BadInputError(
            "Chat Completions output cannot represent Responses incomplete reason "
            f"{output.incomplete_reason!r}."
        )
    _validate_responses_output_for_chat(output.output.values)
    # Chat has no reasoning-item shape. The Chat composer emits only assistant
    # messages and function calls, intentionally discarding Responses reasoning.
    return chat_from_output_ir(output)


async def responses_from_chat_output_event_stream(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OpenAIResponsesStreamEvent]:
    """Render Chat-native semantic events as an OpenAI Responses stream."""

    async for event in responses_from_output_event_stream(_reject_chat_audio(events)):
        yield event


async def chat_from_responses_output_event_stream(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Render Responses-native semantic events as a Chat stream."""

    # The Chat stream is synthesized from the complete event sequence, but its
    # public chunks intentionally omit Responses reasoning events.
    async for event in chat_from_output_event_stream(_reject_unsupported_chat_output(events)):
        yield event


async def _reject_chat_audio(events: AsyncIterable[OutputEvent]) -> AsyncIterator[OutputEvent]:
    async for event in events:
        if (
            isinstance(event, OutputItemDoneEvent)
            and isinstance(event.item, AssistantMessageItem)
            and event.item.audio is not None
        ):
            raise BadInputError("Responses output cannot represent generated Chat audio.")
        yield event


def _validate_responses_output_for_chat(items: tuple[object, ...]) -> None:
    messages = [item for item in items if isinstance(item, AssistantMessageItem)]
    if len(messages) > 1:
        raise BadInputError("Chat Completions output cannot represent multiple messages.")
    if any(isinstance(item, OpenAIHostedToolItem) for item in items):
        raise BadInputError(
            "Chat Completions output cannot represent Responses hosted-tool items."
        )
    for message in messages:
        if message.phase is not None:
            raise BadInputError("Chat Completions output cannot represent Responses phase.")
        part_types = [type(part) for part in message.content]
        if any(part_types.count(part_type) > 1 for part_type in set(part_types)):
            raise BadInputError(
                "Chat Completions output cannot represent repeated content-part types."
            )


async def _reject_unsupported_chat_output(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OutputEvent]:
    assistant_messages = 0
    async for event in events:
        if isinstance(event, ResponseFailedEvent):
            raise BadInputError("Chat Completions output cannot represent a failed Response.")
        if isinstance(event, ResponseIncompleteEvent) and event.reason not in {
            "max_output_tokens",
            "content_filter",
        }:
            raise BadInputError(
                "Chat Completions output cannot represent Responses incomplete reason "
                f"{event.reason!r}."
            )
        if isinstance(event, OutputItemDoneEvent):
            _validate_responses_output_for_chat((event.item,))
            if isinstance(event.item, AssistantMessageItem):
                assistant_messages += 1
                if assistant_messages > 1:
                    raise BadInputError(
                        "Chat Completions output cannot represent multiple messages."
                    )
        yield event
