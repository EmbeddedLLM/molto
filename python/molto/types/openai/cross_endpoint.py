"""OpenAI-compatible cross-endpoint output adapters."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from molto.types.ir.events import OutputEvent
from molto.types.ir.output_items import OpenAIHostedToolItem, ReasoningItem
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

    return responses_from_output_ir(output)


def chat_from_responses_output_ir(output: OutputIR) -> OpenAIChatCompletionResponse:
    """Render Responses-native semantic output as a Chat response."""

    if any(
        isinstance(item, (ReasoningItem, OpenAIHostedToolItem)) for item in output.output.values
    ):
        raise BadInputError(
            "Chat Completions output cannot represent reasoning or hosted-tool items."
        )
    return chat_from_output_ir(output)


async def responses_from_chat_output_event_stream(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OpenAIResponsesStreamEvent]:
    """Render Chat-native semantic events as an OpenAI Responses stream."""

    async for event in responses_from_output_event_stream(events):
        yield event


async def chat_from_responses_output_event_stream(
    events: AsyncIterable[OutputEvent],
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Render Responses-native semantic events as a Chat stream."""

    async for event in chat_from_output_event_stream(events):
        yield event
