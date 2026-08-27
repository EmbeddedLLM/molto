"""Pydantic request models and SDK-native result aliases for Anthropic Messages."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias, Union, get_args

from anthropic import AsyncStream
from anthropic.types import (
    CacheControlEphemeralParam,
    ContentBlock,
    ContentBlockParam,
    Message,
    MetadataParam,
    OutputConfigParam,
    RawMessageStreamEvent,
    TextBlockParam,
    ThinkingBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolUnionParam,
)
from pydantic import Field, TypeAdapter
from typing_extensions import NotRequired, Required, TypedDict

from molto.types.base import ExcludeNoneModel


class AnthropicThinkingBlockParam(TypedDict):
    """Thinking history accepted by compatible Messages APIs without a signature."""

    thinking: Required[str]
    type: Required[Literal["thinking"]]
    signature: NotRequired[str]


_RETAINED_CONTENT_BLOCKS = tuple(
    block for block in get_args(ContentBlockParam) if block is not ThinkingBlockParam
)
AnthropicContentBlockParam: TypeAlias = Union[
    AnthropicThinkingBlockParam,
    *_RETAINED_CONTENT_BLOCKS,  # pyright: ignore[reportInvalidTypeForm]
    ContentBlock,
]


class AnthropicMessageParam(ExcludeNoneModel):
    """One input turn in an Anthropic Messages request."""

    content: Annotated[
        str | list[AnthropicContentBlockParam],
        Field(description="The content of the conversation turn."),
    ]
    role: Annotated[
        Literal["user", "assistant", "system"],
        Field(description="The role that produced the conversation turn."),
    ]


class AnthropicMessagesCreateRequest(ExcludeNoneModel):
    """The JSON parameters accepted by `AsyncAnthropic.messages.create`."""

    max_tokens: Annotated[int, Field(description="The maximum number of output tokens.")]
    messages: Annotated[
        list[AnthropicMessageParam],
        Field(description="The conversation history."),
    ]
    model: Annotated[str, Field(description="The model ID used to generate the message.")]
    cache_control: Annotated[
        CacheControlEphemeralParam | None,
        Field(description="Top-level prompt cache control."),
    ] = None
    container: Annotated[
        str | None,
        Field(description="A container identifier to reuse across requests."),
    ] = None
    inference_geo: Annotated[
        str | None,
        Field(description="The geographic region used for inference."),
    ] = None
    metadata: Annotated[
        MetadataParam | None,
        Field(description="Metadata attached to the request."),
    ] = None
    output_config: Annotated[
        OutputConfigParam | None,
        Field(description="Output format configuration."),
    ] = None
    service_tier: Annotated[
        Literal["auto", "standard_only"] | None,
        Field(description="The service tier requested for the message."),
    ] = None
    stop_sequences: Annotated[
        list[str] | None,
        Field(description="Text sequences that stop generation."),
    ] = None
    stream: Annotated[
        Literal[True, False] | None,
        Field(description="Whether to stream message events."),
    ] = None
    system: Annotated[
        str | list[TextBlockParam] | None,
        Field(description="The system prompt."),
    ] = None
    temperature: Annotated[
        float | None,
        Field(description="The sampling temperature."),
    ] = None
    thinking: Annotated[
        ThinkingConfigParam | None,
        Field(description="Extended-thinking configuration."),
    ] = None
    tool_choice: Annotated[
        ToolChoiceParam | None,
        Field(description="How the model should select tools."),
    ] = None
    tools: Annotated[
        list[ToolUnionParam] | None,
        Field(description="Tools available to the model."),
    ] = None
    top_k: Annotated[int | None, Field(description="Top-k sampling cutoff.")] = None
    top_p: Annotated[float | None, Field(description="Nucleus-sampling probability mass.")] = None
    user_profile_id: Annotated[
        str | None,
        Field(description="A stable identifier for an end user."),
    ] = None


AnthropicMessagesResponse = Message
AnthropicMessagesStreamEvent = RawMessageStreamEvent
AnthropicMessagesStreamEventAdapter = TypeAdapter(AnthropicMessagesStreamEvent)
AnthropicMessagesStream = AsyncStream[AnthropicMessagesStreamEvent]
AnthropicMessagesCreateResult = AnthropicMessagesResponse | AnthropicMessagesStream

__all__ = [
    "AnthropicMessageParam",
    "AnthropicMessagesCreateRequest",
    "AnthropicMessagesCreateResult",
    "AnthropicMessagesResponse",
    "AnthropicMessagesStream",
    "AnthropicMessagesStreamEvent",
    "AnthropicMessagesStreamEventAdapter",
    "AnthropicThinkingBlockParam",
]
