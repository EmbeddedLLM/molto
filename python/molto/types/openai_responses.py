"""Pydantic models for `AsyncOpenAI.responses.create`.

The nested parameter types are imported from the pinned OpenAI SDK.  They are
generated from OpenAI's API schema and preserve the SDK's discriminated unions,
while this module provides Pydantic entry points for request bodies and results.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, Union, get_args

from openai import AsyncStream
from openai.types.responses import Response, ResponseCompletedEvent, ResponseStreamEvent
from openai.types.responses.response_create_params import (
    ContextManagement,
    Conversation,
    Moderation,
    PromptCacheOptions,
    StreamOptions,
    ToolChoice,
)
from openai.types.responses.response_includable import ResponseIncludable
from openai.types.responses.response_input_param import ResponseInputItemParam
from openai.types.responses.response_output_message_param import (
    Content as ResponseOutputMessageContentParam,
)
from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.responses.response_reasoning_item_param import (
    Content as ResponseReasoningContentParam,
)
from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam
from openai.types.responses.response_reasoning_item_param import (
    Summary as ResponseReasoningSummaryParam,
)
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.metadata import Metadata
from openai.types.shared.reasoning import Reasoning
from openai.types.shared.responses_model import ResponsesModel
from pydantic import Discriminator, Field, TypeAdapter, model_validator

from molto.types.base import ExcludeNoneModel


class ResponseReasoningItemOptionalIDParam(ExcludeNoneModel):
    """Reasoning history accepted by compatible Responses APIs without an item ID."""

    id: Annotated[str | None, Field(description="The optional reasoning item ID.")] = None
    summary: Annotated[
        list[ResponseReasoningSummaryParam],
        Field(description="Reasoning summary content."),
    ]
    type: Annotated[
        Literal["reasoning"],
        Field(description="The reasoning item discriminator."),
    ]
    content: Annotated[
        list[ResponseReasoningContentParam] | None,
        Field(description="Raw reasoning text content."),
    ] = None
    encrypted_content: Annotated[
        str | None,
        Field(description="Encrypted reasoning content."),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(description="The reasoning item status."),
    ] = None


class ResponseOutputMessageOptionalIDParam(ExcludeNoneModel):
    """Assistant output history accepted by compatible Responses APIs without an ID."""

    id: Annotated[
        str | None,
        Field(description="The optional output message ID."),
    ] = None
    content: Annotated[
        list[ResponseOutputMessageContentParam],
        Field(description="The assistant message content."),
    ]
    role: Annotated[
        Literal["assistant"],
        Field(description="The output message role."),
    ]
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"],
        Field(description="The output message status."),
    ]
    type: Annotated[
        Literal["message"],
        Field(description="The output message discriminator."),
    ]
    phase: Annotated[
        Literal["commentary", "final_answer"] | None,
        Field(description="The assistant message phase."),
    ] = None


# Compatible providers omit IDs from response history, so replace only the two
# overly strict SDK members while inheriting every other current SDK input type.
_REPLACED_SDK_INPUT_ITEMS = {
    ResponseReasoningItemParam,
    ResponseOutputMessageParam,
}
_RETAINED_SDK_INPUT_ITEMS = tuple(
    item for item in get_args(ResponseInputItemParam) if item not in _REPLACED_SDK_INPUT_ITEMS
)
OpenAIResponsesInputItemParam: TypeAlias = Union[
    ResponseReasoningItemOptionalIDParam,
    ResponseOutputMessageOptionalIDParam,
    *_RETAINED_SDK_INPUT_ITEMS,  # pyright: ignore[reportInvalidTypeForm]
]
OpenAIResponsesInputParam: TypeAlias = list[OpenAIResponsesInputItemParam]


class OpenAIResponsesCreateRequest(ExcludeNoneModel):
    """The JSON parameters accepted by `AsyncOpenAI.responses.create`.

    Use `model_dump()` when sending a partially populated request so omitted
    optional parameters remain omitted from the payload.
    """

    background: Annotated[
        bool | None,
        Field(description="Whether to run the model response in the background."),
    ] = None
    context_management: Annotated[
        list[ContextManagement] | None,
        Field(description="Context management configuration for this request."),
    ] = None
    conversation: Annotated[
        Conversation | None,
        Field(description="The conversation that this response belongs to."),
    ] = None
    include: Annotated[
        list[ResponseIncludable] | None,
        Field(description="Additional output data to include in the response."),
    ] = None
    input: Annotated[
        str | OpenAIResponsesInputParam | None,
        Field(
            description=(
                "Text, image, file, or provider response-history items sent to the model."
            ),
        ),
    ] = None
    instructions: Annotated[
        str | None,
        Field(description="A system or developer message for the model."),
    ] = None
    max_output_tokens: Annotated[
        int | None,
        Field(description="The upper bound for generated tokens."),
    ] = None
    max_tool_calls: Annotated[
        int | None,
        Field(description="The maximum number of built-in tool calls."),
    ] = None
    metadata: Annotated[
        Metadata | None,
        Field(description="Key-value metadata attached to the response."),
    ] = None
    model: Annotated[
        ResponsesModel | None,
        Field(description="The model ID used to generate the response."),
    ] = None
    moderation: Annotated[
        Moderation | None,
        Field(description="Configuration for input and output moderation."),
    ] = None
    parallel_tool_calls: Annotated[
        bool | None,
        Field(description="Whether the model may run tool calls in parallel."),
    ] = None
    previous_response_id: Annotated[
        str | None,
        Field(description="The unique ID of a previous response."),
    ] = None
    prompt: Annotated[
        ResponsePromptParam | None,
        Field(description="A prompt template and its variables."),
    ] = None
    prompt_cache_key: Annotated[
        str | None,
        Field(description="A key used to optimize prompt-cache hits."),
    ] = None
    prompt_cache_options: Annotated[
        PromptCacheOptions | None,
        Field(description="Options for prompt caching."),
    ] = None
    prompt_cache_retention: Annotated[
        Literal["in_memory", "24h"] | None,
        Field(description="The prompt-cache retention policy."),
    ] = None
    reasoning: Annotated[
        Reasoning | None,
        Field(description="Reasoning configuration for supported models."),
    ] = None
    safety_identifier: Annotated[
        str | None,
        Field(description="A stable identifier for a user or application user."),
    ] = None
    service_tier: Annotated[
        Literal["auto", "default", "flex", "scale", "priority", "fast"] | None,
        Field(description="The processing service tier for the request."),
    ] = None
    store: Annotated[
        bool | None,
        Field(description="Whether to store the generated response."),
    ] = None
    stream: Annotated[
        Literal[True, False] | None,
        Field(description="Whether to stream the response."),
    ] = None
    stream_options: Annotated[
        StreamOptions | None,
        Field(description="Options for streaming responses."),
    ] = None
    temperature: Annotated[
        float | None,
        Field(description="Sampling temperature between 0 and 2."),
    ] = None
    text: Annotated[
        ResponseTextConfigParam | None,
        Field(description="Configuration for a text response."),
    ] = None
    tool_choice: Annotated[
        ToolChoice | None,
        Field(description="How the model should select tools."),
    ] = None
    tools: Annotated[
        list[ToolParam] | None,
        Field(description="Tools available to the model."),
    ] = None
    top_logprobs: Annotated[
        int | None,
        Field(description="The maximum number of token log probabilities."),
    ] = None
    top_p: Annotated[
        float | None,
        Field(description="The nucleus-sampling probability mass."),
    ] = None
    truncation: Annotated[
        Literal["auto", "disabled"] | None,
        Field(description="The context-window truncation strategy."),
    ] = None
    user: Annotated[
        str | None,
        Field(description="A stable identifier for the end user."),
    ] = None


class OpenAIResponseCompletedEvent(ResponseCompletedEvent):
    """ResponseCompletedEvent but with usage that is set to zero when null."""

    @model_validator(mode="before")
    @classmethod
    def set_usage_to_zero(cls, data: dict[str, Any]) -> dict[str, Any]:
        usage = data.get("response", {}).get("usage")
        if (
            usage is not None
            and usage.get("input_tokens_details", {}).get("cache_write_tokens") is None
        ):
            usage["input_tokens_details"]["cache_write_tokens"] = 0
        return data


# Some compatible providers omit `cache_write_tokens`
_REPLACED_SDK_STREAM_ITEMS = {
    ResponseCompletedEvent,
}
_RETAINED_SDK_STREAM_ITEMS = tuple(
    item
    for item in get_args(get_args(ResponseStreamEvent)[0])
    if item not in _REPLACED_SDK_STREAM_ITEMS
)
OpenAIResponsesStreamEvent: TypeAlias = Annotated[
    Union[
        OpenAIResponseCompletedEvent,
        *_RETAINED_SDK_STREAM_ITEMS,  # pyright: ignore[reportInvalidTypeForm]
    ],
    Discriminator("type"),
]

# These are aliases rather than duplicate classes intentionally: the SDK's response
# models are already Pydantic models generated from the same schema as the endpoint.
OpenAIResponsesResponse = Response
# OpenAIResponsesStreamEvent = ResponseStreamEvent
OpenAIResponsesStreamEventAdapter = TypeAdapter(OpenAIResponsesStreamEvent)
OpenAIResponsesStream = AsyncStream[OpenAIResponsesStreamEvent]
OpenAIResponsesCreateResult = OpenAIResponsesResponse | OpenAIResponsesStream


__all__ = [
    "OpenAIResponsesCreateRequest",
    "OpenAIResponsesCreateResult",
    "OpenAIResponsesInputItemParam",
    "OpenAIResponsesInputParam",
    "ResponseOutputMessageOptionalIDParam",
    "ResponseReasoningItemOptionalIDParam",
    "OpenAIResponsesResponse",
    "OpenAIResponsesStream",
    "OpenAIResponsesStreamEvent",
]
