"""Pydantic models for `AsyncOpenAI.responses.create`.

The nested parameter types are imported from the pinned OpenAI SDK.  They are
generated from OpenAI's API schema and preserve the SDK's discriminated unions,
while this module provides Pydantic entry points for request bodies and results.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, Union, get_args

from openai import AsyncStream
from openai.types.responses import Response, ResponseCompletedEvent, ResponseStreamEvent
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_create_params import (
    ContextManagement,
    Conversation,
    Moderation,
    PromptCacheOptions,
    StreamOptions,
    ToolChoice,
)
from openai.types.responses.response_function_call_arguments_done_event import (
    ResponseFunctionCallArgumentsDoneEvent,
)
from openai.types.responses.response_function_web_search_param import (
    ActionFind,
    ActionOpenPage,
    ActionSearch,
    ResponseFunctionWebSearchParam,
)
from openai.types.responses.response_includable import ResponseIncludable
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_param import Message as SDKResponseInputMessageParam
from openai.types.responses.response_input_param import ResponseInputItemParam
from openai.types.responses.response_input_text_param import (
    PromptCacheBreakpoint,
    ResponseInputTextParam,
)
from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
from openai.types.responses.response_output_refusal_param import ResponseOutputRefusalParam
from openai.types.responses.response_output_text_param import (
    Annotation,
    AnnotationURLCitation,
    ResponseOutputTextParam,
)
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
from pydantic import ConfigDict, Discriminator, Field, TypeAdapter, model_validator, with_config
from typing_extensions import Required, TypedDict

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


class AnthropicSearchSourceMetadataParam(TypedDict, total=False):
    """Original Messages search-result state needed for Anthropic continuation."""

    encrypted_content: Required[str]
    title: str | None
    page_age: str | None


# Nested TypedDicts inherit the enclosing open source's `extra="allow"`. Molto's own
# continuation state forbids unknown keys explicitly, and its nested types inherit that.
@with_config(ConfigDict(extra="forbid"))
class ResponseSearchSourceExtraContentParam(TypedDict):
    anthropic: AnthropicSearchSourceMetadataParam


@with_config(ConfigDict(extra="allow"))
class ResponseWebSearchSourceParam(TypedDict, total=False):
    """A search source of any type, such as a URL or OpenAI's `oai-weather` `api` source.

    The SDK declares only URL sources, but OpenAI returns others, so source types and
    fields are left for the upstream to validate. Only Molto's own `extra_content` is
    checked, because Molto consumes it and no upstream can.
    """

    extra_content: ResponseSearchSourceExtraContentParam


class ResponseWebSearchActionSearchParam(ActionSearch, total=False):
    sources: list[ResponseWebSearchSourceParam]


class ResponseFunctionWebSearchHistoryParam(ResponseFunctionWebSearchParam):
    """Search history carrying original provider state alongside native sources."""

    action: Required[ResponseWebSearchActionSearchParam | ActionOpenPage | ActionFind]


class AnthropicCitationMetadataParam(TypedDict):
    """Original Messages citation state needed for Anthropic continuation."""

    encrypted_index: str
    cited_text: str


class ResponseCitationExtraContentParam(TypedDict):
    anthropic: AnthropicCitationMetadataParam


class ResponseURLCitationParam(AnnotationURLCitation, total=False):
    extra_content: ResponseCitationExtraContentParam


_RETAINED_SDK_ANNOTATIONS = tuple(
    annotation for annotation in get_args(Annotation) if annotation is not AnnotationURLCitation
)
ResponseAnnotationParam: TypeAlias = Union[
    ResponseURLCitationParam,
    *_RETAINED_SDK_ANNOTATIONS,  # pyright: ignore[reportInvalidTypeForm]
]


class ResponseOutputTextCacheBreakpointParam(ResponseOutputTextParam, total=False):
    """Output history with cache markers and typed provider citation state."""

    prompt_cache_breakpoint: PromptCacheBreakpoint
    # Eager lists prevent SDK Iterable validation from discarding history entries
    # while evaluating union branches or consuming a repeated serialization.
    annotations: Required[list[ResponseAnnotationParam]]


class ResponseOutputMessageOptionalIDParam(ExcludeNoneModel):
    """Assistant output history accepted by compatible Responses APIs without an ID."""

    id: Annotated[
        str | None,
        Field(description="The optional output message ID."),
    ] = None
    content: Annotated[
        list[ResponseOutputTextCacheBreakpointParam | ResponseOutputRefusalParam],
        Field(description="The assistant message content."),
    ]
    role: Annotated[Literal["assistant"], Field(description="The output message role.")]
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"],
        Field(description="The output message status."),
    ]
    type: Annotated[Literal["message"], Field(description="The output message discriminator.")]
    phase: Annotated[
        Literal["commentary", "final_answer"] | None,
        Field(description="The assistant message phase."),
    ] = None


class ResponseInputImageOptionalDetailParam(ResponseInputImageParam, total=False):
    """An input image whose SDK-required `detail` may be omitted.

    The SDK marks `detail` as required, but the API defaults it to `auto` and
    compatible providers omit it, so the passthrough boundary preserves the omission.
    """

    detail: Literal["low", "high", "auto", "original"]


OpenAIResponsesInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageOptionalDetailParam,
    ResponseInputFileParam,
]


class ResponseEasyInputMessageParam(EasyInputMessageParam):
    """The SDK easy input message with the detail-optional image content part."""

    content: Required[str | list[OpenAIResponsesInputContentParam]]


class ResponseInputMessageParam(SDKResponseInputMessageParam):
    """The SDK typed input message with the detail-optional image content part."""

    content: Required[list[OpenAIResponsesInputContentParam]]


# The SDK input items are retained as-is so native passthrough accepts every
# variant the SDK models, including `file_id` images and `input_file` parts.
# Only five items are replaced: the two message wrappers re-point their content
# lists at the detail-optional image, and compatible providers also omit IDs
# from reasoning and output-history items. Output history also accepts cache
# markers supported by the live API and Anthropic citation state; search history
# retains Anthropic source state. Translatability is enforced by the adapter.
_REPLACED_SDK_INPUT_ITEMS = {
    EasyInputMessageParam,
    SDKResponseInputMessageParam,
    ResponseReasoningItemParam,
    ResponseOutputMessageParam,
    ResponseFunctionWebSearchParam,
}
_RETAINED_SDK_INPUT_ITEMS = tuple(
    item for item in get_args(ResponseInputItemParam) if item not in _REPLACED_SDK_INPUT_ITEMS
)
OpenAIResponsesInputItemParam: TypeAlias = Union[
    ResponseReasoningItemOptionalIDParam,
    ResponseOutputMessageOptionalIDParam,
    ResponseEasyInputMessageParam,
    ResponseInputMessageParam,
    ResponseFunctionWebSearchHistoryParam,
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
            description=("Text, image, or provider response-history items sent to the model."),
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

    def has_anthropic_search_metadata(self) -> bool:
        """Report provider-owned search state for upstream route validation."""
        if not isinstance(self.input, list):
            return False
        for item in self.input:
            if isinstance(item, ResponseOutputMessageOptionalIDParam):
                for part in item.content:
                    if part["type"] == "output_text" and any(
                        "extra_content" in annotation for annotation in part["annotations"]
                    ):
                        return True
            elif isinstance(item, dict) and item.get("type") == "web_search_call":
                if any("extra_content" in source for source in item["action"].get("sources", [])):
                    return True
        return False


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


class OpenAIResponseFunctionCallArgumentsDoneEvent(ResponseFunctionCallArgumentsDoneEvent):
    """Accept Responses-compatible streams that omit the redundant function name."""

    name: str | None = None


# Some compatible providers omit `cache_write_tokens`
_REPLACED_SDK_STREAM_ITEMS = {
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
}
_RETAINED_SDK_STREAM_ITEMS = tuple(
    item
    for item in get_args(get_args(ResponseStreamEvent)[0])
    if item not in _REPLACED_SDK_STREAM_ITEMS
)
OpenAIResponsesStreamEvent: TypeAlias = Annotated[
    Union[
        OpenAIResponseCompletedEvent,
        OpenAIResponseFunctionCallArgumentsDoneEvent,
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
    "OpenAIResponsesInputContentParam",
    "OpenAIResponsesInputItemParam",
    "OpenAIResponsesInputParam",
    "ResponseOutputMessageOptionalIDParam",
    "ResponseReasoningItemOptionalIDParam",
    "OpenAIResponsesResponse",
    "OpenAIResponsesStream",
    "OpenAIResponsesStreamEvent",
]
