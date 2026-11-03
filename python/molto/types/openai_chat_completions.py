"""Pydantic models for `AsyncOpenAI.chat.completions.create`.

The nested parameter and result types come from the pinned OpenAI SDK. They are
generated from OpenAI's API schema and preserve the SDK's discriminated unions,
while this module provides a single Pydantic entry point for chat completion
requests and typed result aliases for both response modes.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias, Union, get_args

from openai import AsyncStream
from openai._types import SequenceNotStr
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    File as ChatCompletionContentPartFileParam,
)
from openai.types.chat.chat_completion_content_part_refusal_param import (
    ChatCompletionContentPartRefusalParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)
from openai.types.chat.chat_completion_message_custom_tool_call_param import (
    ChatCompletionMessageCustomToolCallParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.chat.chat_completion_stream_options_param import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.completion_create_params import (
    Function,
    FunctionCall,
    Moderation,
    PromptCacheOptions,
    ResponseFormat,
    WebSearchOptions,
)
from openai.types.shared.chat_model import ChatModel
from openai.types.shared.metadata import Metadata
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import Field, JsonValue, SerializationInfo, TypeAdapter, field_serializer
from typing_extensions import NotRequired, Required, TypedDict

from molto.types.base import ExcludeNoneModel


class GoogleExtraContent(TypedDict):
    """Opaque Google continuation state, retained at its original attachment."""

    google: Required[dict[str, JsonValue]]


class ChatCompletionTextPartParam(ChatCompletionContentPartTextParam):
    """Text content with Google's per-part continuation extension."""

    extra_content: NotRequired[GoogleExtraContent]


class ChatCompletionImagePartParam(ChatCompletionContentPartImageParam):
    """Image content with Google's per-part continuation extension."""

    extra_content: NotRequired[GoogleExtraContent]


class ChatCompletionAudioPartParam(ChatCompletionContentPartInputAudioParam):
    """Audio content with Google's per-part continuation extension."""

    extra_content: NotRequired[GoogleExtraContent]


class ChatCompletionFilePartParam(ChatCompletionContentPartFileParam):
    """File content with Google's per-part continuation extension."""

    extra_content: NotRequired[GoogleExtraContent]


class ChatCompletionRefusalPartParam(ChatCompletionContentPartRefusalParam):
    """Refusal content with Google's per-part continuation extension."""

    extra_content: NotRequired[GoogleExtraContent]


class ChatCompletionFunctionToolCallParam(ChatCompletionMessageFunctionToolCallParam):
    """A function call retaining its own Google continuation state."""

    extra_content: NotRequired[GoogleExtraContent]


OpenAIChatCompletionContentPartParam: TypeAlias = Union[
    ChatCompletionTextPartParam,
    ChatCompletionImagePartParam,
    ChatCompletionAudioPartParam,
    ChatCompletionFilePartParam,
]


class ChatCompletionAssistantMessageWithReasoningParam(ChatCompletionAssistantMessageParam):
    """Provider-compatible assistant message containing either raw reasoning or summary."""

    reasoning_content: NotRequired[str]
    reasoning: NotRequired[str | list[str]]
    extra_content: NotRequired[GoogleExtraContent]
    tool_calls: NotRequired[
        list[ChatCompletionFunctionToolCallParam | ChatCompletionMessageCustomToolCallParam]
    ]
    content: NotRequired[
        str | list[ChatCompletionTextPartParam | ChatCompletionRefusalPartParam] | None
    ]


class ChatCompletionUserMessageListParam(TypedDict):
    """User message with eagerly validated, repeatable content parts."""

    content: Required[str | list[OpenAIChatCompletionContentPartParam]]
    role: Required[Literal["user"]]
    name: NotRequired[str]
    extra_content: NotRequired[GoogleExtraContent]


class ChatCompletionSystemMessageListParam(TypedDict):
    """System message with eagerly validated, repeatable content parts."""

    content: Required[str | list[ChatCompletionTextPartParam]]
    role: Required[Literal["system"]]
    name: NotRequired[str]
    extra_content: NotRequired[GoogleExtraContent]


class ChatCompletionDeveloperMessageListParam(TypedDict):
    """Developer message with eagerly validated, repeatable content parts."""

    content: Required[str | list[ChatCompletionTextPartParam]]
    role: Required[Literal["developer"]]
    name: NotRequired[str]
    extra_content: NotRequired[GoogleExtraContent]


class ChatCompletionToolMessageListParam(TypedDict):
    """Tool message with eagerly validated, repeatable text content parts."""

    content: Required[str | list[ChatCompletionTextPartParam]]
    role: Required[Literal["tool"]]
    tool_call_id: Required[str]
    extra_content: NotRequired[GoogleExtraContent]


_RETAINED_CHAT_MESSAGE_PARAMS = tuple(
    item
    for item in get_args(ChatCompletionMessageParam)
    if item
    not in {
        ChatCompletionAssistantMessageParam,
        ChatCompletionDeveloperMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionUserMessageParam,
    }
)
OpenAIChatCompletionMessageParam: TypeAlias = Union[
    ChatCompletionDeveloperMessageListParam,
    ChatCompletionSystemMessageListParam,
    ChatCompletionUserMessageListParam,
    ChatCompletionToolMessageListParam,
    ChatCompletionAssistantMessageWithReasoningParam,
    *_RETAINED_CHAT_MESSAGE_PARAMS,  # pyright: ignore[reportInvalidTypeForm]
]
_CHAT_MESSAGES_ADAPTER = TypeAdapter(list[OpenAIChatCompletionMessageParam])


class OpenAIChatCompletionsCreateRequest(ExcludeNoneModel):
    """The JSON parameters accepted by `AsyncOpenAI.chat.completions.create`."""

    messages: Annotated[
        list[OpenAIChatCompletionMessageParam],
        Field(description="A list of messages comprising the conversation."),
    ]
    model: Annotated[
        str | ChatModel,
        Field(description="The model ID used to generate the completion."),
    ]
    audio: Annotated[
        ChatCompletionAudioParam | None,
        Field(description="Parameters for audio output."),
    ] = None
    frequency_penalty: Annotated[
        float | None,
        Field(description="A penalty for tokens based on their frequency."),
    ] = None
    function_call: Annotated[
        FunctionCall | None,
        Field(description="Deprecated control for function calling."),
    ] = None
    functions: Annotated[
        list[Function] | None,
        Field(description="Deprecated function definitions."),
    ] = None
    logit_bias: Annotated[
        dict[str, int] | None,
        Field(description="Token IDs and their associated likelihood bias."),
    ] = None
    logprobs: Annotated[
        bool | None,
        Field(description="Whether to return output token log probabilities."),
    ] = None
    max_completion_tokens: Annotated[
        int | None,
        Field(description="The upper bound for generated completion tokens."),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(description="Deprecated maximum number of generated tokens."),
    ] = None
    metadata: Annotated[
        Metadata | None,
        Field(description="Key-value metadata attached to the completion."),
    ] = None
    modalities: Annotated[
        list[Literal["text", "audio"]] | None,
        Field(description="Output modalities requested from the model."),
    ] = None
    moderation: Annotated[
        Moderation | None,
        Field(description="Input and output moderation configuration."),
    ] = None
    n: Annotated[
        int | None,
        Field(description="The number of completion choices to generate."),
    ] = None
    parallel_tool_calls: Annotated[
        bool | None,
        Field(description="Whether to enable parallel tool calls."),
    ] = None
    prediction: Annotated[
        ChatCompletionPredictionContentParam | None,
        Field(description="Static predicted output content."),
    ] = None
    presence_penalty: Annotated[
        float | None,
        Field(description="A penalty for tokens based on their presence."),
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
    reasoning_effort: Annotated[
        ReasoningEffort | None,
        Field(description="Reasoning effort for supported models."),
    ] = None
    response_format: Annotated[
        ResponseFormat | None,
        Field(description="The format that the model must output."),
    ] = None
    safety_identifier: Annotated[
        str | None,
        Field(description="A stable identifier for an end user."),
    ] = None
    seed: Annotated[
        int | None,
        Field(description="A best-effort deterministic sampling seed."),
    ] = None
    service_tier: Annotated[
        Literal["auto", "default", "flex", "scale", "priority", "fast"] | None,
        Field(description="The processing service tier for the request."),
    ] = None
    stop: Annotated[
        str | SequenceNotStr[str] | None,
        Field(description="Up to four sequences that stop generation."),
    ] = None
    store: Annotated[
        bool | None,
        Field(description="Whether to store the completion."),
    ] = None
    stream: Annotated[
        Literal[True, False] | None,
        Field(description="Whether to stream the completion."),
    ] = None
    stream_options: Annotated[
        ChatCompletionStreamOptionsParam | None,
        Field(description="Options for streaming completions."),
    ] = None
    temperature: Annotated[
        float | None,
        Field(description="Sampling temperature between 0 and 2."),
    ] = None
    tool_choice: Annotated[
        ChatCompletionToolChoiceOptionParam | None,
        Field(description="How the model should select tools."),
    ] = None
    tools: Annotated[
        list[ChatCompletionToolUnionParam] | None,
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
    user: Annotated[
        str | None,
        Field(description="A stable identifier for the end user."),
    ] = None
    verbosity: Annotated[
        Literal["low", "medium", "high"] | None,
        Field(description="The response verbosity for supported models."),
    ] = None
    web_search_options: Annotated[
        WebSearchOptions | None,
        Field(description="Configuration for the web search tool."),
    ] = None

    @field_serializer("messages")
    def serialize_messages(self, messages, info: SerializationInfo):
        # Explicit native history nulls are wire values, unlike unset optional
        # request parameters. In particular, tool-only assistant content is null.
        return _CHAT_MESSAGES_ADAPTER.dump_python(
            messages,
            mode=info.mode,
            include=info.include,
            exclude=info.exclude,
            by_alias=info.by_alias,
            exclude_unset=info.exclude_unset,
            exclude_defaults=info.exclude_defaults,
            exclude_none=False,
        )


# The SDK's response classes are already Pydantic models generated from the
# endpoint schema, so aliases preserve the full nested response structures.
OpenAIChatCompletionsResponse = ChatCompletion
OpenAIChatCompletionsStreamChunk = ChatCompletionChunk
OpenAIChatCompletionsStream = AsyncStream[OpenAIChatCompletionsStreamChunk]
OpenAIChatCompletionsCreateResult = OpenAIChatCompletionsResponse | OpenAIChatCompletionsStream


__all__ = [
    "OpenAIChatCompletionsCreateRequest",
    "OpenAIChatCompletionsCreateResult",
    "OpenAIChatCompletionsResponse",
    "OpenAIChatCompletionsStream",
    "OpenAIChatCompletionsStreamChunk",
]
