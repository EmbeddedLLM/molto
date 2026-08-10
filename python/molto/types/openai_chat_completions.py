"""Pydantic models for ``AsyncOpenAI.chat.completions.create``.

The nested parameter and result types come from the pinned OpenAI SDK. They are
generated from OpenAI's API schema and preserve the SDK's discriminated unions,
while this module provides a single Pydantic entry point for chat completion
requests and typed result aliases for both response modes.
"""

from __future__ import annotations

from typing import Annotated, Literal

from openai import AsyncStream
from openai._types import SequenceNotStr
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.chat.chat_completion_stream_options_param import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam
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
from pydantic import Field

from molto.types.base import ExcludeNoneModel


class OpenAIChatCompletionsCreateRequest(ExcludeNoneModel):
    """The JSON parameters accepted by ``AsyncOpenAI.chat.completions.create``."""

    messages: Annotated[
        list[ChatCompletionMessageParam],
        Field(description="A list of messages comprising the conversation."),
    ]
    model: Annotated[
        str | ChatModel,
        Field(description="The model ID used to generate the completion."),
    ]
    audio: Annotated[
        ChatCompletionAudioParam | None,
        Field(default=None, description="Parameters for audio output."),
    ]
    frequency_penalty: Annotated[
        float | None,
        Field(default=None, description="A penalty for tokens based on their frequency."),
    ]
    function_call: Annotated[
        FunctionCall | None,
        Field(default=None, description="Deprecated control for function calling."),
    ]
    functions: Annotated[
        list[Function] | None,
        Field(default=None, description="Deprecated function definitions."),
    ]
    logit_bias: Annotated[
        dict[str, int] | None,
        Field(default=None, description="Token IDs and their associated likelihood bias."),
    ]
    logprobs: Annotated[
        bool | None,
        Field(default=None, description="Whether to return output token log probabilities."),
    ]
    max_completion_tokens: Annotated[
        int | None,
        Field(default=None, description="The upper bound for generated completion tokens."),
    ]
    max_tokens: Annotated[
        int | None,
        Field(default=None, description="Deprecated maximum number of generated tokens."),
    ]
    metadata: Annotated[
        Metadata | None,
        Field(default=None, description="Key-value metadata attached to the completion."),
    ]
    modalities: Annotated[
        list[Literal["text", "audio"]] | None,
        Field(default=None, description="Output modalities requested from the model."),
    ]
    moderation: Annotated[
        Moderation | None,
        Field(default=None, description="Input and output moderation configuration."),
    ]
    n: Annotated[
        int | None,
        Field(default=None, description="The number of completion choices to generate."),
    ]
    parallel_tool_calls: Annotated[
        bool | None,
        Field(default=None, description="Whether to enable parallel tool calls."),
    ]
    prediction: Annotated[
        ChatCompletionPredictionContentParam | None,
        Field(default=None, description="Static predicted output content."),
    ]
    presence_penalty: Annotated[
        float | None,
        Field(default=None, description="A penalty for tokens based on their presence."),
    ]
    prompt_cache_key: Annotated[
        str | None,
        Field(default=None, description="A key used to optimize prompt-cache hits."),
    ]
    prompt_cache_options: Annotated[
        PromptCacheOptions | None,
        Field(default=None, description="Options for prompt caching."),
    ]
    prompt_cache_retention: Annotated[
        Literal["in_memory", "24h"] | None,
        Field(default=None, description="The prompt-cache retention policy."),
    ]
    reasoning_effort: Annotated[
        ReasoningEffort | None,
        Field(default=None, description="Reasoning effort for supported models."),
    ]
    response_format: Annotated[
        ResponseFormat | None,
        Field(default=None, description="The format that the model must output."),
    ]
    safety_identifier: Annotated[
        str | None,
        Field(default=None, description="A stable identifier for an end user."),
    ]
    seed: Annotated[
        int | None,
        Field(default=None, description="A best-effort deterministic sampling seed."),
    ]
    service_tier: Annotated[
        Literal["auto", "default", "flex", "scale", "priority", "fast"] | None,
        Field(default=None, description="The processing service tier for the request."),
    ]
    stop: Annotated[
        str | SequenceNotStr[str] | None,
        Field(default=None, description="Up to four sequences that stop generation."),
    ]
    store: Annotated[
        bool | None,
        Field(default=None, description="Whether to store the completion."),
    ]
    stream: Annotated[
        Literal[True, False] | None,
        Field(default=None, description="Whether to stream the completion."),
    ]
    stream_options: Annotated[
        ChatCompletionStreamOptionsParam | None,
        Field(default=None, description="Options for streaming completions."),
    ]
    temperature: Annotated[
        float | None,
        Field(default=None, description="Sampling temperature between 0 and 2."),
    ]
    tool_choice: Annotated[
        ChatCompletionToolChoiceOptionParam | None,
        Field(default=None, description="How the model should select tools."),
    ]
    tools: Annotated[
        list[ChatCompletionToolUnionParam] | None,
        Field(default=None, description="Tools available to the model."),
    ]
    top_logprobs: Annotated[
        int | None,
        Field(default=None, description="The maximum number of token log probabilities."),
    ]
    top_p: Annotated[
        float | None,
        Field(default=None, description="The nucleus-sampling probability mass."),
    ]
    user: Annotated[
        str | None,
        Field(default=None, description="A stable identifier for the end user."),
    ]
    verbosity: Annotated[
        Literal["low", "medium", "high"] | None,
        Field(default=None, description="The response verbosity for supported models."),
    ]
    web_search_options: Annotated[
        WebSearchOptions | None,
        Field(default=None, description="Configuration for the web search tool."),
    ]


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
