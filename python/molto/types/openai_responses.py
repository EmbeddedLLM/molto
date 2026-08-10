"""Pydantic models for `AsyncOpenAI.responses.create`.

The nested parameter types are imported from the pinned OpenAI SDK.  They are
generated from OpenAI's API schema and preserve the SDK's discriminated unions,
while this module provides Pydantic entry points for request bodies and results.
"""

from __future__ import annotations

from types import MethodType
from typing import Annotated, Any, Literal

from openai import AsyncStream
from openai.types.responses.response import Response
from openai.types.responses.response_create_params import (
    ContextManagement,
    Conversation,
    Moderation,
    PromptCacheOptions,
    StreamOptions,
    ToolChoice,
)
from openai.types.responses.response_includable import ResponseIncludable
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.metadata import Metadata
from openai.types.shared.reasoning import Reasoning
from openai.types.shared.responses_model import ResponsesModel
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from molto.types.base import _Model
from molto.utils.io import json_dumps, json_loads

RECONSTRUCTIBLE_RESPONSE_FIELDS = {
    "id",
    "model",
    "object",
    "output",
    "status",
    "usage",
}


class OpenAIResponsesCreateRequest(_Model):
    """The JSON parameters accepted by `AsyncOpenAI.responses.create`.

    Use `model_dump()` when sending a partially populated request so omitted
    optional parameters remain omitted from the payload.
    """

    background: Annotated[
        bool | None,
        Field(default=None, description="Whether to run the model response in the background."),
    ]
    context_management: Annotated[
        list[ContextManagement] | None,
        Field(default=None, description="Context management configuration for this request."),
    ]
    conversation: Annotated[
        Conversation | None,
        Field(description="The conversation that this response belongs to.", default=None),
    ]
    include: Annotated[
        list[ResponseIncludable] | None,
        Field(default=None, description="Additional output data to include in the response."),
    ]
    input: Annotated[
        str | ResponseInputParam | None,
        Field(default=None, description="Text, image, or file inputs to the model."),
    ]
    instructions: Annotated[
        str | None,
        Field(default=None, description="A system or developer message for the model."),
    ]
    max_output_tokens: Annotated[
        int | None,
        Field(default=None, description="The upper bound for generated tokens."),
    ]
    max_tool_calls: Annotated[
        int | None,
        Field(default=None, description="The maximum number of built-in tool calls."),
    ]
    metadata: Annotated[
        Metadata | None,
        Field(default=None, description="Key-value metadata attached to the response."),
    ]
    model: Annotated[
        ResponsesModel | None,
        Field(default=None, description="The model ID used to generate the response."),
    ]
    moderation: Annotated[
        Moderation | None,
        Field(default=None, description="Configuration for input and output moderation."),
    ]
    parallel_tool_calls: Annotated[
        bool | None,
        Field(default=None, description="Whether the model may run tool calls in parallel."),
    ]
    previous_response_id: Annotated[
        str | None,
        Field(default=None, description="The unique ID of a previous response."),
    ]
    prompt: Annotated[
        ResponsePromptParam | None,
        Field(default=None, description="A prompt template and its variables."),
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
    reasoning: Annotated[
        Reasoning | None,
        Field(default=None, description="Reasoning configuration for supported models."),
    ]
    safety_identifier: Annotated[
        str | None,
        Field(default=None, description="A stable identifier for a user or application user."),
    ]
    service_tier: Annotated[
        Literal["auto", "default", "flex", "scale", "priority", "fast"] | None,
        Field(default=None, description="The processing service tier for the request."),
    ]
    store: Annotated[
        bool | None,
        Field(default=None, description="Whether to store the generated response."),
    ]
    stream: Annotated[
        Literal[True, False] | None,
        Field(default=None, description="Whether to stream the response."),
    ]
    stream_options: Annotated[
        StreamOptions | None,
        Field(default=None, description="Options for streaming responses."),
    ]
    temperature: Annotated[
        float | None,
        Field(default=None, description="Sampling temperature between 0 and 2."),
    ]
    text: Annotated[
        ResponseTextConfigParam | None,
        Field(default=None, description="Configuration for a text response."),
    ]
    tool_choice: Annotated[
        ToolChoice | None,
        Field(default=None, description="How the model should select tools."),
    ]
    tools: Annotated[
        list[ToolParam] | None,
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
    truncation: Annotated[
        Literal["auto", "disabled"] | None,
        Field(default=None, description="The context-window truncation strategy."),
    ]
    user: Annotated[
        str | None,
        Field(default=None, description="A stable identifier for the end user."),
    ]


# These are aliases rather than duplicate classes intentionally: the SDK's response
# models are already Pydantic models generated from the same schema as the endpoint.
OpenAIResponsesResponse = Response
OpenAIResponsesStreamEvent = Annotated[
    ResponseStreamEvent,
    Field(description="A discriminated event yielded by a streaming response."),
]
OpenAIResponsesStream = AsyncStream[OpenAIResponsesStreamEvent]
OpenAIResponsesCreateResult = OpenAIResponsesResponse | OpenAIResponsesStream

_STREAM_EVENT_ADAPTER = TypeAdapter(OpenAIResponsesStreamEvent)


class _OpenAIResponsesPOCStreamEvent(_Model):
    """Fallback for newer wire events not modeled by the repository's pinned SDK."""

    type: str


def openai_responses_event_from_sse(sse: str) -> ResponseStreamEvent:
    """Parse one Responses SSE record while retaining its original wire representation."""
    data = "\n".join(
        line.removeprefix("data:").removeprefix(" ")
        for line in sse.splitlines()
        if line.startswith("data:")
    )
    try:
        event = _STREAM_EVENT_ADAPTER.validate_json(data)
    except ValidationError:
        event = _OpenAIResponsesPOCStreamEvent.model_validate(json_loads(data))

    def as_sse(_: BaseModel) -> str:
        return sse

    # OpenAI's generated models allow extra runtime attributes. Attaching the raw
    # record keeps the concrete SDK event type while exposing Molto's wire helper.
    object.__setattr__(event, "as_sse", MethodType(as_sse, event))
    return event


def openai_responses_event_from_data(data: dict[str, Any]) -> BaseModel:
    """Build one compact Responses SSE event from translated data."""
    event_type = data["type"]
    return openai_responses_event_from_sse(f"event: {event_type}\ndata: {json_dumps(data)}\n\n")


__all__ = [
    "OpenAIResponsesCreateRequest",
    "OpenAIResponsesCreateResult",
    "OpenAIResponsesResponse",
    "OpenAIResponsesStream",
    "OpenAIResponsesStreamEvent",
    "openai_responses_event_from_data",
    "openai_responses_event_from_sse",
]
