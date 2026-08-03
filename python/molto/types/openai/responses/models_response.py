"""OpenAI Responses response and stream-event API models."""

from __future__ import annotations

import json
from time import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from molto.utils import uuid7_str

from .models_items import (
    OpenAIConversation,
    OpenAIOutput,
    OpenAIOutputRefusal,
    OpenAIOutputTextContent,
    OpenAIPromptTemplate,
    OpenAIReasoningContent,
    OpenAIReasoningSummary,
    OpenAIResponseReasoning,
    OpenAIResponsesOutputTool,
    OpenAITextConfig,
    OpenAITextFormat,
    OpenAIToolChoice,
)


class OpenAIResponsesResponseError(BaseModel):
    code: Annotated[
        str,
        Field(description="The error code for the response."),
    ]
    message: Annotated[
        str,
        Field(description="A human-readable description of the error."),
    ]


class OpenAIResponsesIncompleteDetails(BaseModel):
    """Details about why the response is incomplete."""

    reason: Annotated[
        str,
        Field(description="The reason why the response is incomplete."),
    ]


class OpenAIInputTokenDetails(BaseModel):
    """A detailed breakdown of the input tokens."""

    model_config = ConfigDict(extra="allow")

    cached_tokens: Annotated[
        int | None,
        Field(description="The number of tokens that were retrieved from the cache."),
    ] = None


class OpenAIOutputTokenDetails(BaseModel):
    """A detailed breakdown of the output tokens."""

    model_config = ConfigDict(extra="allow")

    reasoning_tokens: Annotated[
        int | None,
        Field(description="The number of reasoning tokens."),
    ] = None


class OpenAIResponsesUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: Annotated[
        int,
        Field(description="The number of input tokens."),
    ]
    input_tokens_details: Annotated[
        OpenAIInputTokenDetails | None,
        Field(description="A detailed breakdown of the input tokens."),
    ] = None
    output_tokens: Annotated[
        int,
        Field(description="The number of output tokens."),
    ]
    output_tokens_details: Annotated[
        OpenAIOutputTokenDetails | None,
        Field(description="A detailed breakdown of the output tokens."),
    ] = None
    total_tokens: Annotated[
        int,
        Field(description="The total number of tokens used."),
    ]


class OpenAIResponsesResponse(BaseModel):
    # Response payloads can gain provider fields before Molto's request contract
    # does. Preserve those populated fields for same-format parity and classify
    # them in the compatibility matrix before translating them cross-output.
    model_config = ConfigDict(extra="allow")

    background: Annotated[
        bool,
        Field(
            description="Whether to run the model response in the background.",
        ),
    ] = False
    completed_at: Annotated[
        float | None,
        Field(
            description=("Unix timestamp (in seconds) of when this Response was completed."),
        ),
    ] = None
    conversation: Annotated[
        OpenAIConversation | None,
        Field(
            description=(
                "The conversation that this response belongs to. "
                "Input items and output items from this response are automatically added to this conversation."
            )
        ),
    ] = None
    created_at: Annotated[
        float,
        Field(
            default_factory=lambda: int(time()),
            description=("Unix timestamp (in seconds) of when this Response was created."),
        ),
    ]
    error: Annotated[
        OpenAIResponsesResponseError | None,
        Field(description="An error object returned when the model fails to generate a Response."),
    ] = None
    id: Annotated[
        str,
        Field(
            default_factory=lambda: uuid7_str("resp_"),
            description="Unique identifier for this Response.",
        ),
    ]
    incomplete_details: Annotated[
        OpenAIResponsesIncompleteDetails | None,
        Field(description="Details about why the response is incomplete."),
    ] = None
    instructions: Annotated[
        str | None,
        Field(
            description=(
                "A system (or developer) message inserted into the model's context. "
                "When using along with `previous_response_id`, the instructions from a previous response will not be carried over to the next response. "
                "This makes it simple to swap out system (or developer) messages in new responses."
            ),
        ),
    ] = None
    max_output_tokens: Annotated[
        int | None,
        Field(
            description=(
                "An upper bound for the number of tokens that can be generated for a response, "
                "including visible output tokens and reasoning tokens."
            ),
        ),
    ] = None
    max_tool_calls: Annotated[
        int | None,
        Field(
            description=(
                "The maximum number of total calls to native tools that can be processed in a response. "
                "This maximum number applies across all native tool calls, not per individual tool. "
                "Any further attempts to call a tool by the model will be ignored. "
                "As of 2026-02-22, observed OpenAI runtime behavior is not always consistent with this documented rule across model/tool paths."
            ),
        ),
    ] = None
    metadata: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Set of key-value pairs that can be attached to an object. "
                "This can be useful for storing additional information about the object in a structured format."
            ),
        ),
    ] = Field(default_factory=dict)
    model: Annotated[
        str,
        Field(description="Model ID used to generate the response."),
    ]
    object: Annotated[
        Literal["response"],
        Field(description="The object type of this resource. Always `response`."),
    ] = "response"
    output: Annotated[
        list[OpenAIOutput],
        Field(
            description=(
                "An array of content items generated by the model. "
                "The length and order of items in the output array is dependent on the model's response."
            )
        ),
    ] = Field(default_factory=list)
    parallel_tool_calls: Annotated[
        bool,
        Field(
            description="Whether to allow the model to run tool calls in parallel.",
        ),
    ] = True
    previous_response_id: Annotated[
        str | None,
        Field(
            description=(
                "The unique ID of the previous response to the model. "
                "Use this to create multi-turn conversations. "
                "Cannot be used in conjunction with conversation."
            ),
        ),
    ] = None
    prompt: Annotated[
        OpenAIPromptTemplate | None,
        Field(
            description=("Reference to a prompt template and its variables."),
        ),
    ] = None
    prompt_cache_key: Annotated[
        str | None,
        Field(
            description=(
                "Used to cache responses for similar requests to optimize your cache hit rates. "
                "Replaces the `user` field."
            ),
        ),
    ] = None
    prompt_cache_retention: Annotated[
        Literal["in_memory", "24h"] | None,
        Field(
            description=(
                "The retention policy for the prompt cache. "
                "Set to `24h` to enable extended prompt caching, "
                "which keeps cached prefixes active for longer, up to a maximum of 24 hours."
            ),
        ),
    ] = None
    reasoning: Annotated[
        OpenAIResponseReasoning,
        Field(
            description=(
                "Configuration options for reasoning models. "
                "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response."
            )
        ),
    ] = Field(default_factory=OpenAIResponseReasoning)
    safety_identifier: Annotated[
        str | None,
        Field(
            description=(
                "A stable identifier used to help detect users of your application that may be violating usage policies. "
                "The IDs should be a string that uniquely identifies each user. "
                "We recommend hashing their username or email address, in order to avoid sending us any identifying information."
            ),
        ),
    ] = None
    service_tier: Annotated[
        Literal["auto", "default", "flex", "scale", "priority"],
        Field(
            description=("Specifies the processing type used for serving the request."),
        ),
    ] = "default"
    store: Annotated[
        bool,
        Field(
            description="Whether to store the generated model response for later retrieval via API."
        ),
    ] = True
    status: Annotated[
        Literal["completed", "failed", "in_progress", "cancelled", "queued", "incomplete"],
        Field(
            description=(
                "The status of the response generation. "
                "One of `completed`, `failed`, `in_progress`, `cancelled`, `queued`, or `incomplete`."
            ),
        ),
    ] = "in_progress"
    temperature: Annotated[
        float | None,
        Field(description=("What sampling temperature to use.")),
    ] = None
    text: Annotated[
        OpenAITextConfig,
        Field(
            description=(
                "Configuration options for a text response from the model. "
                "Can be plain text or structured JSON data."
            ),
        ),
    ] = Field(
        default_factory=lambda: OpenAITextConfig(format=OpenAITextFormat(), verbosity="medium")
    )
    tool_choice: Annotated[
        OpenAIToolChoice,
        Field(description=("Controls which (if any) tool is called by the model.")),
    ] = "auto"
    tools: Annotated[
        list[OpenAIResponsesOutputTool],
        Field(description=("An array of tools the model may call while generating a response.")),
    ] = Field(default_factory=list)
    presence_penalty: Annotated[
        float,
        Field(
            description="Unused by Responses API; always 0 in this gateway (schema-compat default)."
        ),
    ] = 0.0
    frequency_penalty: Annotated[
        float,
        Field(
            description="Unused by Responses API; always 0 in this gateway (schema-compat default)."
        ),
    ] = 0.0
    top_logprobs: Annotated[
        int,
        Field(
            description=(
                "An integer specifying the number of most likely tokens to return at each token position, "
                "each with an associated log probability."
            ),
        ),
    ] = 0
    top_p: Annotated[
        float | None,
        Field(
            ge=0,
            le=1,
            description=(
                "An alternative to sampling with `temperature`, called nucleus sampling, "
                "where the model considers the results of the tokens with `top_p` probability mass."
            ),
        ),
    ] = None
    truncation: Annotated[
        Literal["auto", "disabled"],
        Field(description=("The truncation strategy to use for the model response.")),
    ] = "disabled"
    usage: Annotated[
        OpenAIResponsesUsage | None,
        Field(
            description=(
                "Represents token usage details including input tokens, output tokens, a breakdown of output tokens, and the total tokens used."
            )
        ),
    ] = None


class OpenAIResponsesStreamEvent(BaseModel):
    # Preserve newly populated provider event fields (for example stream
    # obfuscation) so same-format replay exposes SDK additions instead of
    # silently dropping them.
    model_config = ConfigDict(extra="allow")

    type: Annotated[
        str,
        Field(description=("The type of the event.")),
    ]
    sequence_number: Annotated[
        int,
        Field(description=("The sequence number for this event.")),
    ]

    def as_responses_chunk(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json(exclude_none=True)}\n\n"


class OpenAIResponsesError(OpenAIResponsesStreamEvent):
    """
    An error event returned when the model fails to generate a Response.
    Also used for non-streaming error.
    """

    code: Annotated[
        str | None,
        Field(description=("The error code.")),
    ] = None
    message: Annotated[
        str,
        Field(description=("The error message.")),
    ]
    param: Annotated[
        str | None,
        Field(description=("The error parameter.")),
    ] = None
    type: Annotated[
        Literal["error"],
        Field(description=("The type of the event.")),
    ] = "error"


class OpenAIResponsesStream(OpenAIResponsesStreamEvent):
    response: Annotated[
        OpenAIResponsesResponse,
        Field(description=("The response.")),
    ]
    type: Annotated[
        Literal[
            "response.created",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
            "response.queued",
        ],
        Field(description=("The type of the event.")),
    ]

    def as_responses_chunk(self) -> str:
        # OpenAI Responses operational parity:
        # - Output stream events embed a full ResponseResource object.
        # - Many keys are required even when their values are `null`.
        # So we must not drop `None` values from the embedded response.
        payload = dict(
            type=self.type,
            sequence_number=self.sequence_number,
            response=self.response.model_dump(mode="python", exclude_none=False),
        )
        _ensure_code_interpreter_outputs_key(payload)
        return f"event: {self.type}\ndata: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


class OpenAIResponsesStreamOutput(OpenAIResponsesStreamEvent):
    item: Annotated[
        OpenAIOutput,
        Field(description=("The output item.")),
    ]
    output_index: Annotated[
        int,
        Field(description=("The index of the output item that was added.")),
    ]
    type: Annotated[
        Literal[
            "response.output_item.added",
            "response.output_item.done",
        ],
        Field(description=("The type of the event.")),
    ]

    def as_responses_chunk(self) -> str:
        payload = self.model_dump(mode="python", exclude_none=True)
        _ensure_code_interpreter_outputs_key(payload)
        return f"event: {self.type}\ndata: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


class OpenAIResponsesStreamImageGeneration(OpenAIResponsesStreamEvent):
    background: Annotated[
        Literal["transparent", "opaque", "auto"] | None,
        Field(description="The background mode used for the partial image."),
    ] = None
    item_id: Annotated[
        str,
        Field(description="The ID of the image generation output item."),
    ]
    output_format: Annotated[
        Literal["png", "webp", "jpeg"] | None,
        Field(description="The encoded partial-image format."),
    ] = None
    output_index: Annotated[
        int,
        Field(description="The index of the image generation output item."),
    ]
    partial_image_b64: Annotated[
        str | None,
        Field(description="A base64-encoded partial image."),
    ] = None
    partial_image_index: Annotated[
        int | None,
        Field(ge=0, description="The zero-based index of the partial image."),
    ] = None
    quality: Annotated[
        Literal["low", "medium", "high", "auto"] | None,
        Field(description="The image-generation quality setting."),
    ] = None
    size: Annotated[
        Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None,
        Field(description="The dimensions requested for the generated image."),
    ] = None
    type: Annotated[
        Literal[
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.image_generation_call.partial_image",
            "response.image_generation_call.completed",
        ],
        Field(description="The image generation event type."),
    ]


def _ensure_code_interpreter_outputs_key(value: Any) -> None:
    """OpenAI operational parity: include `outputs: null` for code_interpreter_call items when outputs are omitted.

    Our default stream serialization uses `exclude_none=True`, which would drop `outputs` entirely when it is `None`.
    OpenAI's observed behavior includes `"outputs": null` unless the request includes `code_interpreter_call.outputs`.
    """

    if isinstance(value, dict):
        if value.get("type") == "code_interpreter_call" and "outputs" not in value:
            value["outputs"] = None
        for v in value.values():
            _ensure_code_interpreter_outputs_key(v)
        return
    if isinstance(value, list):
        for v in value:
            _ensure_code_interpreter_outputs_key(v)


class OpenAIResponsesStreamEventBase(OpenAIResponsesStreamEvent):
    content_index: Annotated[
        int,
        Field(description=("The index of the content part.")),
    ] = 0
    item_id: Annotated[
        str,
        Field(description=("The ID of the output item that the content part.")),
    ]
    output_index: Annotated[
        int,
        Field(description=("The index of the output item.")),
    ]


class OpenAIResponsesStreamPart(OpenAIResponsesStreamEventBase):
    part: Annotated[
        OpenAIOutputTextContent
        | OpenAIOutputRefusal
        | OpenAIReasoningContent
        | OpenAIReasoningSummary,
        Field(description=("The content part.")),
    ]
    summary_index: Annotated[
        int | None,
        Field(description=("The index of the summary part within the reasoning summary.")),
    ] = None
    type: Annotated[
        Literal[
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        ],
        Field(description=("The type of the event.")),
    ]


class OpenAIResponsesStreamText(OpenAIResponsesStreamEventBase):
    # Do not add a top-level stream `error` field here for `response.mcp_call.failed`.
    # MCP failure text is item-level (`mcp_call.error`) to match OpenAI event shape and
    # avoid OpenAI SDK stream parsing this as a fatal error event.
    delta: Annotated[
        str | None,
        Field(description=("The delta for text, refusal, function-call arguments, or reasoning.")),
    ] = None
    text: Annotated[
        str | None,
        Field(description=("The text content that is finalized.")),
    ] = None
    refusal: Annotated[
        str | None,
        Field(description=("The refusal text that is finalized.")),
    ] = None
    arguments: Annotated[
        str | None,
        Field(description=("The function-call arguments.")),
    ] = None
    code: Annotated[
        str | None,
        Field(description=("The final code snippet output by the code interpreter.")),
    ] = None
    output: Annotated[
        str | None,
        Field(description=("The finalized output text for hosted MCP calls.")),
    ] = None
    logprobs: Annotated[
        # TODO: Define logprob structure
        list[dict[str, Any]] | None,
        Field(description=("The log probabilities of the tokens in the delta.")),
    ] = None
    summary_index: Annotated[
        int | None,
        Field(description=("The index of the summary part within the reasoning summary.")),
    ] = None
    type: Annotated[
        Literal[
            "response.output_text.delta",
            "response.output_text.done",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.refusal.delta",
            "response.refusal.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call.completed",
            "response.code_interpreter_call_code.delta",
            "response.code_interpreter_call_code.done",
            "response.mcp_call.in_progress",
            "response.mcp_call_arguments.delta",
            "response.mcp_call_arguments.done",
            "response.mcp_call.completed",
            "response.mcp_call.failed",
        ],
        Field(description=("The type of the event.")),
    ]


class OpenAIResponsesStreamAnnotation(OpenAIResponsesStreamEventBase):
    annotation: Annotated[
        # TODO: Define annotation structure
        dict[str, Any],
        Field(description=("The annotation object.")),
    ]
    annotation_index: Annotated[
        int | None,
        Field(description=("The index of the annotation within the content part.")),
    ] = None
    type: Annotated[
        Literal["response.output_text.annotation.added"],
        Field(description=("The type of the event.")),
    ]


def parse_stream_event(payload: dict[str, Any]) -> OpenAIResponsesStreamEvent:
    """Parse an official SDK stream payload into Molto's public event model."""

    event_type = payload.get("type")
    if event_type == "error":
        model = OpenAIResponsesError
    elif event_type in {
        "response.created",
        "response.in_progress",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "response.queued",
    }:
        model = OpenAIResponsesStream
    elif event_type in {"response.output_item.added", "response.output_item.done"}:
        model = OpenAIResponsesStreamOutput
    elif event_type.startswith("response.image_generation_call."):
        model = OpenAIResponsesStreamImageGeneration
    elif event_type in {
        "response.content_part.added",
        "response.content_part.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
    }:
        model = OpenAIResponsesStreamPart
    elif event_type == "response.output_text.annotation.added":
        model = OpenAIResponsesStreamAnnotation
    else:
        model = OpenAIResponsesStreamText
    return model.model_validate(payload)
