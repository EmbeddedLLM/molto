"""OpenAI Chat Completions request API models."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from molto.types.ir.portable import (
    InputIR,
)
from molto.types.openai._common import OpenAIReasoningEffortLevel

# ---------------------------------------------------------------------------
# Request message types
# ---------------------------------------------------------------------------


class ChatCompletionContentPartText(BaseModel):
    type: Annotated[
        Literal["text"],
        Field(description="The type of the content part. Always `text`."),
    ] = "text"
    text: Annotated[
        str,
        Field(description="The text content."),
    ]


class ChatCompletionContentPartImageImageUrl(BaseModel):
    url: Annotated[
        str,
        Field(
            description="Either a URL of the image or the base64 encoded image data.",
        ),
    ]
    detail: Annotated[
        Literal["auto", "low", "high"],
        Field(description="Specifies the detail level of the image."),
    ] = "auto"


class ChatCompletionContentPartImage(BaseModel):
    type: Annotated[
        Literal["image_url"],
        Field(description="The type of the content part. Always `image_url`."),
    ] = "image_url"
    image_url: Annotated[
        ChatCompletionContentPartImageImageUrl,
        Field(description="The image URL and detail level."),
    ]


class ChatCompletionContentPartInputAudioData(BaseModel):
    data: Annotated[
        str,
        Field(description="Base64-encoded audio data."),
    ]
    format: Annotated[
        Literal["wav", "mp3"],
        Field(description="The format of the encoded input audio."),
    ]


class ChatCompletionContentPartInputAudio(BaseModel):
    type: Annotated[
        Literal["input_audio"],
        Field(description="The type of the content part. Always `input_audio`."),
    ] = "input_audio"
    input_audio: Annotated[
        ChatCompletionContentPartInputAudioData,
        Field(description="The input audio data and format."),
    ]


ChatCompletionContentPart = Union[
    ChatCompletionContentPartText,
    ChatCompletionContentPartImage,
    ChatCompletionContentPartInputAudio,
]


class ChatCompletionFunctionCall(BaseModel):
    name: Annotated[
        str,
        Field(description="The name of the function to call."),
    ]
    arguments: Annotated[
        str,
        Field(description="The arguments to call the function with, as a JSON string."),
    ]


class ChatCompletionToolCall(BaseModel):
    id: Annotated[
        str,
        Field(description="The ID of the tool call."),
    ]
    type: Annotated[
        Literal["function"],
        Field(description="The type of the tool. Always `function`."),
    ] = "function"
    function: Annotated[
        ChatCompletionFunctionCall,
        Field(description="The function that the model called."),
    ]


class _ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Annotated[
        str,
        Field(
            description="The role of the messages author.",
        ),
    ]


class ChatCompletionSystemMessage(_ChatCompletionMessage):
    role: Annotated[
        Literal["system", "developer"],
        Field(description="The role of the messages author."),
    ]
    content: Annotated[
        str | None,
        Field(description="The contents of the system message."),
    ]
    name: Annotated[
        str | None,
        Field(description="An optional name for the participant."),
    ] = None


class ChatCompletionUserMessage(_ChatCompletionMessage):
    role: Annotated[
        Literal["user"],
        Field(description="The role of the messages author. Always `user`."),
    ] = "user"
    content: Annotated[
        str | list[ChatCompletionContentPart] | None,
        Field(description="The contents of the user message."),
    ]
    name: Annotated[
        str | None,
        Field(description="An optional name for the participant."),
    ] = None


class ChatCompletionAudioReference(BaseModel):
    id: Annotated[
        str,
        Field(description="The unique ID of a previous audio response."),
    ]


class ChatCompletionAssistantMessage(_ChatCompletionMessage):
    role: Annotated[
        Literal["assistant"],
        Field(description="The role of the messages author. Always `assistant`."),
    ] = "assistant"
    content: Annotated[
        str | None,
        Field(description="The contents of the assistant message."),
    ] = None
    refusal: Annotated[
        str | None,
        Field(description="The refusal message by the assistant."),
    ] = None
    audio: Annotated[
        ChatCompletionAudioReference | None,
        Field(description="A reference to a previous audio response from the assistant."),
    ] = None
    tool_calls: Annotated[
        list[ChatCompletionToolCall] | None,
        Field(description="The tool calls generated by the model."),
    ] = None
    name: Annotated[
        str | None,
        Field(description="An optional name for the participant."),
    ] = None


class ChatCompletionToolMessage(_ChatCompletionMessage):
    role: Annotated[
        Literal["tool"],
        Field(description="The role of the messages author. Always `tool`."),
    ] = "tool"
    content: Annotated[
        str | None,
        Field(description="The contents of the tool message."),
    ]
    tool_call_id: Annotated[
        str,
        Field(description="Tool call that this message is responding to."),
    ]
    name: Annotated[
        str | None,
        Field(description="An optional tool name."),
    ] = None


ChatCompletionMessage = Annotated[
    Union[
        ChatCompletionSystemMessage,
        ChatCompletionUserMessage,
        ChatCompletionAssistantMessage,
        ChatCompletionToolMessage,
    ],
    Field(discriminator="role"),
]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class ChatCompletionFunction(BaseModel):
    name: Annotated[
        str,
        Field(description="The name of the function."),
    ]
    description: Annotated[
        str | None,
        Field(description="A description of what the function does."),
    ] = None
    parameters: Annotated[
        dict[str, Any] | None,
        Field(description="The parameters the function accepts as a JSON Schema object."),
    ] = None
    strict: Annotated[
        bool | None,
        Field(description="Whether to enable strict schema adherence."),
    ] = None


class ChatCompletionTool(BaseModel):
    type: Annotated[
        Literal["function"],
        Field(description="The type of the tool. Always `function`."),
    ] = "function"
    function: Annotated[
        ChatCompletionFunction,
        Field(description="The function definition."),
    ]


class ChatCompletionToolChoiceFunction(BaseModel):
    name: Annotated[
        str,
        Field(description="The name of the function to call."),
    ]


class ChatCompletionNamedToolChoice(BaseModel):
    type: Annotated[
        Literal["function"],
        Field(description="The type of tool to call. Always `function`."),
    ] = "function"
    function: Annotated[
        ChatCompletionToolChoiceFunction,
        Field(description="The function to call."),
    ]


class ChatCompletionAllowedTool(BaseModel):
    type: Annotated[
        Literal["function"],
        Field(description="The allowed tool type. Always `function`."),
    ] = "function"
    function: Annotated[
        ChatCompletionToolChoiceFunction,
        Field(description="The allowed function."),
    ]


class ChatCompletionAllowedToolsChoice(BaseModel):
    type: Annotated[
        Literal["allowed_tools"],
        Field(description="Constrains the model to a subset of tools."),
    ] = "allowed_tools"
    mode: Annotated[
        Literal["auto", "required"],
        Field(description="Whether the model may or must call one of the allowed tools."),
    ] = "auto"
    tools: Annotated[
        list[ChatCompletionAllowedTool],
        Field(description="The allowed tools."),
    ]


ChatCompletionToolChoiceOption = Union[
    Literal["none", "auto", "required"],
    ChatCompletionNamedToolChoice,
    ChatCompletionAllowedToolsChoice,
]


class ChatCompletionResponseFormatText(BaseModel):
    type: Annotated[
        Literal["text"],
        Field(description="The type of response format. Always `text`."),
    ] = "text"


class ChatCompletionResponseFormatJsonObject(BaseModel):
    type: Annotated[
        Literal["json_object"],
        Field(description="The type of response format. Always `json_object`."),
    ] = "json_object"


class ChatCompletionResponseFormatJsonSchema(BaseModel):
    type: Annotated[
        Literal["json_schema"],
        Field(description="The type of response format. Always `json_schema`."),
    ] = "json_schema"
    json_schema: Annotated[
        dict[str, Any],
        Field(description="The JSON schema for the response format."),
    ]
    name: Annotated[
        str | None,
        Field(description="The name of the schema."),
    ] = None
    strict: Annotated[
        bool | None,
        Field(description="Whether to enable strict schema adherence."),
    ] = None
    description: Annotated[
        str | None,
        Field(description="A description of the response format."),
    ] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_sdk_json_schema_shape(cls, data: Any) -> Any:
        """Accept the SDK's nested Chat response-format shape.

        The original public model accepted ``json_schema`` as the schema itself.
        OpenAI's Chat Completions wire contract instead nests that schema alongside
        its metadata under ``response_format.json_schema``. Normalize the native
        shape while preserving the legacy form used by the existing IR test.
        """
        if not isinstance(data, dict):
            return data

        json_schema = data.get("json_schema")
        if not isinstance(json_schema, dict) or "schema" not in json_schema:
            return data

        return {
            **data,
            "json_schema": json_schema["schema"],
            "name": data.get("name", json_schema.get("name")),
            "strict": data.get("strict", json_schema.get("strict")),
            "description": data.get("description", json_schema.get("description")),
        }

    @model_serializer
    def _serialize_sdk_json_schema_shape(self) -> dict[str, Any]:
        json_schema = {
            "name": self.name,
            "schema": self.json_schema,
            "strict": self.strict,
            "description": self.description,
        }
        return {
            "type": self.type,
            "json_schema": {key: value for key, value in json_schema.items() if value is not None},
        }


ChatCompletionResponseFormat = Annotated[
    Union[
        ChatCompletionResponseFormatText,
        ChatCompletionResponseFormatJsonObject,
        ChatCompletionResponseFormatJsonSchema,
    ],
    Field(discriminator="type"),
]


class ChatCompletionStreamOptions(BaseModel):
    include_usage: Annotated[
        bool | None,
        Field(description="Whether to include usage data."),
    ] = None
    include_obfuscation: Annotated[
        bool | None,
        Field(description="Whether to include stream obfuscation fields."),
    ] = None


class ChatCompletionCustomVoice(BaseModel):
    id: Annotated[
        str,
        Field(description="The custom voice ID."),
    ]


class ChatCompletionAudioConfig(BaseModel):
    format: Annotated[
        Literal["wav", "aac", "mp3", "flac", "opus", "pcm16"],
        Field(description="The generated audio format."),
    ]
    voice: Annotated[
        str | ChatCompletionCustomVoice,
        Field(description="The built-in voice name or custom voice reference."),
    ]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: Annotated[
        list[ChatCompletionMessage],
        Field(description="A list of messages comprising the conversation so far."),
    ]
    model: Annotated[
        str,
        Field(description="The model to use for generating the chat completion."),
    ]
    upstream: Annotated[
        Literal["responses", "chat_completions"] | None,
        Field(
            exclude=True,
            description="Molto execution endpoint. This routing field is consumed before IR conversion.",
        ),
    ] = None
    audio: Annotated[
        ChatCompletionAudioConfig | None,
        Field(description="Parameters for generated audio output."),
    ] = None
    frequency_penalty: Annotated[
        float | None,
        Field(
            ge=-2,
            le=2,
            description=(
                "Number between -2.0 and 2.0. Positive values penalize new tokens based on their "
                "existing frequency in the text so far."
            ),
        ),
    ] = None
    logprobs: Annotated[
        bool | None,
        Field(
            description="Whether to return log probabilities of the output tokens or not.",
        ),
    ] = None
    top_logprobs: Annotated[
        int | None,
        Field(
            ge=0,
            le=20,
            description=(
                "An integer between 0 and 20 specifying the number of most likely tokens to return "
                "at each token position."
            ),
        ),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "The maximum number of tokens that can be generated in the chat completion."
            ),
        ),
    ] = None
    max_completion_tokens: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "An upper bound for the number of tokens that can be generated for a completion, "
                "including visible output tokens and reasoning tokens."
            ),
        ),
    ] = None
    n: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of completion choices to generate.",
        ),
    ] = None
    modalities: Annotated[
        list[Literal["text", "audio"]] | None,
        Field(description="The output modalities requested from the model."),
    ] = None
    presence_penalty: Annotated[
        float | None,
        Field(
            ge=-2,
            le=2,
            description=(
                "Number between -2.0 and 2.0. Positive values penalize new tokens based on "
                "whether they appear in the text so far."
            ),
        ),
    ] = None
    response_format: Annotated[
        ChatCompletionResponseFormat | None,
        Field(description="An object specifying the format that the model must output."),
    ] = None
    seed: Annotated[
        int | None,
        Field(
            description="If specified, our system will make a best effort to sample deterministically."
        ),
    ] = None
    stop: Annotated[
        str | list[str] | None,
        Field(description="Up to 4 sequences where the API will stop generating further tokens."),
    ] = None
    stream: Annotated[
        bool | None,
        Field(
            description="If set to true, the model response data will be streamed as it is generated."
        ),
    ] = None
    stream_options: Annotated[
        ChatCompletionStreamOptions | None,
        Field(description="Options for streaming response."),
    ] = None
    store: Annotated[
        bool | None,
        Field(description="Whether to store the generated completion."),
    ] = None
    prompt_cache_key: Annotated[
        str | None,
        Field(description="Cache key for prompt prefix caching."),
    ] = None
    prompt_cache_retention: Annotated[
        Literal["in_memory", "24h"] | None,
        Field(description="Retention policy for prompt cache entries."),
    ] = None
    safety_identifier: Annotated[
        str | None,
        Field(description="A stable identifier for your end users."),
    ] = None
    service_tier: Annotated[
        str | None,
        Field(description="The service tier used to process the request."),
    ] = None
    temperature: Annotated[
        float | None,
        Field(
            ge=0,
            le=2,
            description="What sampling temperature to use, between 0 and 2.",
        ),
    ] = None
    top_p: Annotated[
        float | None,
        Field(
            ge=0,
            le=1,
            description="An alternative to sampling with temperature, called nucleus sampling.",
        ),
    ] = None
    tools: Annotated[
        list[ChatCompletionTool] | None,
        Field(description="A list of tools the model may call."),
    ] = None
    tool_choice: Annotated[
        ChatCompletionToolChoiceOption | None,
        Field(description="Controls which tool is called by the model."),
    ] = None
    parallel_tool_calls: Annotated[
        bool | None,
        Field(description="Whether to enable parallel function calling."),
    ] = None
    user: Annotated[
        str | None,
        Field(description="A unique identifier representing your end-user."),
    ] = None
    reasoning_effort: Annotated[
        OpenAIReasoningEffortLevel | None,
        Field(description="Constrains effort on reasoning for reasoning models."),
    ] = None
    extra_body: Annotated[
        dict[str, Any] | None,
        Field(description="Additional provider-specific request body fields."),
    ] = None

    def to_ir(self) -> InputIR:
        """Convert the public Chat request into the mandatory portable request IR."""

        from .to_ir import request_to_ir

        return request_to_ir(self)
