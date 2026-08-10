from textwrap import dedent
from typing import Annotated, Any, Literal, TypeAlias, Union

from openai.types.responses.response_create_params import ToolChoice
from openai.types.responses.response_input_param import (
    EasyInputMessageParam,
    FunctionCallOutput,
    Message,
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)
from openai.types.responses.tool_param import ToolParam
from pydantic import Discriminator, Field

from molto.types.base import _NormalizedModel


class SystemPrompt(_NormalizedModel):
    """A system prompt, generally written by the application developer."""

    content: Annotated[str, Field(description="The content of the prompt.")]
    input_type: Annotated[
        Literal["system-prompt"],
        Field(description="Content type identifier as a discriminator."),
    ] = "system-prompt"


class TextInput(_NormalizedModel):
    """Text content with application-only metadata."""

    content: Annotated[str, Field(description="The content sent to the model.")]
    metadata: Annotated[
        Any | None,
        Field(description="Additional application-only metadata."),
    ] = None
    input_type: Annotated[
        Literal["text-input"],
        Field(description="Content type identifier as a discriminator."),
    ] = "text-input"


# class _FileUrlInput(_NormalizedModel):
#     """Common fields for URL-based user content."""

#     url: Annotated[str, Field(description="The URL of the file.")]
#     vendor_metadata: Annotated[
#         dict[str, Any] | None,
#         Field(description="Vendor-specific metadata for the file."),
#     ] = None
#     media_type: Annotated[
#         str | None,
#         Field(description="The media type of the file."),
#     ] = None
#     identifier: Annotated[
#         str | None,
#         Field(description="An optional identifier for the file."),
#     ] = None


# class ImageUrl(_FileUrl):
#     """An image referenced by URL."""

#     input_type: Annotated[
#         Literal["image-url"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "image-url"


# class AudioUrl(_FileUrl):
#     """An audio file referenced by URL."""

#     input_type: Annotated[
#         Literal["audio-url"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "audio-url"


# class DocumentUrl(_FileUrl):
#     """A document referenced by URL."""

#     input_type: Annotated[
#         Literal["document-url"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "document-url"


# class VideoUrl(_FileUrl):
#     """A video referenced by URL."""

#     input_type: Annotated[
#         Literal["video-url"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "video-url"


# class BinaryInput(_NormalizedModel):
#     """Binary user content."""

#     data: Annotated[bytes, Field(description="The binary file data.")]
#     media_type: Annotated[str, Field(description="The media type of the binary data.")]
#     vendor_metadata: Annotated[
#         dict[str, Any] | None,
#         Field(description="Vendor-specific metadata for the file."),
#     ] = None
#     identifier: Annotated[
#         str | None,
#         Field(description="An optional identifier for the file."),
#     ] = None
#     input_type: Annotated[
#         Literal["binary"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "binary"


# UploadedFileProviderName: TypeAlias = Literal[
#     "anthropic",
#     "openai",
#     "google",
#     "google-cloud",
#     "google-gla",
#     "google-vertex",
#     "bedrock",
#     "xai",
# ]


# class UploadedFileInput(_NormalizedModel):
#     """A reference to a provider-uploaded file."""

#     file_id: Annotated[str, Field(description="The provider-specific file identifier.")]
#     provider_name: Annotated[
#         UploadedFileProviderName,
#         Field(description="The provider that owns the file."),
#     ]
#     vendor_metadata: Annotated[
#         dict[str, Any] | None,
#         Field(description="Vendor-specific metadata for the file."),
#     ] = None
#     media_type: Annotated[
#         str | None,
#         Field(description="The media type of the file."),
#     ] = None
#     identifier: Annotated[
#         str | None,
#         Field(description="An optional identifier for the file."),
#     ] = None
#     input_type: Annotated[
#         Literal["uploaded-file"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "uploaded-file"


# class CachePointInput(_NormalizedModel):
#     """A prompt-caching boundary marker."""

#     input_type: Annotated[
#         Literal["cache-point"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "cache-point"
#     ttl: Annotated[
#         Literal["5m", "1h"],
#         Field(description="The cache time-to-live."),
#     ] = "5m"


class ToolReturn(_NormalizedModel):
    """Tool return parts."""

    id: Annotated[
        str | None,
        Field(description="The unique ID of the function tool call output."),
    ] = None
    tool_name: Annotated[
        str,
        Field(description="The name of the tool that was called."),
    ]
    content: Annotated[
        str | list[TextInput],
        Field(description="The tool return content, which may include multimodal files."),
    ]
    tool_call_id: Annotated[
        str,
        Field(description="The tool call identifier."),
    ]
    outcome: Annotated[
        Literal["success", "failed", "denied", "interrupted"],
        Field(
            description=(
                "The outcome of the tool call. "
                "Only `'failed'` is mapped to a provider's native error channel "
                "(e.g. Anthropic `is_error`, Bedrock `status='error'`)"
            )
        ),
    ]
    input_type: Annotated[
        Literal["tool-return"],
        Field(description="Content type identifier as a discriminator."),
    ] = "tool-return"


ResponseInputItemParam: TypeAlias = Union[
    EasyInputMessageParam,
    Message,
    ResponseOutputMessageParam,
    # ResponseFileSearchToolCallParam,
    # ResponseComputerToolCallParam,
    # ComputerCallOutput,
    # ResponseFunctionWebSearchParam,
    ResponseFunctionToolCallParam,
    FunctionCallOutput,
    # ToolSearchCall,
    # ResponseToolSearchOutputItemParamParam,
    # AdditionalTools,
    ResponseReasoningItemParam,
    # ResponseCompactionItemParamParam,
    # ImageGenerationCall,
    # ResponseCodeInterpreterToolCallParam,
    # LocalShellCall,
    # LocalShellCallOutput,
    # ShellCall,
    # ShellCallOutput,
    # ApplyPatchCall,
    # ApplyPatchCallOutput,
    # McpListTools,
    # McpApprovalRequest,
    # McpApprovalResponse,
    # McpCall,
    # ResponseCustomToolCallOutputParam,
    # ResponseCustomToolCallParam,
    # CompactionTrigger,
    # ItemReference,
    # Program,
    # ProgramOutput,
]
ResponseInputParam: TypeAlias = list[ResponseInputItemParam]

InputItem = Annotated[
    SystemPrompt | TextInput | ToolReturn,
    # | ToolSearchReturnPart
    # | LoadCapabilityReturnPart
    # | ToolAvailabilityDeltaPart
    Discriminator("input_type"),
]
Input: TypeAlias = list[InputItem]


ReasoningEffort: TypeAlias = (
    Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
)


class Reasoning(_NormalizedModel):
    """
    Configuration options for
    [reasoning models](https://platform.openai.com/docs/guides/reasoning).
    """

    effort: Annotated[
        ReasoningEffort | None,
        Field(description="Constrains effort on reasoning for reasoning models."),
    ] = None
    summary: Annotated[
        Literal["auto", "concise", "detailed"] | None,
        Field(description="A summary of the reasoning performed by the model."),
    ] = None


class StreamOptions(_NormalizedModel):
    """Options for streaming responses. Only set this when you set `stream: true`."""

    include_obfuscation: Annotated[
        bool | None,
        Field(
            description=dedent("""\
            When true, stream obfuscation will be enabled.

            Stream obfuscation adds random characters to an `obfuscation` field on streaming
            delta events to normalize payload sizes as a mitigation to certain side-channel
            attacks. These obfuscation fields are included by default, but add a small
            amount of overhead to the data stream.
            """)
        ),
    ] = None


class NormalizedRequest(_NormalizedModel):
    input: Annotated[
        str | InputItem | None,
        Field(description="Text, image, or file inputs to the model."),
    ] = None
    max_output_tokens: Annotated[
        int | None,
        Field(description="The upper bound for generated tokens."),
    ] = None
    max_tool_calls: Annotated[
        int | None,
        Field(description="The maximum number of built-in tool calls."),
    ] = None
    model: Annotated[
        str | None,
        Field(description="The model ID used to generate the response."),
    ] = None
    parallel_tool_calls: Annotated[
        bool | None,
        Field(description="Whether the model may run tool calls in parallel."),
    ] = None
    reasoning: Annotated[
        Reasoning | None,
        Field(description="Reasoning configuration for supported models."),
    ] = None
    stream: Annotated[
        bool | None,
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
    tool_choice: Annotated[
        ToolChoice | None,
        Field(description="How the model should select tools."),
    ] = None
    tools: Annotated[
        list[ToolParam] | None,
        Field(description="Tools available to the model."),
    ] = None
    top_p: Annotated[
        float | None,
        Field(description="The nucleus-sampling probability mass."),
    ] = None


class TextOutput(_NormalizedModel):
    """A plain text response from a model."""

    content: Annotated[
        str,
        Field(description="The text content of the response."),
    ]
    id: Annotated[
        str | None,
        Field(description="An optional identifier of the text part."),
    ] = None
    output_type: Annotated[
        Literal["text-output"],
        Field(description="Content type identifier as a discriminator."),
    ] = "text-output"


class ToolCall(_NormalizedModel):
    """A tool call from a model."""

    tool_name: Annotated[
        str,
        Field(description="The name of the tool to call."),
    ]
    args: Annotated[
        str | None,
        Field(description="The arguments to pass to the tool."),
    ] = None
    tool_call_id: Annotated[
        str,
        Field(description="The tool call identifier."),
    ]
    id: Annotated[
        str | None,
        Field(description="An optional identifier of the tool call part"),
    ] = None
    output_type: Annotated[
        Literal["tool-call"],
        Field(description="Content type identifier as a discriminator."),
    ] = "tool-call"


class Thinking(_NormalizedModel):
    """A thinking response from a model."""

    content: Annotated[
        str,
        Field(description="The thinking content of the response."),
    ]
    id: Annotated[
        str | None,
        Field(description="The identifier of the thinking part."),
    ] = None
    signature: Annotated[
        str | None,
        Field(description="The signature or encrypted content of the thinking."),
    ] = None
    output_type: Annotated[
        Literal["thinking"],
        Field(description="Content type identifier as a discriminator."),
    ] = "thinking"


OutputItem = Annotated[
    TextOutput | ToolCall | Thinking,
    # | ToolSearchCallPart
    # | LoadCapabilityCallPart
    # | NativeToolSearchCallPart
    # | NativeToolCallPart
    # | NativeToolSearchReturnPart
    # | NativeToolReturnPart
    # | CompactionPart
    # | FilePart,
    Discriminator("output_type"),
]
