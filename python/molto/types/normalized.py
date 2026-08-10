from textwrap import dedent
from typing import Annotated, Literal, Sequence, TypeAlias, Union

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
from pydantic_ai import (
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnContent,
    UserContent,
)

from molto.types.base import _NormalizedModel


class SystemPrompt(_NormalizedModel):
    """A system prompt, generally written by the application developer."""

    content: Annotated[str, Field(description="The content of the prompt.")]
    part_type: Annotated[
        Literal["system-prompt"],
        Field(description="Part type identifier as a discriminator."),
    ] = "system-prompt"


class UserPrompt(_NormalizedModel):
    """A user prompt, generally written by the end user."""

    content: Annotated[
        str | Sequence[UserContent],
        Field(description="The content of the prompt."),
    ]
    part_type: Annotated[
        Literal["user-prompt"],
        Field(description="Part type identifier as a discriminator."),
    ] = "user-prompt"


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
        ToolReturnContent,
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
    SystemPrompt | UserPrompt | ToolReturn,
    # | ToolSearchReturnPart
    # | LoadCapabilityReturnPart
    # | ToolAvailabilityDeltaPart
    Discriminator("part_type"),
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


class TextPart(_NormalizedModel):
    """A plain text response from a model."""

    content: str
    """The text content of the response."""

    id: str | None = None
    """An optional identifier of the text part.

    When this field is set, `provider_name` is required to identify the provider that generated this data.
    """

    provider_name: str | None = None
    """The name of the provider that generated the response.

    Required to be set when `provider_details` or `id` is set.
    """

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically.
    When this field is set, `provider_name` is required to identify the provider that generated this data.
    """

    part_kind: Literal["text"] = "text"
    """Part type identifier, this is available on all parts as a discriminator."""


class ToolCallPart(_NormalizedModel):
    """A tool call from a model."""

    tool_name: str
    """The name of the tool to call."""

    args: str | dict[str, Any] | None = None
    """The arguments to pass to the tool.

    This is stored either as a JSON string or a Python dictionary depending on how data was received.
    """

    tool_call_id: str = field(default_factory=_generate_tool_call_id)
    """The tool call identifier, this is used by some models including OpenAI.

    In case the tool call id is not provided by the model, Pydantic AI will generate a random one.
    """

    tool_kind: ToolPartKind | None = None
    """Discriminator for the typed subclass of this part (e.g. `'tool-search'`).

    See [`BaseToolReturnPart.tool_kind`][pydantic_ai.messages.BaseToolReturnPart.tool_kind] for
    the full semantics.
    """

    id: str | None = None
    """An optional identifier of the tool call part, separate from the tool call ID.

    This is used by some APIs like OpenAI Responses.
    When this field is set, `provider_name` is required to identify the provider that generated this data.
    """

    provider_name: str | None = None
    """The name of the provider that generated the response.

    Native tool calls are only sent back to the same provider.
    Required to be set when `provider_details` or `id` is set.
    """

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically.
    When this field is set, `provider_name` is required to identify the provider that generated this data.
    """


class ThinkingPart(_NormalizedModel):
    """A thinking response from a model."""

    content: str
    """The thinking content of the response."""

    id: str | None = None
    """The identifier of the thinking part.

    When this field is set, `provider_name` is required to identify the provider that generated this data.
    """

    signature: str | None = None
    """The signature of the thinking.

    Supported by:

    * Anthropic (corresponds to the `signature` field)
    * Bedrock (corresponds to the `signature` field)
    * Google (corresponds to the `thought_signature` field)
    * OpenAI (corresponds to the `encrypted_content` field)

    When this field is set, `provider_name` is required to identify the provider that generated this data.
    """

    provider_name: str | None = None
    """The name of the provider that generated the response.

    Signatures are only sent back to the same provider.
    Required to be set when `provider_details`, `id` or `signature` is set.
    """

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically.
    When this field is set, `provider_name` is required to identify the provider that generated this data.
    """

    part_kind: Literal["thinking"] = "thinking"
    """Part type identifier, this is available on all parts as a discriminator."""


OutputItem = Annotated[
    TextPart | ToolCallPart | ThinkingPart,
    # | ToolSearchCallPart
    # | LoadCapabilityCallPart
    # | NativeToolSearchCallPart
    # | NativeToolCallPart
    # | NativeToolSearchReturnPart
    # | NativeToolReturnPart
    # | CompactionPart
    # | FilePart,
    Discriminator("part_type"),
]
