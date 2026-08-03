"""Shared OpenAI Responses item, content, and tool API models.

Both the request and response modules build on these shapes.
"""

from typing import Annotated, Any, Literal, Union

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    WrapValidator,
)
from pydantic_extra_types.timezone_name import TimeZoneName

from molto.types.openai._common import OpenAIReasoningEffortLevel


def _normalize_hosted_tool_type(tool_type: str) -> str:
    if tool_type in {"web_search_preview", "web_search_2025_08_26"}:
        return "web_search"
    return tool_type


def _json_schema_validator(value: Any, handler):
    if isinstance(value, dict):
        try:
            Draft202012Validator.check_schema(value)
        except jsonschema_exceptions.SchemaError as exc:
            raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc
        return value
    return handler(value)


JsonSchema = Annotated[dict[str, Any], WrapValidator(_json_schema_validator)]

# OpenAI documentation-parity text shared by the standard item-status fields.
_ITEM_STATUS_DESCRIPTION = (
    "The status of item. "
    "One of `in_progress`, `completed`, or `incomplete`. "
    "Populated when items are returned via API."
)


class OpenAIConversation(BaseModel):
    """The conversation that this response belongs to."""

    id: Annotated[
        str,
        Field(description="The unique ID of the conversation."),
    ]


class OpenAIReasoningEffort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: Annotated[
        OpenAIReasoningEffortLevel | None,
        Field(
            description=(
                "Constrains effort on reasoning for reasoning models. "
                "Currently supported values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`. "
                "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response."
            ),
        ),
    ] = None
    summary: Annotated[
        Literal["auto", "concise", "detailed"] | None,
        Field(
            description=(
                "A summary of the reasoning performed by the model. "
                "This can be useful for debugging and understanding the model's reasoning process. "
                "One of `auto`, `concise`, or `detailed`."
            ),
        ),
    ] = None


class OpenAIResponseReasoning(OpenAIReasoningEffort):
    """Reasoning response config, including provider-added populated fields."""

    model_config = ConfigDict(extra="allow")


# ---- Responses `include[]` schema ----

# Keep this as a finite set (enum-like) so we fail fast on typos while staying spec-aligned.
# Note: The gateway currently acts upon `web_search_call.action.sources` and
# `code_interpreter_call.outputs` (others are parsed but ignored).
OpenAIResponsesInclude = Literal[
    "web_search_call.action.sources",
    "code_interpreter_call.outputs",
    "computer_call_output.output.image_url",
    "file_search_call.results",
    "message.input_image.image_url",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
]


class _OpenAIMessage(BaseModel):
    role: Annotated[
        Literal["user", "assistant", "system", "developer"],
        Field(
            description=(
                "The role of the message input. "
                "One of `user`, `assistant`, `system`, or `developer`."
            )
        ),
    ]
    type: Annotated[
        Literal["message"], Field(description="The type of the message input. Always `message`.")
    ] = "message"


class OpenAITextContent(BaseModel):
    """A text input to the model."""

    text: Annotated[
        str,
        Field(description=("The text input to the model.")),
    ]
    type: Annotated[
        Literal["input_text"],
        Field(description=("The type of the input item. Always `input_text`.")),
    ] = "input_text"


class OpenAIImageContent(BaseModel):
    """An image input to the model."""

    detail: Annotated[
        Literal["high", "low", "auto"],
        Field(
            description=(
                "The detail level of the image to be sent to the model. "
                "One of `high`, `low`, or `auto`. Defaults to `auto`."
            ),
        ),
    ] = "auto"
    type: Annotated[
        Literal["input_image"],
        Field(description=("The type of the input item. Always `input_image`.")),
    ] = "input_image"
    image_url: Annotated[
        str,
        Field(
            description=(
                "The URL of the image to be sent to the model. "
                "A fully qualified URL or base64 encoded image in a data URL."
            ),
        ),
    ]


class OpenAIInputMessage(_OpenAIMessage):
    role: Annotated[
        Literal["user", "system", "developer"],
        Field(description="The non-assistant role of the input message."),
    ]
    content: Annotated[
        # str | list[OpenAITextContent | OpenAIImageContent | OpenAIFileContent]
        str | list[OpenAITextContent | OpenAIImageContent],
        Field(
            description=(
                "Text, image, or audio input to the model, used to generate a response. "
                "Can also contain previous assistant responses."
            ),
        ),
    ]
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(
            description=_ITEM_STATUS_DESCRIPTION,
            exclude=True,
        ),
    ] = None


class OpenAIOutputTextContent(BaseModel):
    """A text output from the model."""

    annotations: Annotated[
        # TODO: Define annotation structure
        list[dict[str, Any]],
        Field(description=("The annotations of the text output.")),
    ] = Field(default_factory=list)
    logprobs: Annotated[
        # TODO: Define logprob structure
        list[dict[str, Any]],
        Field(description=("Log probability information.")),
    ] = Field(default_factory=list)
    text: Annotated[
        str,
        Field(description=("The text output from the model.")),
    ]
    type: Annotated[
        Literal["output_text"],
        Field(description=("The type of the output text. Always `output_text`.")),
    ] = "output_text"


class OpenAIOutputRefusal(BaseModel):
    """A refusal from the model."""

    refusal: Annotated[
        str,
        Field(description=("The refusal explanation from the model.")),
    ]
    type: Annotated[
        Literal["refusal"],
        Field(description=("The type of the refusal. Always `refusal`.")),
    ] = "refusal"


class OpenAIOutputItem(_OpenAIMessage):
    content: Annotated[
        list[OpenAIOutputTextContent | OpenAIOutputRefusal],
        Field(description=("The content of the output message.")),
    ]
    id: Annotated[
        str,
        Field(description=("The unique ID of the output message.")),
    ]
    role: Annotated[
        Literal["assistant"],
        Field(description=("The role of the output message. Always `assistant`.")),
    ] = "assistant"
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"],
        Field(
            description=_ITEM_STATUS_DESCRIPTION,
        ),
    ]
    phase: Annotated[
        Literal["commentary", "final_answer"] | None,
        Field(
            description=(
                "Whether this assistant message is intermediate commentary or the final answer."
            ),
        ),
    ] = None


class OpenAIReasoningContent(BaseModel):
    """Reasoning text content."""

    text: Annotated[
        str,
        Field(description=("The reasoning text from the model.")),
    ]
    type: Annotated[
        Literal["reasoning_text"],
        Field(description=("The type of the reasoning text. Always `reasoning_text`.")),
    ] = "reasoning_text"


class OpenAIReasoningSummary(BaseModel):
    """Reasoning summary content."""

    text: Annotated[
        str,
        Field(description=("A summary of the reasoning output from the model so far.")),
    ]
    type: Annotated[
        Literal["summary_text"],
        Field(description=("The type of the summary text. Always `summary_text`.")),
    ] = "summary_text"


class OpenAIReasoningItem(BaseModel):
    """
    A description of the chain of thought used by a reasoning model while generating a response.
    Be sure to include these items in your input to the Responses API
    for subsequent turns of a conversation if you are manually managing context.
    """

    id: Annotated[
        str,
        Field(description=("The unique identifier of the reasoning content.")),
    ]
    summary: Annotated[
        list[OpenAIReasoningSummary],
        Field(description=("Reasoning summary content.")),
    ] = Field(default_factory=list)
    type: Annotated[
        Literal["reasoning"],
        Field(description=("The type of the object. Always `reasoning`.")),
    ] = "reasoning"
    content: Annotated[
        list[OpenAIReasoningContent],
        Field(description=("Reasoning text content.")),
    ] = Field(default_factory=list)
    encrypted_content: Annotated[
        str | None,
        Field(
            description=(
                "The encrypted content of the reasoning item - "
                "populated when a response is generated with `reasoning.encrypted_content` in the `include` parameter."
            )
        ),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(description=_ITEM_STATUS_DESCRIPTION),
    ] = None


class OpenAIFunctionToolCall(BaseModel):
    """A tool call to run a function."""

    arguments: Annotated[
        str,
        Field(description=("A JSON string of the arguments to pass to the function.")),
    ]
    call_id: Annotated[
        str,
        Field(description=("The unique ID of the function tool call generated by the model.")),
    ]
    name: Annotated[
        str,
        Field(description=("The name of the function to run.")),
    ]
    type: Annotated[
        Literal["function_call"],
        Field(description=("The type of the function tool call. Always `function_call`.")),
    ] = "function_call"
    id: Annotated[
        str | None,
        Field(description=("The unique ID of the function tool call.")),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(
            description=_ITEM_STATUS_DESCRIPTION,
        ),
    ] = None


class OpenAIMcpToolCall(BaseModel):
    """An MCP tool call executed by the gateway."""

    id: Annotated[
        str,
        Field(description=("The unique ID of the MCP tool call.")),
    ]
    server_label: Annotated[
        str,
        Field(description=("The MCP server label for this call.")),
    ]
    name: Annotated[
        str,
        Field(description=("The MCP tool name.")),
    ]
    arguments: Annotated[
        str,
        Field(description=("A JSON string of MCP tool arguments.")),
    ]
    status: Annotated[
        Literal["in_progress", "completed", "incomplete", "failed"],
        Field(
            description=(
                "The status of the MCP call item. "
                "One of `in_progress`, `completed`, `incomplete`, or `failed`."
            )
        ),
    ]
    output: Annotated[
        str | None,
        Field(description=("The tool output text for successful MCP calls.")),
    ] = None
    error: Annotated[
        str | None,
        Field(description=("The error text for failed MCP calls.")),
    ] = None
    approval_request_id: Annotated[
        str | None,
        Field(description=("The ID of the MCP tool call approval request.")),
    ] = None
    type: Annotated[
        Literal["mcp_call"],
        Field(description=("The type of the MCP tool call. Always `mcp_call`.")),
    ] = "mcp_call"


class OpenAIFunctionToolOutput(BaseModel):
    """The output of a function tool call."""

    call_id: Annotated[
        str,
        Field(description=("The unique ID of the function tool call generated by the model.")),
    ]
    output: Annotated[
        # str | list[OpenAITextContent | OpenAIImageContent | OpenAIFileContent]
        str | list[OpenAITextContent | OpenAIImageContent],
        Field(description=("Text, image, or file output of the function tool call.")),
    ]
    type: Annotated[
        Literal["function_call_output"],
        Field(
            description=(
                "The type of the function tool call output. Always `function_call_output`."
            )
        ),
    ] = "function_call_output"
    id: Annotated[
        str | None,
        Field(
            description=(
                "The unique ID of the function tool call output. "
                "Populated when this item is returned via API."
            )
        ),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(
            description=_ITEM_STATUS_DESCRIPTION,
        ),
    ] = None


class OpenAICodeOutputLog(BaseModel):
    """The logs output from the code interpreter."""

    logs: Annotated[
        str,
        Field(description=("The logs output from the code interpreter.")),
    ]
    type: Annotated[
        Literal["logs"],
        Field(description=("The type of the output. Always `logs`")),
    ] = "logs"


class OpenAICodeOutputImage(BaseModel):
    """The image output from the code interpreter."""

    type: Annotated[
        Literal["image"],
        Field(description=("The type of the output. Always `image`")),
    ] = "image"
    url: Annotated[
        str,
        Field(description=("The URL of the image output from the code interpreter.")),
    ]


class OpenAICodeToolCall(BaseModel):
    """Code interpreter tool call."""

    code: Annotated[
        str | None,
        Field(description=("The code to run, or null if not available.")),
    ]
    container_id: Annotated[
        str,
        Field(description=("The ID of the container used to run the code.")),
    ]
    id: Annotated[
        str,
        Field(description=("The unique ID of the code interpreter tool call.")),
    ]
    outputs: Annotated[
        list[OpenAICodeOutputLog | OpenAICodeOutputImage] | None,
        Field(
            description=(
                "The outputs generated by the code interpreter, such as logs or images. "
                "Can be null if no outputs are available."
            )
        ),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete", "interpreting", "failed"],
        Field(
            description=(
                "The status of the code interpreter tool call. "
                "Valid values are `in_progress`, `completed`, `incomplete`, `interpreting`, and `failed`."
            )
        ),
    ]
    type: Annotated[
        Literal["code_interpreter_call"],
        Field(
            description=(
                "The type of the code interpreter tool call. Always `code_interpreter_call`."
            )
        ),
    ] = "code_interpreter_call"


class OpenAIImageGenerationCall(BaseModel):
    """An image generation request made by the model."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[
        str,
        Field(description="The unique ID of the image generation call."),
    ]
    action: Annotated[
        Literal["generate", "edit", "auto"] | None,
        Field(description="The image operation performed by the tool."),
    ] = None
    background: Annotated[
        Literal["transparent", "opaque", "auto"] | None,
        Field(description="The generated image background mode."),
    ] = None
    output_format: Annotated[
        Literal["png", "webp", "jpeg"] | None,
        Field(description="The generated image encoding."),
    ] = None
    quality: Annotated[
        Literal["low", "medium", "high", "auto"] | None,
        Field(description="The generated image quality setting."),
    ] = None
    result: Annotated[
        str | None,
        Field(description="The base64-encoded generated image, when available."),
    ] = None
    revised_prompt: Annotated[
        str | None,
        Field(description="The provider-revised image prompt."),
    ] = None
    size: Annotated[
        Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None,
        Field(description="The generated image dimensions."),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "generating", "failed"],
        Field(description="The status of the image generation call."),
    ]
    type: Annotated[
        Literal["image_generation_call"],
        Field(description="The image generation output-item type."),
    ] = "image_generation_call"


class OpenAIImageGenerationCallInput(OpenAIImageGenerationCall):
    """An image-generation call reused as Responses conversation history."""

    status: Annotated[
        Literal["in_progress", "completed", "generating", "failed"] | None,
        Field(description="The status of the image generation call, when supplied."),
    ] = None


class OpenAIWebSearchSource(BaseModel):
    type: Annotated[
        str,
        Field(description="The type of source (e.g. `url`, `api`)."),
    ] = "url"
    url: Annotated[
        str | None,
        Field(description="The canonical URL used by the web search tool."),
    ] = None
    name: Annotated[
        str | None,
        Field(description="The name of the source (for API-type sources)."),
    ] = None


class OpenAIWebSearchActionSearch(BaseModel):
    type: Annotated[
        Literal["search"],
        Field(description="The web search action type."),
    ] = "search"
    query: Annotated[
        str | None,
        Field(description="The primary search query executed by the tool."),
    ] = None
    queries: Annotated[
        list[str] | None,
        Field(description="Optional related or rewritten queries used by the tool."),
    ] = None
    sources: Annotated[
        list[OpenAIWebSearchSource] | None,
        Field(description="Optional normalized source URLs used by the search action."),
    ] = None


class OpenAIWebSearchActionOpenPage(BaseModel):
    type: Annotated[
        Literal["open_page"],
        Field(description="The web search action type."),
    ] = "open_page"
    url: Annotated[
        str | None,
        Field(description="The page URL opened by the tool."),
    ] = None


class OpenAIWebSearchActionFindInPage(BaseModel):
    type: Annotated[
        Literal["find_in_page"],
        Field(description="The web search action type."),
    ] = "find_in_page"
    pattern: Annotated[
        str,
        Field(description="The text pattern searched within the page."),
    ]
    url: Annotated[
        str | None,
        Field(description="The page URL searched by the tool."),
    ] = None


class OpenAIWebSearchToolCall(BaseModel):
    action: (
        Annotated[
            OpenAIWebSearchActionSearch
            | OpenAIWebSearchActionOpenPage
            | OpenAIWebSearchActionFindInPage,
            Field(discriminator="type", description="The completed web search action."),
        ]
        | None
    ) = None
    id: Annotated[
        str,
        Field(description="The unique ID of the web search tool call."),
    ]
    status: Annotated[
        Literal["in_progress", "searching", "completed", "failed"],
        Field(description="The status of the web search tool call."),
    ]
    type: Annotated[
        Literal["web_search_call"],
        Field(description="The type of the web search tool call."),
    ] = "web_search_call"


OpenAIInput = Union[
    OpenAIInputMessage,
    OpenAIOutputItem,
    Annotated[
        Union[
            OpenAIReasoningItem,
            OpenAIFunctionToolCall,
            OpenAIFunctionToolOutput,
            OpenAIMcpToolCall,
            OpenAICodeToolCall,
            OpenAIImageGenerationCallInput,
            OpenAIWebSearchToolCall,
        ],
        Discriminator("type"),
    ],
]
OpenAIOutput = Annotated[
    Union[
        OpenAIOutputItem,
        OpenAIReasoningItem,
        OpenAIFunctionToolCall,
        OpenAIMcpToolCall,
        OpenAICodeToolCall,
        OpenAIImageGenerationCall,
        OpenAIWebSearchToolCall,
    ],
    Discriminator("type"),
]


class OpenAIPromptTemplate(BaseModel):
    id: Annotated[str, Field(description=("The unique identifier of the prompt template to use."))]
    variables: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional map of values to substitute in for variables in your prompt. "
                "The substitution values can either be strings, or other Response input types like images or files."
            ),
        ),
    ] = None
    version: Annotated[
        str | None,
        Field(
            description=("Optional version of the prompt template."),
        ),
    ] = None


class OpenAITextFormat(BaseModel):
    type: Annotated[
        Literal["text"],
        Field(
            description=("The type of response format being defined. Always `text`."),
        ),
    ] = "text"


class OpenAIJsonSchemaFormat(BaseModel):
    name: Annotated[
        str,
        Field(
            pattern=r"^[a-zA-Z0-9_-]{1,64}$",
            description=(
                "The name of the response format. "
                "Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64."
            ),
        ),
    ]
    schema_: Annotated[
        dict[str, Any],
        Field(
            validation_alias=AliasChoices("schema", "schema_"),
            serialization_alias="schema",
            description=("The schema for the response format, described as a JSON Schema object."),
        ),
    ]
    type: Annotated[
        Literal["json_schema"],
        Field(
            description=("The type of response format being defined. Always `json_schema`."),
        ),
    ] = "json_schema"
    description: Annotated[
        str | None,
        Field(
            description=(
                "A description of what the response format is for, used by the model to determine how to respond in the format."
            ),
        ),
    ] = None
    strict: Annotated[
        bool,
        Field(
            description=(
                "Whether to enable strict schema adherence when generating the output. "
                "If set to `true`, the model will always follow the exact schema defined in the schema field. "
                "Only a subset of JSON Schema is supported when `strict` is `true`."
            ),
        ),
    ] = False


class OpenAIJsonObjectFormat(BaseModel):
    type: Annotated[
        Literal["json_object"],
        Field(
            description=("The type of response format being defined. Always `json_object`."),
        ),
    ] = "json_object"


class OpenAIResponsesPlainTextConfig(BaseModel):
    format: Annotated[
        OpenAITextFormat | None,
        Field(
            description=(
                "An object specifying the format that the model must output. "
                'Configuring `{ "type": "json_schema" }` enables Structured Outputs, '
                "which ensures the model will match your supplied JSON schema. "
                'The default format is `{ "type": "text" }` with no additional options. '
                'Setting to `{ "type": "json_object" }` enables the older JSON mode, '
                "which ensures the message the model generates is valid JSON. "
                "Using `json_schema` is preferred for models that support it."
            ),
        ),
    ] = None


class OpenAITextConfig(BaseModel):
    format: Annotated[
        OpenAITextFormat | OpenAIJsonSchemaFormat | OpenAIJsonObjectFormat | None,
        Field(
            description=(
                "An object specifying the format that the model must output. "
                'Configuring `{ "type": "json_schema" }` enables Structured Outputs, '
                "which ensures the model will match your supplied JSON schema. "
                'The default format is `{ "type": "text" }` with no additional options. '
                'Setting to `{ "type": "json_object" }` enables the older JSON mode, '
                "which ensures the message the model generates is valid JSON. "
                "Using `json_schema` is preferred for models that support it."
            ),
        ),
    ] = None
    verbosity: Annotated[
        Literal["low", "medium", "high"],
        Field(
            description=(
                "Constrains the verbosity of the model's response. "
                "Lower values will result in more concise responses, "
                "while higher values will result in more verbose responses. "
                "Currently supported values are `low`, `medium`, and `high`."
            ),
        ),
    ] = "medium"

    @property
    def requested_verbosity(self) -> Literal["low", "medium", "high"] | None:
        """Return verbosity only when the client explicitly supplied it."""
        if "verbosity" not in self.model_fields_set:
            return None
        return self.verbosity


class OpenAINativeToolChoice(BaseModel):
    """
    Indicates that the model should use a native tool to generate a response.
    """

    type: Annotated[
        Literal[
            "code_interpreter",
            "image_generation",
            "web_search",
            "web_search_preview",
            "web_search_2025_08_26",
        ],
        Field(
            description=(
                "The type of native tool the model should use. Allowed values are:\n"
                "- `code_interpreter`\n"
                "- `image_generation`\n"
                "- `web_search`\n"
                "- `web_search_preview`\n"
                "- `web_search_2025_08_26`"
            ),
        ),
    ]


class OpenAIFunctionToolChoice(BaseModel):
    """
    Use this option to force the model to call a specific function.
    """

    name: Annotated[
        str,
        Field(
            description="The name of the function to call.",
        ),
    ]
    type: Annotated[
        Literal["function"],
        Field(
            description="For function calling, the type is always `function`.",
        ),
    ] = "function"


class OpenAIMcpToolChoice(BaseModel):
    """
    Use this option to force the model to call a specific tool on a remote MCP server.
    """

    server_label: Annotated[
        str,
        Field(
            description="The label of the MCP server to use.",
        ),
    ]
    type: Annotated[
        Literal["mcp"],
        Field(
            description="For MCP tools, the type is always `mcp`.",
        ),
    ] = "mcp"
    name: Annotated[
        str | None,
        Field(
            description="The name of the tool to call on the server.",
        ),
    ] = None


class OpenAIAllowedToolsChoice(BaseModel):
    """
    Constrains the tools available to the model to a pre-defined set.
    """

    mode: Annotated[
        Literal["auto", "required"],
        Field(
            description=(
                "Constrains the tools available to the model to a pre-defined set. "
                "`auto` allows the model to pick from among the allowed tools and generate a message. "
                "`required` requires the model to call one or more of the allowed tools."
            ),
        ),
    ]
    tools: Annotated[
        list[OpenAINativeToolChoice | OpenAIFunctionToolChoice | OpenAIMcpToolChoice],
        Field(
            description=(
                "A list of tool definitions that the model should be allowed to call. "
                "For the Responses API, the list of tool definitions might look like: \n"
                "```\n"
                "[\n"
                '  { "type": "function", "name": "get_weather" },\n'
                '  { "type": "mcp", "server_label": "deepwiki" },\n'
                '  { "type": "image_generation" }\n'
                "]\n"
                "```"
            ),
        ),
    ]
    type: Annotated[
        Literal["allowed_tools"],
        Field(
            description="Allowed tool configuration type. Always `allowed_tools`.",
        ),
    ] = "allowed_tools"

    def validate_hosted_tools(self) -> None:
        if any(
            isinstance(tool, OpenAINativeToolChoice)
            and _normalize_hosted_tool_type(tool.type) not in {"code_interpreter", "web_search"}
            for tool in self.tools
        ):
            raise ValueError(
                "vLLM does not support hosted tools other than code interpreter and web_search."
            )


OpenAIToolChoice = Union[
    Literal["none", "auto", "required"],
    Annotated[
        Union[
            OpenAIAllowedToolsChoice,
            OpenAINativeToolChoice,
            OpenAIFunctionToolChoice,
            OpenAIMcpToolChoice,
        ],
        Discriminator("type"),
    ],
]


class OpenAIResponsesFunctionTool(BaseModel):
    """
    Defines a function in your own code the model can choose to call.
    """

    name: Annotated[
        str,
        Field(
            description="The name of the function to call.",
        ),
    ]
    parameters: Annotated[
        # TODO: Perhaps we should skip validation?
        JsonSchema,
        Field(
            description="A JSON schema object describing the parameters of the function.",
        ),
    ]
    strict: Annotated[
        bool,
        Field(
            description="Whether to enforce strict parameter validation. Defaults to `true`.",
        ),
    ] = True
    type: Annotated[
        Literal["function"],
        Field(
            description="The type of the function tool. Always `function`.",
        ),
    ] = "function"
    description: Annotated[
        str | None,
        Field(
            description=(
                "A description of the function. "
                "Used by the model to determine whether or not to call the function."
            ),
        ),
    ] = None


class OpenAIResponsesMcpTool(BaseModel):
    """MCP server declaration for gateway-side execution."""

    model_config = {"extra": "forbid"}

    type: Annotated[
        Literal["mcp"],
        Field(description="The MCP tool type. Always `mcp`."),
    ] = "mcp"
    server_label: Annotated[
        str,
        Field(description="The request-visible MCP server label."),
    ]
    server_url: Annotated[
        str | None,
        Field(
            description=(
                "Optional request-remote MCP endpoint URL. "
                "When omitted, the declaration resolves to hosted mode by `server_label`."
            )
        ),
    ] = None
    connector_id: Annotated[
        str | None,
        Field(description="Connector-based MCP mode is unsupported by this gateway."),
    ] = None
    headers: Annotated[
        dict[str, str] | None,
        Field(
            description=(
                "Optional request-scoped headers for request-remote MCP declarations. "
                "Hosted MCP declarations do not accept `headers`."
            )
        ),
    ] = None
    authorization: Annotated[
        str | None,
        Field(
            description=(
                "Optional request-scoped bearer token for request-remote MCP. "
                "Mapped to outbound `Authorization: Bearer <token>`."
            )
        ),
    ] = None
    allowed_tools: Annotated[
        list[str] | None,
        Field(description="Optional request-scoped tool allowlist for this MCP declaration."),
    ] = None
    require_approval: Annotated[
        str | None,
        Field(description="Approval policy for MCP calls. Only `never` is supported."),
    ] = None


class OpenAIResponsesWebSearchFilters(BaseModel):
    allowed_domains: Annotated[
        list[str],
        Field(
            description=(
                "Allowed domains for the search. "
                "If not provided, all domains are allowed. "
                "Subdomains of the provided domains are allowed as well."
            ),
            examples=["pubmed.ncbi.nlm.nih.gov"],
        ),
    ] = []


class OpenAIResponsesWebSearchUserLocation(BaseModel):
    city: Annotated[
        str | None,
        Field(
            description="Free text input for the city of the user, e.g. `San Francisco`.",
        ),
    ] = None
    country: Annotated[
        str | None,
        Field(
            max_length=2,
            description="The two-letter ISO country code of the user, e.g. `US`.",
        ),
    ] = None
    region: Annotated[
        str | None,
        Field(
            description="Free text input for the region of the user, e.g. `California`.",
        ),
    ] = None
    timezone: Annotated[
        TimeZoneName | None,
        Field(
            description="The IANA timezone of the user, e.g. `America/Los_Angeles`.",
        ),
    ] = None
    type: Annotated[
        Literal["approximate"],
        Field(
            description="The type of location approximation. Always `approximate`.",
        ),
    ] = "approximate"


class OpenAIResponsesWebSearchTool(BaseModel):
    """
    Search the Internet for sources related to the prompt.
    """

    type: Annotated[
        Literal[
            "web_search",
            "web_search_2025_08_26",
            "web_search_preview",
            "web_search_preview_2025_03_11",
        ],
        Field(description="The type of tool."),
    ] = "web_search"
    filters: Annotated[
        OpenAIResponsesWebSearchFilters | None,
        Field(
            description="Filters for the search.",
        ),
    ] = None
    search_context_size: Annotated[
        Literal["low", "medium", "high"],
        Field(
            description=(
                "High level guidance for the amount of context window space to use for the search. "
                "One of `low`, `medium`, or `high`. `medium` is the default."
            ),
        ),
    ] = "medium"
    search_content_types: Annotated[
        list[Literal["text", "image"]] | None,
        Field(description="Content types included by preview web search."),
    ] = None
    return_token_budget: Annotated[
        str | int | None,
        Field(description="Provider-reported token budget for web search results."),
    ] = None
    user_location: Annotated[
        OpenAIResponsesWebSearchUserLocation | None,
        Field(
            description="The approximate location of the user.",
        ),
    ] = None


class OpenAIResponsesCodeContainer(BaseModel):
    """
    Configuration for a code interpreter container. Optionally specify the IDs of the files to run the code on.
    """

    type: Annotated[
        Literal["auto"],
        Field(description=("Always `auto`.")),
    ] = "auto"
    file_ids: Annotated[
        list[str] | None,
        Field(
            description=("An optional list of uploaded files to make available to your code."),
        ),
    ] = None
    memory_limit: Annotated[
        str | None,
        Field(
            description=("The memory limit for the code interpreter container."),
        ),
    ] = None


class OpenAIResponsesCodeToolBase(BaseModel):
    type: Annotated[
        Literal["code_interpreter"],
        Field(description="The type of tool."),
    ] = "code_interpreter"


class OpenAIResponsesCodeTool(OpenAIResponsesCodeToolBase):
    container: Annotated[
        str | OpenAIResponsesCodeContainer,
        Field(
            description=(
                "The code interpreter container. "
                "Can be a container ID or an object that specifies uploaded file IDs to make available to your code, "
                "along with an optional `memory_limit` setting."
            ),
        ),
    ] = OpenAIResponsesCodeContainer()


class OpenAIResponsesImageGenerationInputImageMask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: Annotated[
        str | None,
        Field(description="The file ID for the mask image."),
    ] = None
    image_url: Annotated[
        str | None,
        Field(description="The base64-encoded mask image."),
    ] = None


class OpenAIResponsesImageGenerationTool(BaseModel):
    """A tool that generates images using the GPT image models."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image_generation"] = "image_generation"
    action: Literal["generate", "edit", "auto"] | None = None
    background: Literal["transparent", "opaque", "auto"] | None = None
    input_fidelity: Literal["high", "low"] | None = None
    input_image_mask: OpenAIResponsesImageGenerationInputImageMask | None = None
    model: str | None = None
    moderation: Literal["auto", "low"] | None = None
    output_compression: int | None = None
    output_format: Literal["png", "webp", "jpeg"] | None = None
    partial_images: Annotated[int | None, Field(ge=0, le=3)] = None
    quality: Literal["low", "medium", "high", "auto"] | None = None
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = None


class OpenAIResponsesOutputImageGenerationTool(OpenAIResponsesImageGenerationTool):
    """Image-generation tool metadata returned in a Responses snapshot."""

    n: Annotated[int | None, Field(ge=1)] = None


OpenAIResponsesRestrictedTool = Annotated[
    OpenAIResponsesCodeToolBase
    | OpenAIResponsesWebSearchTool
    | OpenAIResponsesFunctionTool
    | OpenAIResponsesMcpTool,
    Field(discriminator="type", description="The type of tool."),
]
OpenAIResponsesTool = Annotated[
    OpenAIResponsesWebSearchTool
    | OpenAIResponsesCodeTool
    | OpenAIResponsesImageGenerationTool
    | OpenAIResponsesFunctionTool
    | OpenAIResponsesMcpTool,
    Field(discriminator="type", description="The type of tool."),
]
OpenAIResponsesOutputTool = Annotated[
    OpenAIResponsesWebSearchTool
    | OpenAIResponsesCodeTool
    | OpenAIResponsesOutputImageGenerationTool
    | OpenAIResponsesFunctionTool
    | OpenAIResponsesMcpTool,
    Field(discriminator="type", description="The type of tool returned by the provider."),
]


class OpenAIStreamOptions(BaseModel):
    include_obfuscation: Annotated[
        bool | None,
        Field(
            description=(
                "When `true`, stream obfuscation will be enabled. "
                "Stream obfuscation adds random characters to an obfuscation field on streaming delta events to normalize payload sizes as a mitigation to certain side-channel attacks. "
                "These obfuscation fields are included by default, but add a small amount of overhead to the data stream. "
                "You can set include_obfuscation to false to optimize for bandwidth if you trust the network links between your application and the API."
            ),
        ),
    ] = None
