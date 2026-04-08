from typing import Annotated, Any, Literal, Union

from httpx import AsyncClient
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import ModelMessage

RESPONSE_TOOL_NAME = "final_result"
RESPONSE_KEY = "response"


class _Message(BaseModel):
    type: Annotated[
        str,
        Field(description="Message type."),
    ]


class _MessageWithContent(_Message):
    content: Annotated[
        str,
        Field(description="Message content."),
    ]


class UserMessage(_MessageWithContent):
    type: Annotated[
        Literal["user.message"],
        Field(description="Message type."),
    ] = "user.message"
    content: Annotated[
        str,
        Field(description="Message content."),
    ]


class AgentMessageStart(_Message):
    type: Annotated[
        Literal["agent.message.start"],
        Field(description="Message type."),
    ] = "agent.message.start"


class AgentMessageDelta(_Message):
    type: Annotated[
        Literal["agent.message.delta"],
        Field(description="Message type."),
    ] = "agent.message.delta"
    content: str


class AgentMessageEnd(_Message):
    type: Annotated[
        Literal["agent.message.end"],
        Field(description="Message type."),
    ] = "agent.message.end"


class AgentMessage(_MessageWithContent):
    type: Annotated[
        Literal["agent.message"],
        Field(description="Message type."),
    ] = "agent.message"


class AgentThoughtStart(_Message):
    type: Annotated[
        Literal["agent.thought.start"],
        Field(description="Message type."),
    ] = "agent.thought.start"


class AgentThoughtDelta(_Message):
    type: Annotated[
        Literal["agent.thought.delta"],
        Field(description="Message type."),
    ] = "agent.thought.delta"
    content: str
    signature: str


class AgentThoughtEnd(_Message):
    type: Annotated[
        Literal["agent.thought.end"],
        Field(description="Message type."),
    ] = "agent.thought.end"


class AgentThought(_MessageWithContent):
    type: Annotated[
        Literal["agent.thought"],
        Field(description="Message type."),
    ] = "agent.thought"


class AgentToolStart(_Message):
    type: Annotated[
        Literal["agent.tool.start"],
        Field(description="Message type."),
    ] = "agent.tool.start"
    tool_call_id: str
    tool_name: str


class AgentToolDelta(_Message):
    type: Annotated[
        Literal["agent.tool.delta"],
        Field(description="Message type."),
    ] = "agent.tool.delta"
    tool_call_id: str
    tool_name: str
    args: str


class AgentToolEnd(_Message):
    type: Annotated[
        Literal["agent.tool.end"],
        Field(description="Message type."),
    ] = "agent.tool.end"
    tool_call_id: str


class AgentToolCall(BaseModel):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


class AgentToolCalls(_Message):
    type: Annotated[
        Literal["agent.tool.call"],
        Field(description="Message type."),
    ] = "agent.tool.call"
    tool_calls: list[AgentToolCall]


class AgentToolResult(_Message):
    type: Annotated[
        Literal["agent.tool.result"],
        Field(description="Message type."),
    ] = "agent.tool.result"
    tool_call_id: str
    tool_name: str
    result: Any


class AgentError(_Message):
    type: Annotated[
        Literal["agent.error"],
        Field(description="Message type."),
    ] = "agent.error"
    error: str
    cause: str

class InputTokensDetails(BaseModel):
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens_details: OutputTokensDetails = Field(default_factory=OutputTokensDetails)


class AgentRunResult(_Message):
    type: Annotated[
        Literal["agent.run.result"],
        Field(description="Message type."),
    ] = "agent.run.result"
    tool_calls: list[AgentToolCall] | None = None
    message: str | None = None
    error: str | None = None
    usage: Usage | None = None



AgentEvent = Annotated[
    Union[
        AgentMessageStart,
        AgentMessageDelta,
        AgentMessageEnd,
        AgentMessage,
        AgentThoughtStart,
        AgentThoughtDelta,
        AgentThoughtEnd,
        AgentThought,
        AgentToolStart,
        AgentToolDelta,
        AgentToolEnd,
        AgentToolCall,
        AgentToolResult,
        AgentError,
        AgentRunResult,
    ],
    Field(discriminator="type"),
]
NormalisedHistoryItem = Annotated[
    Union[
        UserMessage,
        AgentMessage,
        AgentThought,
        AgentToolCall,
        AgentToolResult,
    ],
    Field(discriminator="type"),
]

# The Output IR is the stream of AgentEvent values produced by Runner.run_stream().
OutputIR = AgentEvent


# ---------------------------------------------------------------------------
# Input IR — provider-agnostic request bundle produced by Type.to_ir()
# ---------------------------------------------------------------------------

class TextFormat(BaseModel):
    kind: Literal["text"] = "text"


class JsonObjectFormat(BaseModel):
    kind: Literal["json_object"] = "json_object"


class JsonSchemaFormat(BaseModel):
    kind: Literal["json_schema"] = "json_schema"
    name: str
    json_schema: dict[str, Any]
    strict: bool = False
    description: str | None = None


TextOutputFormat = Annotated[
    Union[TextFormat, JsonObjectFormat, JsonSchemaFormat],
    Field(discriminator="kind"),
]


# ---- Tool types ----


class FunctionTool(BaseModel):
    kind: Literal["function"] = "function"
    name: str
    parameters: dict[str, Any]
    strict: bool = True
    description: str | None = None


class CodeInterpreterTool(BaseModel):
    kind: Literal["code_interpreter"] = "code_interpreter"


class WebSearchUserLocation(BaseModel):
    city: str | None = None
    country: str | None = None  # 2-letter ISO country code
    region: str | None = None
    timezone: str | None = None  # IANA timezone string


class WebSearchTool(BaseModel):
    kind: Literal["web_search"] = "web_search"
    search_context_size: Literal["low", "medium", "high"] = "medium"
    allowed_domains: list[str] = Field(default_factory=list)
    user_location: WebSearchUserLocation | None = None


class McpTool(BaseModel):
    kind: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str | None = None
    headers: dict[str, str] | None = None
    authorization: str | None = None
    allowed_tools: list[str] | None = None
    require_approval: str | None = None


Tool = Annotated[
    Union[FunctionTool, CodeInterpreterTool, WebSearchTool, McpTool],
    Field(discriminator="kind"),
]


# ---- Tool choice types ----


class BuiltinToolChoice(BaseModel):
    kind: Literal["builtin"] = "builtin"
    name: Literal["code_interpreter", "web_search"]


class FunctionToolChoice(BaseModel):
    kind: Literal["function"] = "function"
    name: str


class McpToolChoice(BaseModel):
    kind: Literal["mcp"] = "mcp"
    server_label: str
    name: str | None = None


_SpecificToolChoice = Annotated[
    Union[BuiltinToolChoice, FunctionToolChoice, McpToolChoice],
    Field(discriminator="kind"),
]


class AllowedToolsChoice(BaseModel):
    kind: Literal["allowed_tools"] = "allowed_tools"
    mode: Literal["auto", "required"]
    tools: list[_SpecificToolChoice]


ToolChoice = Union[
    Literal["none", "auto", "required"],
    Annotated[
        Union[AllowedToolsChoice, BuiltinToolChoice, FunctionToolChoice, McpToolChoice],
        Field(discriminator="kind"),
    ],
]


# ---- InputIR ----


class InputIR(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Routing
    model: str
    stream: bool | None = None

    # Conversation (pydantic-ai abstraction)
    message_history: list[ModelMessage]
    instructions: str | None

    # Sampling parameters (provider renames/maps these for its backend)
    temperature: float
    top_p: float
    max_output_tokens: int | None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    reasoning_summary: Literal["auto", "concise", "detailed"] | None
    top_logprobs: int | None
    parallel_tool_calls: bool | None = None
    max_tool_calls: int | None

    # Text output format
    text_format: TextOutputFormat | None

    # Tools
    tools: list[Tool] | None
    tool_choice: ToolChoice | None

    # Passthrough metadata
    include: list[str] | None
    store: bool | None = None
    previous_response_id: str | None
    metadata: dict[str, Any] | None
