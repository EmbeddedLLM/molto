from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

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


class AgentToolDelta(_Message):
    type: Annotated[
        Literal["agent.tool.delta"],
        Field(description="Message type."),
    ] = "agent.tool.delta"
    tool_name: str
    args: str


class AgentToolEnd(_Message):
    type: Annotated[
        Literal["agent.tool.end"],
        Field(description="Message type."),
    ] = "agent.tool.end"


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
