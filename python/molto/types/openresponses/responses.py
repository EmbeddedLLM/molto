from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from molto.types._portable_adapters import (
    REASONING_CONTENT,
    REASONING_ENCRYPTED,
    REASONING_SUMMARY,
    dispatch_output_item,
)
from molto.types.ir.portable import (
    AllowedFunctionTools,
    AssistantMessageItem,
    ContentPartAddedEvent,
    ContentPartDeltaEvent,
    ContentPartDoneEvent,
    ErrorDetail,
    FunctionArgumentsDeltaEvent,
    FunctionArgumentsDoneEvent,
    FunctionCallItem,
    FunctionCallRequestItem,
    FunctionCallResultItem,
    FunctionToolChoice,
    FunctionToolDefinition,
    GenerationConfig,
    InputImagePart,
    InputIR,
    InputTextPart,
    ItemStatus,
    JsonObjectOutputFormat,
    JsonSchemaOutputFormat,
    OutputIR,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputStreamEvent,
    OutputTextPart,
    PreservedOpenAIResponseEvent,
    PreservedOpenAIStreamEvent,
    ReasoningContentPart,
    ReasoningItem,
    ReasoningSummaryPart,
    RefusalPart,
    RequestMessageItem,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseStartedEvent,
    TextOutputFormat,
)


class _PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenResponsesInputText(_PublicModel):
    type: Literal["input_text"] = "input_text"
    text: str


class OpenResponsesInputImage(_PublicModel):
    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: Literal["auto", "low", "high"] = "auto"


OpenResponsesInputContent = Annotated[
    Union[OpenResponsesInputText, OpenResponsesInputImage],
    Field(discriminator="type"),
]


class OpenResponsesMessageInput(_PublicModel):
    type: Literal["message"] = "message"
    id: str | None = None
    role: Literal["system", "developer", "user", "assistant"]
    content: tuple[OpenResponsesInputContent, ...]


class OpenResponsesFunctionCallInput(_PublicModel):
    type: Literal["function_call"] = "function_call"
    id: str | None = None
    status: ItemStatus | None = None
    call_id: str
    name: str
    arguments: str


class OpenResponsesFunctionCallOutput(_PublicModel):
    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    call_id: str
    output: str


OpenResponsesInputItem = Annotated[
    Union[
        OpenResponsesMessageInput,
        OpenResponsesFunctionCallInput,
        OpenResponsesFunctionCallOutput,
    ],
    Field(discriminator="type"),
]


class OpenResponsesTextFormat(_PublicModel):
    type: Literal["text"] = "text"


class OpenResponsesJsonObjectFormat(_PublicModel):
    type: Literal["json_object"] = "json_object"


class OpenResponsesJsonSchemaFormat(_PublicModel):
    model_config = ConfigDict(
        serialize_by_alias=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    type: Literal["json_schema"] = "json_schema"
    name: str
    schema_: dict[str, object] = Field(alias="schema", serialization_alias="schema")
    description: str | None = None
    strict: bool | None = None


OpenResponsesOutputFormat = Annotated[
    Union[
        OpenResponsesTextFormat,
        OpenResponsesJsonObjectFormat,
        OpenResponsesJsonSchemaFormat,
    ],
    Field(discriminator="type"),
]


class OpenResponsesFunctionTool(_PublicModel):
    type: Literal["function"] = "function"
    name: str
    parameters: dict[str, object] | None = None
    description: str | None = None
    strict: bool | None = None


class OpenResponsesFunctionToolChoice(_PublicModel):
    type: Literal["function"] = "function"
    name: str


class OpenResponsesAllowedToolsChoice(_PublicModel):
    type: Literal["allowed_tools"] = "allowed_tools"
    mode: Literal["auto", "required"]
    tools: tuple[OpenResponsesFunctionToolChoice, ...]


OpenResponsesToolChoice: TypeAlias = (
    Literal["none", "auto", "required"]
    | OpenResponsesFunctionToolChoice
    | OpenResponsesAllowedToolsChoice
)


class OpenResponsesRequest(_PublicModel):
    model: str
    input: str | tuple[OpenResponsesInputItem, ...]
    stream: bool = False
    instructions: str | None = None
    text: OpenResponsesOutputFormat = OpenResponsesTextFormat()
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    top_p: float | None = Field(default=None, ge=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    top_logprobs: int | None = Field(default=None, ge=0)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None
    parallel_tool_calls: bool | None = None
    max_tool_calls: int | None = Field(default=None, ge=1)
    tools: tuple[OpenResponsesFunctionTool, ...] = ()
    tool_choice: OpenResponsesToolChoice | None = None

    def to_ir(self) -> InputIR:
        if isinstance(self.input, str):
            items = (
                RequestMessageItem(
                    role="user",
                    content=(InputTextPart(text=self.input),),
                ),
            )
        else:
            items = tuple(_input_item_to_ir(item) for item in self.input)

        if isinstance(self.text, OpenResponsesJsonObjectFormat):
            output_format = JsonObjectOutputFormat()
        elif isinstance(self.text, OpenResponsesJsonSchemaFormat):
            output_format = JsonSchemaOutputFormat(
                name=self.text.name,
                schema=self.text.schema_,
                description=self.text.description,
                strict=self.text.strict,
            )
        else:
            output_format = TextOutputFormat()

        tool_choice: object = self.tool_choice
        if isinstance(tool_choice, OpenResponsesFunctionToolChoice):
            tool_choice = FunctionToolChoice(name=tool_choice.name)
        elif isinstance(tool_choice, OpenResponsesAllowedToolsChoice):
            tool_choice = AllowedFunctionTools(
                mode=tool_choice.mode,
                tools=tuple(FunctionToolChoice(name=tool.name) for tool in tool_choice.tools),
            )

        return InputIR(
            model=self.model,
            stream=self.stream,
            instructions=self.instructions,
            items=items,
            output_format=output_format,
            generation=GenerationConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                top_logprobs=self.top_logprobs,
                reasoning_effort=self.reasoning_effort,
                reasoning_summary=self.reasoning_summary,
                parallel_tool_calls=self.parallel_tool_calls,
                max_tool_calls=self.max_tool_calls,
            ),
            tools=tuple(
                FunctionToolDefinition(
                    name=tool.name,
                    parameters=tool.parameters,
                    description=tool.description,
                    strict=tool.strict,
                )
                for tool in self.tools
            ),
            tool_choice=tool_choice,
        )


def _input_item_to_ir(item: OpenResponsesInputItem):
    if isinstance(item, OpenResponsesMessageInput):
        return RequestMessageItem(
            id=item.id,
            role=item.role,
            content=tuple(
                InputTextPart(text=part.text)
                if isinstance(part, OpenResponsesInputText)
                else InputImagePart(image_url=part.image_url, detail=part.detail)
                for part in item.content
            ),
        )
    if isinstance(item, OpenResponsesFunctionCallInput):
        return FunctionCallRequestItem(
            id=item.id,
            status=item.status,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )
    return FunctionCallResultItem(
        id=item.id,
        call_id=item.call_id,
        output=item.output,
    )


class OpenResponsesOutputText(_PublicModel):
    type: Literal["output_text"] = "output_text"
    text: str


class OpenResponsesRefusal(_PublicModel):
    type: Literal["refusal"] = "refusal"
    refusal: str


class OpenResponsesReasoningSummary(_PublicModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class OpenResponsesReasoningText(_PublicModel):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str
    provider: str


class OpenResponsesEncryptedReasoning(_PublicModel):
    type: Literal["encrypted_reasoning"] = "encrypted_reasoning"
    data: str
    provider: str


OpenResponsesAssistantContent = Annotated[
    Union[OpenResponsesOutputText, OpenResponsesRefusal],
    Field(discriminator="type"),
]
OpenResponsesReasoningContent = Annotated[
    Union[
        OpenResponsesReasoningSummary,
        OpenResponsesReasoningText,
        OpenResponsesEncryptedReasoning,
    ],
    Field(discriminator="type"),
]
OpenResponsesContentPart = Annotated[
    Union[
        OpenResponsesOutputText,
        OpenResponsesRefusal,
        OpenResponsesReasoningSummary,
        OpenResponsesReasoningText,
        OpenResponsesEncryptedReasoning,
    ],
    Field(discriminator="type"),
]


class OpenResponsesMessageOutput(_PublicModel):
    type: Literal["message"] = "message"
    id: str
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    role: Literal["assistant"] = "assistant"
    content: tuple[OpenResponsesAssistantContent, ...]
    phase: Literal["commentary", "final_answer"] | None = None


class OpenResponsesReasoningOutput(_PublicModel):
    type: Literal["reasoning"] = "reasoning"
    id: str
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    content: tuple[OpenResponsesReasoningContent, ...]


class OpenResponsesFunctionCall(_PublicModel):
    type: Literal["function_call"] = "function_call"
    id: str
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    call_id: str
    name: str
    arguments: str


OpenResponsesOutputItem = Annotated[
    Union[
        OpenResponsesMessageOutput,
        OpenResponsesReasoningOutput,
        OpenResponsesFunctionCall,
    ],
    Field(discriminator="type"),
]


class OpenResponsesUsage(_PublicModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    reasoning_output_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_breakdowns(self) -> OpenResponsesUsage:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens.")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens.")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning_output_tokens cannot exceed output_tokens.")
        return self


OpenResponsesError: TypeAlias = ErrorDetail


def _usage_from_ir(usage) -> OpenResponsesUsage:
    return OpenResponsesUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        reasoning_output_tokens=usage.reasoning_output_tokens,
    )


class OpenResponsesResponse(_PublicModel):
    id: str
    status: Literal["in_progress", "completed", "incomplete", "failed", "cancelled"]
    output: tuple[OpenResponsesOutputItem, ...]
    usage: OpenResponsesUsage
    error: OpenResponsesError | None = None
    incomplete_reason: str | None = None


def from_output_ir(output: OutputIR) -> OpenResponsesResponse:
    return OpenResponsesResponse(
        id=output.id,
        status=output.status.value,
        output=tuple(_output_item_from_output_ir(item) for item in output.output.values),
        usage=_usage_from_ir(output.usage),
        error=output.error,
        incomplete_reason=output.incomplete_reason,
    )


def _assistant_content_from_output_ir(part, content_index: int = 0):
    del content_index
    if isinstance(part, OutputTextPart):
        return OpenResponsesOutputText(text=part.text)
    return OpenResponsesRefusal(refusal=part.refusal)


def _reasoning_content_from_output_ir(part) -> tuple[str, object]:
    if isinstance(part, ReasoningSummaryPart):
        return REASONING_SUMMARY, OpenResponsesReasoningSummary(text=part.text)
    if isinstance(part, ReasoningContentPart):
        return REASONING_CONTENT, OpenResponsesReasoningText(
            text=part.text, provider=part.provider
        )
    return REASONING_ENCRYPTED, OpenResponsesEncryptedReasoning(
        data=part.data, provider=part.provider
    )


def _output_item_from_output_ir(item):
    def assistant_message(item: AssistantMessageItem, content: list) -> OpenResponsesMessageOutput:
        return OpenResponsesMessageOutput(
            id=item.id,
            status=item.status.value,
            content=tuple(content),
            phase=item.phase,
        )

    def reasoning_item(item: ReasoningItem, parts: list[tuple[str, object]]):
        return OpenResponsesReasoningOutput(
            id=item.id,
            status=item.status.value,
            content=tuple(rendered for _tag, rendered in parts),
        )

    def function_call(item: FunctionCallItem) -> OpenResponsesFunctionCall:
        return OpenResponsesFunctionCall(
            id=item.id,
            status=item.status.value,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )

    return dispatch_output_item(
        item,
        assistant_message=assistant_message,
        assistant_content=_assistant_content_from_output_ir,
        reasoning_item=reasoning_item,
        reasoning_content=_reasoning_content_from_output_ir,
        function_call=function_call,
    )


class OpenResponsesResponseStarted(_PublicModel):
    type: Literal["response.started"] = "response.started"
    response_id: str
    sequence_number: int
    status: Literal["in_progress"] = "in_progress"


class OpenResponsesOutputItemAdded(_PublicModel):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    response_id: str
    sequence_number: int
    item_index: int
    item: OpenResponsesOutputItem


class OpenResponsesContentPartAdded(_PublicModel):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    response_id: str
    sequence_number: int
    item_id: str
    item_index: int
    content_index: int
    part: OpenResponsesContentPart


class OpenResponsesContentPartDelta(_PublicModel):
    type: Literal["response.content_part.delta"] = "response.content_part.delta"
    response_id: str
    sequence_number: int
    item_id: str
    item_index: int
    content_index: int
    delta: str


class OpenResponsesContentPartDone(OpenResponsesContentPartAdded):
    type: Literal["response.content_part.done"] = "response.content_part.done"


class OpenResponsesFunctionArgumentsDelta(_PublicModel):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    response_id: str
    sequence_number: int
    item_id: str
    item_index: int
    delta: str


class OpenResponsesFunctionArgumentsDone(_PublicModel):
    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    response_id: str
    sequence_number: int
    item_id: str
    item_index: int
    arguments: str


class OpenResponsesOutputItemDone(OpenResponsesOutputItemAdded):
    type: Literal["response.output_item.done"] = "response.output_item.done"


class OpenResponsesResponseCompleted(_PublicModel):
    type: Literal["response.completed"] = "response.completed"
    response_id: str
    sequence_number: int
    status: Literal["completed"] = "completed"
    usage: OpenResponsesUsage


class OpenResponsesResponseIncomplete(_PublicModel):
    type: Literal["response.incomplete"] = "response.incomplete"
    response_id: str
    sequence_number: int
    status: Literal["incomplete"] = "incomplete"
    reason: str
    usage: OpenResponsesUsage


class OpenResponsesResponseFailed(_PublicModel):
    type: Literal["response.failed"] = "response.failed"
    response_id: str
    sequence_number: int
    status: Literal["failed"] = "failed"
    error: OpenResponsesError
    usage: OpenResponsesUsage


OpenResponsesStreamEvent: TypeAlias = (
    OpenResponsesResponseStarted
    | OpenResponsesOutputItemAdded
    | OpenResponsesContentPartAdded
    | OpenResponsesContentPartDelta
    | OpenResponsesContentPartDone
    | OpenResponsesFunctionArgumentsDelta
    | OpenResponsesFunctionArgumentsDone
    | OpenResponsesOutputItemDone
    | OpenResponsesResponseCompleted
    | OpenResponsesResponseIncomplete
    | OpenResponsesResponseFailed
)


async def from_output_event_stream(
    events: AsyncIterable[OutputStreamEvent],
) -> AsyncIterator[OpenResponsesStreamEvent]:
    async for event in events:
        if isinstance(event, (PreservedOpenAIResponseEvent, PreservedOpenAIStreamEvent)):
            continue
        yield _event_from_output_event(event)


def _event_from_output_event(event) -> OpenResponsesStreamEvent:
    common = {
        "response_id": event.response_id,
        "sequence_number": event.sequence_number,
    }
    if isinstance(event, ResponseStartedEvent):
        return OpenResponsesResponseStarted(**common)
    if isinstance(event, OutputItemAddedEvent):
        return OpenResponsesOutputItemAdded(
            **common,
            item_index=event.item_index,
            item=_output_item_from_output_ir(event.item),
        )
    if isinstance(event, ContentPartAddedEvent):
        converter = (
            _assistant_content_from_output_ir
            if isinstance(event.part, (OutputTextPart, RefusalPart))
            else lambda part: _reasoning_content_from_output_ir(part)[1]
        )
        return OpenResponsesContentPartAdded(
            **common,
            item_id=event.item_id,
            item_index=event.item_index,
            content_index=event.content_index,
            part=converter(event.part),
        )
    if isinstance(event, ContentPartDeltaEvent):
        return OpenResponsesContentPartDelta(
            **common,
            item_id=event.item_id,
            item_index=event.item_index,
            content_index=event.content_index,
            delta=event.delta,
        )
    if isinstance(event, ContentPartDoneEvent):
        converter = (
            _assistant_content_from_output_ir
            if isinstance(event.part, (OutputTextPart, RefusalPart))
            else lambda part: _reasoning_content_from_output_ir(part)[1]
        )
        return OpenResponsesContentPartDone(
            **common,
            item_id=event.item_id,
            item_index=event.item_index,
            content_index=event.content_index,
            part=converter(event.part),
        )
    if isinstance(event, FunctionArgumentsDeltaEvent):
        return OpenResponsesFunctionArgumentsDelta(
            **common,
            item_id=event.item_id,
            item_index=event.item_index,
            delta=event.delta,
        )
    if isinstance(event, FunctionArgumentsDoneEvent):
        return OpenResponsesFunctionArgumentsDone(
            **common,
            item_id=event.item_id,
            item_index=event.item_index,
            arguments=event.arguments,
        )
    if isinstance(event, OutputItemDoneEvent):
        return OpenResponsesOutputItemDone(
            **common,
            item_index=event.item_index,
            item=_output_item_from_output_ir(event.item),
        )
    if isinstance(event, ResponseCompletedEvent):
        return OpenResponsesResponseCompleted(**common, usage=_usage_from_ir(event.usage))
    if isinstance(event, ResponseIncompleteEvent):
        return OpenResponsesResponseIncomplete(
            **common,
            reason=event.reason,
            usage=_usage_from_ir(event.usage),
        )
    if isinstance(event, ResponseFailedEvent):
        return OpenResponsesResponseFailed(
            **common,
            error=event.error,
            usage=_usage_from_ir(event.usage),
        )
    raise TypeError(f"Unsupported portable output event {type(event).__name__}.")
