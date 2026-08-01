from collections.abc import AsyncIterator

from molto.types.ir.portable import (
    AssistantMessageItem,
    ContentPartAddedEvent,
    ContentPartDeltaEvent,
    ContentPartDoneEvent,
    FunctionCallRequestItem,
    FunctionCallResultItem,
    InputImagePart,
    InputTextPart,
    ItemStatus,
    OutputEventSequence,
    OutputIR,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputItems,
    OutputTextPart,
    ReasoningItem,
    ReasoningSummaryPart,
    RefusalPart,
    RequestMessageItem,
    ResponseCompletedEvent,
    ResponseStartedEvent,
    ResponseStatus,
    Usage,
)
from molto.types.openresponses import (
    OpenResponsesFunctionToolChoice,
    OpenResponsesJsonSchemaFormat,
    OpenResponsesRequest,
    OpenResponsesResponseCompleted,
    from_output_event_stream,
    from_output_ir,
)


def _completed_output() -> OutputIR:
    usage = Usage(input_tokens=3, output_tokens=2, total_tokens=5)
    started_item = AssistantMessageItem(
        id="msg_1",
        status=ItemStatus.IN_PROGRESS,
        content=(),
    )
    completed_item = AssistantMessageItem(
        id="msg_1",
        status=ItemStatus.COMPLETED,
        content=(OutputTextPart(text="Hello"),),
        phase="final_answer",
    )
    event_sequence = OutputEventSequence(
        events=(
            ResponseStartedEvent(response_id="resp_1", sequence_number=0),
            OutputItemAddedEvent(
                response_id="resp_1",
                sequence_number=1,
                item_index=0,
                item=started_item,
            ),
            ContentPartAddedEvent(
                response_id="resp_1",
                sequence_number=2,
                item_id="msg_1",
                item_index=0,
                content_index=0,
                part=OutputTextPart(text=""),
            ),
            ContentPartDeltaEvent(
                response_id="resp_1",
                sequence_number=3,
                item_id="msg_1",
                item_index=0,
                content_index=0,
                delta="Hello",
            ),
            ContentPartDoneEvent(
                response_id="resp_1",
                sequence_number=4,
                item_id="msg_1",
                item_index=0,
                content_index=0,
                part=OutputTextPart(text="Hello"),
            ),
            OutputItemDoneEvent(
                response_id="resp_1",
                sequence_number=5,
                item_index=0,
                item=completed_item,
            ),
            ResponseCompletedEvent(
                response_id="resp_1",
                sequence_number=6,
                usage=usage,
            ),
        )
    )
    return OutputIR(
        id="resp_1",
        status=ResponseStatus.COMPLETED,
        output=OutputItems(values=(completed_item,)),
        usage=usage,
        event_sequence=event_sequence,
    )


def test_request_converts_portable_input_without_provider_metadata() -> None:
    request = OpenResponsesRequest(
        model="openai/gpt-test",
        stream=True,
        instructions="Be concise.",
        input=(
            {
                "type": "message",
                "id": "user_1",
                "role": "user",
                "content": (
                    {"type": "input_text", "text": "Describe this."},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/image.png",
                        "detail": "high",
                    },
                ),
            },
            {
                "type": "function_call",
                "id": "call_item_1",
                "status": "completed",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"id":1}',
            },
            {
                "type": "function_call_output",
                "id": "result_1",
                "call_id": "call_1",
                "output": "done",
            },
        ),
        text=OpenResponsesJsonSchemaFormat(
            name="answer",
            schema={"type": "object"},
            strict=True,
        ),
        max_output_tokens=50,
        reasoning_summary="concise",
        tools=(
            {
                "type": "function",
                "name": "lookup",
                "parameters": {"type": "object"},
                "strict": True,
            },
        ),
        tool_choice=OpenResponsesFunctionToolChoice(name="lookup"),
    )

    ir = request.to_ir()

    assert ir.model == "openai/gpt-test"
    assert ir.stream is True
    assert isinstance(ir.items[0], RequestMessageItem)
    assert ir.items[0].content == (
        InputTextPart(text="Describe this."),
        InputImagePart(
            image_url="https://example.com/image.png",
            detail="high",
        ),
    )
    assert isinstance(ir.items[1], FunctionCallRequestItem)
    assert isinstance(ir.items[2], FunctionCallResultItem)
    assert ir.output_format.schema_ == {"type": "object"}
    assert ir.generation.max_output_tokens == 50
    assert ir.generation.reasoning_summary == "concise"
    assert ir.tool_choice.name == "lookup"


def test_string_input_becomes_one_user_text_message() -> None:
    ir = OpenResponsesRequest(model="anthropic/claude-test", input="Hello").to_ir()

    assert ir.items == (
        RequestMessageItem(
            role="user",
            content=(InputTextPart(text="Hello"),),
        ),
    )


def test_non_stream_output_composes_only_portable_fields() -> None:
    response = from_output_ir(_completed_output())

    assert response.status == "completed"
    assert type(response.status) is str
    assert response.output[0].status == "completed"
    assert type(response.output[0].status) is str
    assert response.model_dump(mode="json") == {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
                "phase": "final_answer",
            }
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
        },
        "error": None,
        "incomplete_reason": None,
    }


async def test_stream_composes_every_portable_output_event_in_order() -> None:
    output = _completed_output()

    async def events() -> AsyncIterator:
        for event in output.event_sequence.events:
            yield event

    public_events = [event async for event in from_output_event_stream(events())]

    assert len(public_events) == len(output.event_sequence.events)
    assert [event.sequence_number for event in public_events] == list(range(7))
    assert public_events[3].type == "response.content_part.delta"
    assert public_events[3].delta == "Hello"
    assert public_events[0].status == "in_progress"
    assert type(public_events[0].status) is str
    assert isinstance(public_events[-1], OpenResponsesResponseCompleted)
    assert public_events[-1].status == "completed"
    assert type(public_events[-1].status) is str
    assert public_events[-1].usage.total_tokens == 5


async def test_stream_content_part_events_carry_typed_parts() -> None:
    events = (
        ResponseStartedEvent(response_id="resp_1", sequence_number=0),
        OutputItemAddedEvent(
            response_id="resp_1",
            sequence_number=1,
            item_index=0,
            item=ReasoningItem(id="rs_1", status=ItemStatus.IN_PROGRESS, parts=()),
        ),
        ContentPartAddedEvent(
            response_id="resp_1",
            sequence_number=2,
            item_id="rs_1",
            item_index=0,
            content_index=0,
            part=ReasoningSummaryPart(text=""),
        ),
        ContentPartDoneEvent(
            response_id="resp_1",
            sequence_number=3,
            item_id="rs_1",
            item_index=0,
            content_index=0,
            part=ReasoningSummaryPart(text="thought"),
        ),
        OutputItemAddedEvent(
            response_id="resp_1",
            sequence_number=4,
            item_index=1,
            item=AssistantMessageItem(id="msg_1", status=ItemStatus.IN_PROGRESS, content=()),
        ),
        ContentPartAddedEvent(
            response_id="resp_1",
            sequence_number=5,
            item_id="msg_1",
            item_index=1,
            content_index=0,
            part=RefusalPart(refusal=""),
        ),
        ContentPartDoneEvent(
            response_id="resp_1",
            sequence_number=6,
            item_id="msg_1",
            item_index=1,
            content_index=0,
            part=RefusalPart(refusal="No"),
        ),
        ContentPartAddedEvent(
            response_id="resp_1",
            sequence_number=7,
            item_id="msg_1",
            item_index=1,
            content_index=1,
            part=OutputTextPart(text=""),
        ),
        ContentPartDoneEvent(
            response_id="resp_1",
            sequence_number=8,
            item_id="msg_1",
            item_index=1,
            content_index=1,
            part=OutputTextPart(text="Hi"),
        ),
    )

    async def gen() -> AsyncIterator:
        for event in events:
            yield event

    public = [event async for event in from_output_event_stream(gen())]
    added = [event for event in public if event.type == "response.content_part.added"]
    done = [event for event in public if event.type == "response.content_part.done"]

    assert [event.part.type for event in added] == ["summary_text", "refusal", "output_text"]
    assert [event.part.type for event in done] == ["summary_text", "refusal", "output_text"]
    assert done[0].part.text == "thought"
    assert done[1].part.refusal == "No"
    assert done[2].part.text == "Hi"
