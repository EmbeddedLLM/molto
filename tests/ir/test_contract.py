from molto.providers.openai.usage import usage_from_openai, usage_to_openai
from molto.types.ir.emitter import OutputEventEmitter
from molto.types.ir.events import OutputEventSequence, ResponseCompletedEvent
from molto.types.ir.output_items import AssistantMessageItem
from molto.types.ir.portable import (
    ItemStatus,
    OutputTextPart,
    ResponseMetadata,
    Usage,
)


def test_openai_usage_round_trips_common_counters() -> None:
    payload = {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 1},
        "output_tokens_details": {"reasoning_tokens": 1},
    }
    usage = usage_from_openai(payload, "responses")

    assert usage == Usage(
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
        cached_input_tokens=1,
        cache_write_input_tokens=0,
        reasoning_output_tokens=1,
    )
    response_usage = usage_to_openai(usage, "responses")
    assert response_usage["total_tokens"] == 5
    assert response_usage["input_tokens_details"]["cache_write_tokens"] == 0


def test_output_event_sequence_reconstructs_terminal_snapshot() -> None:
    metadata = ResponseMetadata(model="gpt-test", object="response")
    emitter = OutputEventEmitter("resp_1", metadata=metadata)
    item = AssistantMessageItem(
        id="msg_1",
        status=ItemStatus.IN_PROGRESS,
        content=(OutputTextPart(text="hello"),),
    )
    events = [emitter.started(), emitter.item_added(0, item)]
    events.extend(emitter.replay_part("msg_1", 0, 0, item.content[0], "hello"))
    events.append(emitter.item_done(0, item.model_copy(update={"status": ItemStatus.COMPLETED})))
    events.append(emitter.completed(Usage()))

    sequence = OutputEventSequence(events=tuple(events))

    assert isinstance(sequence.terminal, ResponseCompletedEvent)
    assert sequence.done_items[0].id == "msg_1"
