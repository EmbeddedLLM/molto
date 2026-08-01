import pytest
from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCodeInterpreterToolCall,
)

from molto.types.openai.responses import OpenAICodeToolCall
from tests.support import (
    STREAM_PARAMS,
    assert_events_before_terminal,
    run_responses_parity,
)

REQUEST_BODY = {
    "model": "gpt-5-nano",
    "input": [
        {
            "role": "user",
            "content": "Use the code interpreter to run Python that prints 21 * 2, then report the result.",
        }
    ],
    "include": ["code_interpreter_call.outputs"],
    "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
    "tool_choice": "required",
}


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_code_interpreter_output_round_trip_through_sdk(
    stream: bool, cassette_manager
) -> None:
    """Preserve populated code-interpreter output and every streamed phase one-to-one."""
    req_body = {**REQUEST_BODY, "stream": stream}

    parity = await run_responses_parity(cassette_manager, req_body)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    if stream:
        assert {
            ResponseCodeInterpreterCallInProgressEvent,
            ResponseCodeInterpreterCallInterpretingEvent,
            ResponseCodeInterpreterCallCodeDoneEvent,
            ResponseCodeInterpreterCallCompletedEvent,
        } <= {type(event) for event in parity.sdk_events}
        assert_events_before_terminal(
            parity.sdk_events, "response.code_interpreter_call.completed"
        )

    sdk_call = next(
        item for item in sdk_response.output if isinstance(item, ResponseCodeInterpreterToolCall)
    )
    assert sdk_call.status == "completed"
    assert sdk_call.code
    assert sdk_call.outputs

    if stream:
        assert {
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call_code.done",
            "response.code_interpreter_call.completed",
        } <= {event.type for event in parity.molto_events}
        assert_events_before_terminal(
            parity.molto_events, "response.code_interpreter_call.completed"
        )

    molto_call = next(item for item in response.output if isinstance(item, OpenAICodeToolCall))
    assert molto_call.id == sdk_call.id
    assert molto_call.status == sdk_call.status
    assert molto_call.code == sdk_call.code
    assert [
        output.model_dump(mode="python", exclude_unset=True) for output in molto_call.outputs
    ] == [output.model_dump(mode="python", exclude_unset=True) for output in sdk_call.outputs]
