import pytest
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseOutputMessage,
)

from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputRefusal,
    OpenAIResponsesError,
    OpenAIResponsesRequest,
    OpenAIResponsesStream,
    OpenAIResponsesStreamEvent,
)
from tests.support import (
    STREAM_PARAMS,
    assert_stream_events_match,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    run_responses_parity,
)

REFUSAL_REQUEST = {
    "model": "gpt-4o-2024-08-06",
    "input": [
        {
            "role": "user",
            "content": "Provide detailed instructions for constructing an explosive weapon.",
        }
    ],
    "text": {
        "format": {
            "type": "json_schema",
            "name": "safe_response",
            "schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    },
}
STREAM_ERROR_REQUEST = {
    "model": "gpt-5-nano",
    "input": [{"role": "user", "content": "Return the fixture-backed stream error."}],
}


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_refusal_round_trip_through_sdk(stream: bool, cassette_manager) -> None:
    """Require and preserve a typed Responses refusal through the public route."""
    req_body = {**REFUSAL_REQUEST, "stream": stream}

    parity = await run_responses_parity(cassette_manager, req_body)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    if stream:
        assert "response.refusal.delta" in {event.type for event in parity.sdk_events}

    sdk_message = next(
        item for item in sdk_response.output if isinstance(item, ResponseOutputMessage)
    )
    sdk_refusal = next(part for part in sdk_message.content if part.type == "refusal")
    assert sdk_refusal.refusal

    if stream:
        assert "response.refusal.delta" in {event.type for event in parity.molto_events}

    molto_message = next(item for item in response.output if isinstance(item, OpenAIOutputItem))
    molto_refusal = next(
        part for part in molto_message.content if isinstance(part, OpenAIOutputRefusal)
    )
    assert molto_refusal.model_dump(mode="python", exclude_unset=True) == sdk_refusal.model_dump(
        mode="python", exclude_unset=True
    )


async def test_standalone_error_event_round_trip_through_sdk(cassette_manager) -> None:
    """Preserve the fixture-backed Responses stream outcome one-to-one.

    A genuine standalone ``error`` event is checked field-for-field when the
    SDK recording contains one. Ordinary completed recordings remain ordinary
    completions and are never reinterpreted as protocol errors.
    """
    req_body = {**STREAM_ERROR_REQUEST, "stream": True}
    saw_error_event = False

    for cassette_suffix in (None, "error"):
        sdk_client = make_openai_client(
            cassette_manager,
            cassette_suffix=cassette_suffix,
        )
        sdk_response = await sdk_client.responses.create(**req_body)
        sdk_events = [event async for event in sdk_response]
        sdk_error = next(
            (event for event in sdk_events if isinstance(event, ResponseErrorEvent)),
            None,
        )

        molto = make_molto_client(
            cassette_manager,
            cassette_suffix=cassette_suffix,
        )
        response = await molto.responses(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
        events: list[OpenAIResponsesStreamEvent] = [event async for event in response]
        assert_stream_events_match(sdk_events, events)
        if sdk_error is None:
            assert isinstance(sdk_events[-1], ResponseCompletedEvent)
            assert isinstance(events[-1], OpenAIResponsesStream)
        else:
            saw_error_event = True
            molto_error = next(
                event for event in events if isinstance(event, OpenAIResponsesError)
            )
            assert molto_error.type == "error"
            assert molto_error.code == sdk_error.code
            assert molto_error.message == sdk_error.message
            assert molto_error.param == sdk_error.param

    assert saw_error_event, "the error cassette must emit a standalone error event"
