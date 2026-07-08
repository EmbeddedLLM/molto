import pytest
from openai.types.responses import Response, ResponseIncompleteEvent, ResponseOutputMessage

from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAIResponsesStreamText,
    _OpenAIStreamEvent,
)
from molto.utils.test import (
    CASSETTE_DIR,
    STREAM_PARAMS,
    assert_output_item_order,
    assert_stream_events_match,
    assert_stream_text_events,
    assert_usage,
    make_handler,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
)

EXPECTED_SETTINGS = {
    "max_output_tokens": 20,
    "instructions": "You are a financial advisor for the user.",
}


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_incomplete_response(stream: bool):

    req_body = dict(
        model="gpt-4.1-nano",
        input=[
            {"role": "system", "content": "You are a financial advisor"},
            {"role": "user", "content": "Who are you?"},
        ],
        instructions=EXPECTED_SETTINGS["instructions"],
        max_output_tokens=EXPECTED_SETTINGS["max_output_tokens"],
        stream=stream,
    )

    cassette_path = CASSETTE_DIR / "openai" / "responses" / "incomplete.yaml"
    handler = make_handler(
        cassette_path,
        req_body,
        ["model", "input", "instructions", "max_output_tokens"],
    )

    # ── OpenAI SDK (parametrized stream) ───────────────────────────────
    sdk_client = make_openai_client(handler, load_stream_cassette=stream)
    sdk_response = await sdk_client.responses.create(**req_body)
    if stream:
        sdk_events = [e async for e in sdk_response]
        sdk_event_types = [e.type for e in sdk_events]
        assert isinstance(sdk_events[-1], ResponseIncompleteEvent)
        assert sdk_event_types[-1] == "response.incomplete"

        assert_output_item_order(sdk_events)

        sdk_response = sdk_events[-1].response
    assert isinstance(sdk_response, Response)
    assert sdk_response.status == "incomplete"
    assert sdk_response.incomplete_details is not None
    assert sdk_response.incomplete_details.reason == "max_output_tokens"
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(sdk_response, key)
        assert actual == expected, f"OpenAI SDK response.{key}={actual!r}, expected {expected!r}"

    sdk_msg = next((o for o in sdk_response.output if isinstance(o, ResponseOutputMessage)), None)
    assert sdk_msg is not None
    assert sdk_msg.status == "incomplete"

    # ── Stream SDK reference (matches Molto's internal streaming) ──────
    ref_client = make_openai_client(handler, load_stream_cassette=True)
    ref_events = [
        e async for e in await ref_client.responses.create(**{**req_body, "stream": True})
    ]
    assert isinstance(ref_events[-1], ResponseIncompleteEvent)
    sdk_ref = ref_events[-1].response
    sdk_ref_msg = next(o for o in sdk_ref.output if isinstance(o, ResponseOutputMessage))

    # ── Molto client ───────────────────────────────────────────────────
    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        event_types = [e.type for e in events]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.output_text.delta" in event_types
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert event_types[-1] == "response.incomplete"

        assert_output_item_order(events)
        assert_stream_text_events(events)

        assert_stream_events_match(ref_events, events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == sdk_ref.id
    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(response, key)
        assert actual == expected, f"Molto response.{key}={actual!r}, expected {expected!r}"

    for msg_molto in response.output:
        assert isinstance(msg_molto, OpenAIOutputItem)
        assert msg_molto.type == "message"
        assert msg_molto.role == "assistant"
        assert msg_molto.status == "incomplete"
        assert msg_molto.id == sdk_ref_msg.id
        assert len(msg_molto.content) >= 1

        text_molto = msg_molto.content[0]
        assert isinstance(text_molto, OpenAIOutputTextContent)
        assert text_molto.type == "output_text"
        assert len(text_molto.text) > 0
        assert "financial advisor" in text_molto.text

    assert_usage(response.usage, output_tokens_exact=20)
