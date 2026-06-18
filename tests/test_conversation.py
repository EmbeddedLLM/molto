import pytest
from openai.types.responses import Response, ResponseCompletedEvent, ResponseOutputMessage

from molto.types.openai.responses import (
    OpenAIConversation,
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    _OpenAIStreamEvent,
)
from molto.utils.test import (
    CASSETTE_DIR,
    STREAM_PARAMS,
    assert_output_item_order,
    assert_stream_events_match,
    assert_stream_text_events,
    make_handler,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    read_cassette_request,
)

CASSETTE_PATH = CASSETTE_DIR / "openai" / "responses" / "conversation.yaml"


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_conversation(stream: bool):
    cassette_req = read_cassette_request(CASSETTE_PATH, stream=stream)
    conversation_id = cassette_req["conversation"]

    req_body = dict(
        model=cassette_req["model"],
        input=cassette_req["input"],
        conversation=conversation_id,
        include=cassette_req.get("include"),
        stream=stream,
    )

    handler = make_handler(CASSETTE_PATH, req_body, ["model", "input", "conversation"])

    # ── OpenAI SDK (parametrized stream) ───────────────────────────────
    sdk_client = make_openai_client(handler, load_stream_cassette=stream)
    sdk_response = await sdk_client.responses.create(**req_body)
    if stream:
        sdk_events = [e async for e in sdk_response]
        sdk_event_types = [e.type for e in sdk_events]
        assert isinstance(sdk_events[-1], ResponseCompletedEvent)
        assert sdk_event_types[-1] == "response.completed"

        assert_output_item_order(sdk_events)

        sdk_response = sdk_events[-1].response
    assert isinstance(sdk_response, Response)
    assert sdk_response.status == "completed"
    assert sdk_response.conversation is not None
    assert sdk_response.conversation.id == conversation_id

    sdk_msg = next(o for o in sdk_response.output if isinstance(o, ResponseOutputMessage))
    assert isinstance(sdk_msg, ResponseOutputMessage)
    assert sdk_msg.role == "assistant"
    assert len(sdk_msg.content) >= 1
    assert "scarecrow" in sdk_msg.content[0].text

    # ── Stream SDK reference (matches Molto's internal streaming) ──────
    ref_client = make_openai_client(handler, load_stream_cassette=True)
    ref_events = [e async for e in await ref_client.responses.create(**{**req_body, "stream": True})]
    assert isinstance(ref_events[-1], ResponseCompletedEvent)
    sdk_ref = ref_events[-1].response
    sdk_ref_msg = next(o for o in sdk_ref.output if isinstance(o, ResponseOutputMessage))

    # ── Molto client ───────────────────────────────────────────────────
    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        event_types = [e.type for e in events]
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert event_types[-1] == "response.completed"

        assert_output_item_order(events)

        assert_stream_events_match(ref_events, events)
        assert_stream_text_events(events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == sdk_ref.id
    assert response.status == "completed"
    assert response.conversation is not None
    assert isinstance(response.conversation, OpenAIConversation)
    assert response.conversation.id == conversation_id

    assert len(response.output) >= 1
    msg_molto = response.output[-1]
    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "completed"
    assert msg_molto.id == sdk_ref_msg.id
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert "scarecrow" in text_molto.text

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
