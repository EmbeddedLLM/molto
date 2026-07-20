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
    STREAM_PARAMS,
    assert_output_item_order,
    assert_stream_events_match,
    assert_stream_text_events,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    stream_reference_cassette_suffix,
)

CONVERSATION_SETUP_ITEMS = [
    {"role": "user", "content": "Tell me a short joke about a scarecrow."},
    {
        "role": "assistant",
        "content": (
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ),
    },
]


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_conversation(stream: bool, cassette_manager):
    """Create/replay a provider conversation setup before testing response replay.

    Example:
        # cassette_suffix="setup" returns a conversation id used by the stream/non-stream request.
    """
    setup_client = make_openai_client(
        cassette_manager,
        cassette_suffix="setup",
        cassette_endpoint="conversations",
        expected_request_body={"items": CONVERSATION_SETUP_ITEMS},
        assert_attrs=["items"],
    )
    conversation = await setup_client.conversations.create(items=CONVERSATION_SETUP_ITEMS)
    conversation_id = conversation.id

    req_body = dict(
        model="gpt-5-nano",
        input=[{"role": "user", "content": "what joke was previously mentioned?"}],
        conversation=conversation_id,
        include=["reasoning.encrypted_content"],
        stream=stream,
    )

    assert_attrs = ["model", "input", "conversation"]

    # -- OpenAI SDK (parametrized stream) --------------------------------
    sdk_client = make_openai_client(
        cassette_manager,
        expected_request_body=req_body,
        assert_attrs=assert_attrs,
    )
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
    assert sdk_msg.role == "assistant"
    assert len(sdk_msg.content) >= 1
    assert "scarecrow" in sdk_msg.content[0].text

    stream_ref_suffix = stream_reference_cassette_suffix(stream)

    # -- Stream SDK reference (matches Molto's internal streaming) -------
    stream_req_body = {**req_body, "stream": True}
    ref_client = make_openai_client(
        cassette_manager,
        cassette_suffix=stream_ref_suffix,
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    ref_events = [e async for e in await ref_client.responses.create(**stream_req_body)]
    assert isinstance(ref_events[-1], ResponseCompletedEvent)
    sdk_ref = ref_events[-1].response
    sdk_ref_msg = next(o for o in sdk_ref.output if isinstance(o, ResponseOutputMessage))

    # -- Molto client ----------------------------------------------------
    client = make_molto_client(
        cassette_manager,
        cassette_suffix=stream_ref_suffix,
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
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
