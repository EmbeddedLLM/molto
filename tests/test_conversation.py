import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_text import ResponseOutputText

from molto.types.openai.responses import (
    OpenAIConversation,
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAIResponsesStreamOutput,
    OpenAIResponsesStreamText,
    _OpenAIStreamEvent,
)
from molto.utils.io import json_loads
from molto.utils.test import (
    CASSETTE_DIR,
    STREAM_PARAMS,
    make_handler,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    read_cassette,
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

    # ── (1) Verify the YAML cassette response carries the conversation id ─
    cassette_data = read_cassette(CASSETTE_PATH, stream=stream)
    if stream:
        completed_frames = [
            line
            for line in cassette_data
            if line.startswith("data: ") and '"type":"response.completed"' in line
        ]
        assert len(completed_frames) == 1
        cassette_response = json_loads(completed_frames[0][len("data: ") :])["response"]
    else:
        cassette_response = cassette_data

    assert cassette_response["conversation"]["id"] == conversation_id

    expected_response_id = cassette_response["id"]
    cassette_rs = [o for o in cassette_response["output"] if o["type"] == "reasoning"]
    assert len(cassette_rs) >= 1
    expected_rs_id = cassette_rs[0]["id"]
    cassette_msg = [o for o in cassette_response["output"] if o["type"] == "message"]
    assert len(cassette_msg) == 1
    expected_msg_id = cassette_msg[0]["id"]

    stream_cassette_data = read_cassette(CASSETTE_PATH, stream=True)
    stream_completed = [
        line
        for line in stream_cassette_data
        if line.startswith("data: ") and '"type":"response.completed"' in line
    ]
    assert len(stream_completed) == 1
    stream_resp = json_loads(stream_completed[0][len("data: ") :])["response"]
    expected_molto_response_id = stream_resp["id"]
    stream_msg = [o for o in stream_resp["output"] if o["type"] == "message"]
    expected_molto_msg_id = stream_msg[0]["id"]

    # ── OpenAI SDK client ──────────────────────────────────────────────
    client = make_openai_client(handler, load_stream_cassette=stream)
    response = await client.responses.create(**req_body)
    if stream:
        events = [r async for r in response]
        assert ResponseTextDeltaEvent in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert isinstance(events[-1], ResponseCompletedEvent)
        response = events[-1].response
    assert isinstance(response, Response)

    assert response.id == expected_response_id
    assert response.status == "completed"
    assert response.conversation is not None
    assert response.conversation.id == conversation_id

    # Output: reasoning + message (OpenAI SDK includes both)
    assert len(response.output) == 2
    assert response.output[0].id == expected_rs_id
    msg = response.output[1]
    assert isinstance(msg, ResponseOutputMessage)
    assert msg.id == expected_msg_id
    assert msg.role == "assistant"
    assert msg.status == "completed"
    assert len(msg.content) == 1
    text_content = msg.content[0]
    assert isinstance(text_content, ResponseOutputText)
    assert "scarecrow" in text_content.text

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0

    # ── Molto client ───────────────────────────────────────────────────
    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.completed"

        msg_item_added = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamOutput)
            and e.type == "response.output_item.added"
            and hasattr(e.item, "role")
            and e.item.role == "assistant"
        ]
        msg_item_done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamOutput)
            and e.type == "response.output_item.done"
            and hasattr(e.item, "role")
            and e.item.role == "assistant"
        ]
        assert len(msg_item_added) == 1
        assert len(msg_item_done) == 1
        added_id = msg_item_added[0].item.id
        done_id = msg_item_done[0].item.id
        assert added_id is not None
        assert added_id.startswith("msg_")
        assert added_id == done_id

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == expected_molto_response_id
    assert response.status == "completed"
    assert response.conversation is not None
    assert isinstance(response.conversation, OpenAIConversation)
    assert response.conversation.id == conversation_id

    # Molto strips reasoning from output; only message remains
    assert len(response.output) >= 1
    msg_molto = response.output[-1]
    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "completed"
    assert msg_molto.id == expected_molto_msg_id
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert "scarecrow" in text_molto.text

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
