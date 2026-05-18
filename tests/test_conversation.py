import pytest

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
    assert_stream_text_events,
    extract_cassette_response,
    extract_output_ids,
    extract_stream_cassette_response,
    make_handler,
    make_molto_client,
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

    cassette_response = extract_cassette_response(CASSETTE_PATH, stream=stream)

    assert cassette_response["conversation"]["id"] == conversation_id

    stream_resp = extract_stream_cassette_response(CASSETTE_PATH)
    expected_molto_response_id = stream_resp["id"]
    stream_ids = extract_output_ids(stream_resp, "message")
    expected_molto_msg_id = stream_ids["message"]

    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.completed"

        assert_stream_text_events(events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == expected_molto_response_id
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
    assert msg_molto.id == expected_molto_msg_id
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert "scarecrow" in text_molto.text

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
