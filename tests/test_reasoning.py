import pytest

from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIReasoningItem,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAIResponsesStreamOutput,
    OpenAIResponsesStreamText,
    _OpenAIStreamEvent,
)
from molto.utils.test import (
    CASSETTE_DIR,
    STREAM_PARAMS,
    assert_stream_text_events,
    assert_usage,
    extract_cassette_response,
    extract_stream_cassette_response,
    make_handler,
    make_molto_client,
    prepend_model_prefix,
    read_cassette_request,
)

CASSETTE_PATH = CASSETTE_DIR / "openai" / "responses" / "reasoning.yaml"


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_reasoning(stream: bool):
    cassette_req = read_cassette_request(CASSETTE_PATH, stream=stream)

    req_body = dict(
        model=cassette_req["model"],
        input=cassette_req["input"],
        reasoning=cassette_req["reasoning"],
        include=cassette_req.get("include"),
        stream=stream,
    )

    handler = make_handler(CASSETTE_PATH, req_body, ["model", "input"])

    cassette_response = extract_cassette_response(CASSETTE_PATH, stream=stream)

    assert cassette_response["reasoning"]["effort"] == "high"
    assert cassette_response["reasoning"]["summary"] == "detailed"

    assert len(cassette_response["output"]) == 2

    cassette_reasoning = cassette_response["output"][0]
    assert cassette_reasoning["type"] == "reasoning"
    assert cassette_reasoning["encrypted_content"] is not None
    assert len(cassette_reasoning["encrypted_content"]) > 0
    assert len(cassette_reasoning["summary"]) == 1
    assert cassette_reasoning["summary"][0]["type"] == "summary_text"
    assert len(cassette_reasoning["summary"][0]["text"]) > 0

    cassette_message = cassette_response["output"][1]
    assert cassette_message["type"] == "message"
    assert cassette_message["role"] == "assistant"
    assert len(cassette_message["content"]) == 1
    assert len(cassette_message["content"][0]["text"]) > 0

    assert cassette_response["usage"]["output_tokens_details"]["reasoning_tokens"] > 0

    stream_resp = extract_stream_cassette_response(CASSETTE_PATH)
    expected_molto_response_id = stream_resp["id"]
    expected_molto_rs_id = stream_resp["output"][0]["id"]
    expected_molto_msg_id = stream_resp["output"][1]["id"]

    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert "response.reasoning.delta" in {e.type for e in events}
        assert "response.reasoning.done" in {e.type for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.completed"

        reasoning_item_added = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamOutput)
            and e.type == "response.output_item.added"
            and isinstance(e.item, OpenAIReasoningItem)
        ]
        assert len(reasoning_item_added) == 1
        reasoning_item_done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamOutput)
            and e.type == "response.output_item.done"
            and isinstance(e.item, OpenAIReasoningItem)
        ]
        assert len(reasoning_item_done) == 1
        reasoning_id = reasoning_item_added[0].item.id
        assert reasoning_id is not None
        assert reasoning_id.startswith("rs_")
        assert reasoning_id == reasoning_item_done[0].item.id

        reasoning_deltas = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamText) and e.type == "response.reasoning.delta"
        ]
        assert len(reasoning_deltas) >= 1
        for delta in reasoning_deltas:
            assert delta.item_id == reasoning_id
        assert any(d.delta is not None and len(d.delta) > 0 for d in reasoning_deltas)

        reasoning_done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamText) and e.type == "response.reasoning.done"
        ]
        assert len(reasoning_done) == 1
        assert reasoning_done[0].item_id == reasoning_id
        assert reasoning_done[0].text is not None
        assert len(reasoning_done[0].text) > 0

        assert_stream_text_events(events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == expected_molto_response_id

    assert response.status == "completed"
    assert response.reasoning is not None
    assert response.reasoning.effort == "high"
    assert response.reasoning.summary == "detailed"

    assert_usage(response.usage, expect_reasoning=True)

    assert len(response.output) == 2

    molto_reasoning = response.output[0]
    assert isinstance(molto_reasoning, OpenAIReasoningItem)
    assert molto_reasoning.type == "reasoning"
    assert molto_reasoning.id == expected_molto_rs_id
    assert len(molto_reasoning.content) >= 1
    assert molto_reasoning.content[0].type == "reasoning_text"
    assert len(molto_reasoning.content[0].text) > 0

    msg_molto = response.output[1]

    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "completed"
    assert msg_molto.id == expected_molto_msg_id
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0
