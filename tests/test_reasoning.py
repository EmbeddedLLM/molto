import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_text import ResponseOutputText

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

    # ── (1) Verify the YAML cassette response carries reasoning content ──
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

    # ── (2) OpenAI SDK client ───────────────────────────────────────────
    client = make_openai_client(handler, load_stream_cassette=stream)
    response = await client.responses.create(**req_body)
    if stream:
        events = [r async for r in response]
        assert ResponseReasoningSummaryTextDeltaEvent in {type(e) for e in events}
        assert "response.reasoning_summary_text.delta" in {e.type for e in events}
        assert ResponseReasoningSummaryTextDoneEvent in {type(e) for e in events}
        assert "response.reasoning_summary_text.done" in {e.type for e in events}
        assert ResponseTextDeltaEvent in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert isinstance(events[-1], ResponseCompletedEvent)
        response = events[-1].response
    assert isinstance(response, Response)

    assert response.status == "completed"
    assert response.reasoning is not None
    assert response.reasoning.effort == "high"
    assert response.reasoning.summary == "detailed"

    assert len(response.output) == 2

    sdk_reasoning = response.output[0]
    assert isinstance(sdk_reasoning, ResponseReasoningItem)
    assert sdk_reasoning.type == "reasoning"
    assert sdk_reasoning.id is not None
    assert sdk_reasoning.id.startswith("rs_")
    assert sdk_reasoning.encrypted_content is not None
    assert len(sdk_reasoning.encrypted_content) > 0
    assert len(sdk_reasoning.summary) == 1
    assert sdk_reasoning.summary[0].type == "summary_text"
    assert len(sdk_reasoning.summary[0].text) > 0

    sdk_message = response.output[1]
    assert isinstance(sdk_message, ResponseOutputMessage)
    assert sdk_message.role == "assistant"
    assert sdk_message.status == "completed"
    assert len(sdk_message.content) == 1
    sdk_text = sdk_message.content[0]
    assert isinstance(sdk_text, ResponseOutputText)
    assert len(sdk_text.text) > 0

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.usage.output_tokens_details is not None
    assert response.usage.output_tokens_details.reasoning_tokens > 0

    # ── (3) Molto client ───────────────────────────────────────────────
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

        text_deltas = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamText) and e.type == "response.output_text.delta"
        ]
        assert len(text_deltas) >= 1
        for delta in text_deltas:
            assert delta.item_id == added_id

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.status == "completed"
    assert response.reasoning is not None
    assert response.reasoning.effort == "high"
    assert response.reasoning.summary == "detailed"

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.usage.output_tokens_details is not None
    assert response.usage.output_tokens_details.reasoning_tokens > 0

    assert len(response.output) == 2

    molto_reasoning = response.output[0]
    assert isinstance(molto_reasoning, OpenAIReasoningItem)
    assert molto_reasoning.type == "reasoning"
    assert molto_reasoning.id is not None
    assert molto_reasoning.id.startswith("rs_")
    assert len(molto_reasoning.content) >= 1
    assert molto_reasoning.content[0].type == "reasoning_text"
    assert len(molto_reasoning.content[0].text) > 0

    msg_molto = response.output[1]

    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "completed"
    assert msg_molto.id is not None
    assert msg_molto.id.startswith("msg_")
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0
