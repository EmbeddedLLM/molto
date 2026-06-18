import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
)

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
    assert_output_item_order,
    assert_stream_events_match,
    assert_stream_text_events,
    assert_usage,
    make_handler,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    read_cassette_request,
)

CASSETTE_NAMES = ["reasoning_no_summary", "reasoning_with_summary"]


@pytest.mark.parametrize("cassette_name", CASSETTE_NAMES)
@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_reasoning(stream: bool, cassette_name: str):
    cassette_path = CASSETTE_DIR / "openai" / "responses" / f"{cassette_name}.yaml"
    cassette_req = read_cassette_request(cassette_path, stream=stream)

    req_body = dict(
        model=cassette_req["model"],
        input=cassette_req["input"],
        reasoning=cassette_req["reasoning"],
        include=cassette_req.get("include"),
        stream=stream,
    )

    handler = make_handler(cassette_path, req_body, ["model", "input"])

    # ── OpenAI SDK (parametrized stream) ───────────────────────────────
    sdk_client = make_openai_client(handler, load_stream_cassette=stream)
    sdk_response = await sdk_client.responses.create(**req_body)
    if stream:
        sdk_events = [e async for e in sdk_response]
        sdk_event_types = [e.type for e in sdk_events]
        assert isinstance(sdk_events[-1], ResponseCompletedEvent)
        assert sdk_event_types[-1] == "response.completed"
        # SDK uses "reasoning_summary_text" naming for reasoning deltas
        assert "response.reasoning_summary_text.delta" in sdk_event_types
        assert "response.reasoning_summary_text.done" in sdk_event_types
        assert "response.output_text.delta" in sdk_event_types

        assert_output_item_order(sdk_events)

        # reasoning_summary_text.delta before reasoning_summary_text.done
        rs_delta_indices = [
            i
            for i, e in enumerate(sdk_events)
            if e.type == "response.reasoning_summary_text.delta"
        ]
        rs_done_indices = [
            i for i, e in enumerate(sdk_events) if e.type == "response.reasoning_summary_text.done"
        ]
        assert len(rs_done_indices) >= 1
        assert all(d < rs_done_indices[0] for d in rs_delta_indices)

        sdk_response = sdk_events[-1].response
    assert isinstance(sdk_response, Response)
    assert sdk_response.status == "completed"
    assert sdk_response.reasoning.effort == "high"
    assert sdk_response.reasoning.summary == "detailed"

    sdk_reasoning = next(o for o in sdk_response.output if isinstance(o, ResponseReasoningItem))
    assert isinstance(sdk_reasoning, ResponseReasoningItem)
    assert sdk_reasoning.type == "reasoning"
    assert sdk_reasoning.encrypted_content is not None
    assert len(sdk_reasoning.encrypted_content) > 0
    assert len(sdk_reasoning.summary) >= 1
    assert sdk_reasoning.summary[0].type == "summary_text"
    assert len(sdk_reasoning.summary[0].text) > 0

    sdk_msg = next(o for o in sdk_response.output if isinstance(o, ResponseOutputMessage))
    assert isinstance(sdk_msg, ResponseOutputMessage)
    assert sdk_msg.role == "assistant"
    assert len(sdk_msg.content) >= 1
    assert len(sdk_msg.content[0].text) > 0

    assert sdk_response.usage.output_tokens_details.reasoning_tokens > 0

    # ── Stream SDK reference (matches Molto's internal streaming) ──────
    ref_client = make_openai_client(handler, load_stream_cassette=True)
    ref_events = [
        e async for e in await ref_client.responses.create(**{**req_body, "stream": True})
    ]
    assert isinstance(ref_events[-1], ResponseCompletedEvent)
    sdk_ref = ref_events[-1].response
    sdk_ref_rs = next(o for o in sdk_ref.output if isinstance(o, ResponseReasoningItem))
    sdk_ref_msg = next(o for o in sdk_ref.output if isinstance(o, ResponseOutputMessage))

    # ── Molto client ───────────────────────────────────────────────────
    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        event_types = [e.type for e in events]
        assert "response.reasoning.delta" in event_types
        assert "response.reasoning.done" in event_types
        assert "response.output_text.delta" in event_types
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert event_types[-1] == "response.completed"

        assert_output_item_order(events)

        # reasoning.delta before reasoning.done
        reasoning_delta_indices = [
            i for i, e in enumerate(events) if e.type == "response.reasoning.delta"
        ]
        reasoning_done_indices = [
            i for i, e in enumerate(events) if e.type == "response.reasoning.done"
        ]
        assert len(reasoning_done_indices) >= 1
        assert all(d < reasoning_done_indices[0] for d in reasoning_delta_indices)

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

        assert_stream_events_match(ref_events, events)
        assert_stream_text_events(events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == sdk_ref.id
    assert response.status == "completed"
    assert response.reasoning is not None
    assert response.reasoning.effort == "high"
    assert response.reasoning.summary == "detailed"

    assert_usage(response.usage, expect_reasoning=True)

    assert len(response.output) >= 2

    molto_reasoning = next(o for o in response.output if isinstance(o, OpenAIReasoningItem))
    assert isinstance(molto_reasoning, OpenAIReasoningItem)
    assert molto_reasoning.type == "reasoning"
    assert molto_reasoning.id == sdk_ref_rs.id
    assert len(molto_reasoning.content) >= 1
    assert molto_reasoning.content[0].type == "reasoning_text"
    assert len(molto_reasoning.content[0].text) > 0

    msg_molto = next(o for o in response.output if isinstance(o, OpenAIOutputItem))
    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "completed"
    assert msg_molto.id == sdk_ref_msg.id
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0
