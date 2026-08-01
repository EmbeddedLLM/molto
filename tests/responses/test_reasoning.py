import pytest
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseReasoningItem,
)

from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIReasoningItem,
    OpenAIResponsesStreamOutput,
    OpenAIResponsesStreamText,
)
from tests.support import (
    STREAM_PARAMS,
    assert_stream_text_events,
    assert_usage,
    run_responses_parity,
)

REASONING_CASES = {
    "reasoning_no_summary": {
        "reasoning": {"effort": "high"},
        "include": ["reasoning.encrypted_content"],
        "expect_summary": False,
    },
    "reasoning_with_summary": {
        "reasoning": {"effort": "high", "summary": "detailed"},
        "include": ["reasoning.encrypted_content"],
        "expect_summary": True,
    },
}
CASSETTE_NAMES = list(REASONING_CASES)


@pytest.mark.parametrize("cassette_name", CASSETTE_NAMES)
@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_reasoning_output_round_trip_through_sdk(
    stream: bool, cassette_name: str, cassette_manager
):
    """Compare Molto reasoning output against SDK-parsed Responses cassettes.

    Example:
        # stream=True and cassette_name="reasoning_with_summary"
        # returns stream events ending with "response.completed".
    """
    case = REASONING_CASES[cassette_name]
    reasoning = case["reasoning"]
    expect_summary = case["expect_summary"]
    req_body = dict(
        model="gpt-5.1",
        input=[{"role": "user", "content": "How many Rs in strawberry?"}],
        reasoning=reasoning,
        include=case["include"],
        stream=stream,
    )

    parity = await run_responses_parity(cassette_manager, req_body)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    # -- OpenAI SDK (parametrized stream) --------------------------------
    if stream:
        sdk_events = parity.sdk_events
        sdk_event_types = [e.type for e in sdk_events]
        assert "response.output_text.delta" in sdk_event_types
        if expect_summary:
            # SDK uses "reasoning_summary_text" naming for reasoning deltas.
            assert "response.reasoning_summary_text.delta" in sdk_event_types
            assert "response.reasoning_summary_text.done" in sdk_event_types
        else:
            assert "response.reasoning_summary_text.delta" not in sdk_event_types
            assert "response.reasoning_summary_text.done" not in sdk_event_types

        if expect_summary:
            rs_delta_indices = [
                i
                for i, e in enumerate(sdk_events)
                if e.type == "response.reasoning_summary_text.delta"
            ]
            rs_done_indices = [
                i
                for i, e in enumerate(sdk_events)
                if e.type == "response.reasoning_summary_text.done"
            ]
            assert len(rs_done_indices) >= 1
            assert all(d < rs_done_indices[0] for d in rs_delta_indices)

    assert sdk_response.status == "completed"
    assert sdk_response.reasoning.effort == "high"
    if expect_summary:
        assert sdk_response.reasoning.summary == reasoning["summary"]

    sdk_reasoning = next(o for o in sdk_response.output if isinstance(o, ResponseReasoningItem))
    assert sdk_reasoning.type == "reasoning"
    assert sdk_reasoning.encrypted_content is not None
    assert len(sdk_reasoning.encrypted_content) > 0
    if expect_summary:
        assert len(sdk_reasoning.summary) >= 1
        assert sdk_reasoning.summary[0].type == "summary_text"
        assert len(sdk_reasoning.summary[0].text) > 0
    else:
        assert sdk_reasoning.summary == []

    sdk_msg = next(o for o in sdk_response.output if isinstance(o, ResponseOutputMessage))
    assert sdk_msg.role == "assistant"
    assert len(sdk_msg.content) >= 1
    assert len(sdk_msg.content[0].text) > 0

    assert sdk_response.usage.output_tokens_details.reasoning_tokens > 0

    # -- Molto client ----------------------------------------------------
    if stream:
        events = parity.molto_events
        event_types = [e.type for e in events]
        if expect_summary:
            assert "response.reasoning_summary_text.delta" in event_types
            assert "response.reasoning_summary_text.done" in event_types
        else:
            assert "response.reasoning_summary_text.delta" not in event_types
            assert "response.reasoning_summary_text.done" not in event_types
        assert "response.output_text.delta" in event_types
        assert OpenAIResponsesStreamText in {type(e) for e in events}

        if expect_summary:
            reasoning_delta_indices = [
                i
                for i, e in enumerate(events)
                if e.type == "response.reasoning_summary_text.delta"
            ]
            reasoning_done_indices = [
                i for i, e in enumerate(events) if e.type == "response.reasoning_summary_text.done"
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

        if expect_summary:
            reasoning_deltas = [
                e
                for e in events
                if isinstance(e, OpenAIResponsesStreamText)
                and e.type == "response.reasoning_summary_text.delta"
            ]
            assert len(reasoning_deltas) >= 1
            for delta in reasoning_deltas:
                assert delta.item_id == reasoning_id
            assert any(d.delta is not None and len(d.delta) > 0 for d in reasoning_deltas)

            reasoning_done = [
                e
                for e in events
                if isinstance(e, OpenAIResponsesStreamText)
                and e.type == "response.reasoning_summary_text.done"
            ]
            assert len(reasoning_done) == 1
            assert reasoning_done[0].item_id == reasoning_id
            assert reasoning_done[0].text is not None
            assert len(reasoning_done[0].text) > 0

        assert_stream_text_events(events)

    assert response.id == sdk_response.id
    assert response.status == "completed"
    assert response.reasoning is not None
    assert response.reasoning.effort == "high"
    if expect_summary:
        assert response.reasoning.summary == reasoning["summary"]

    assert_usage(response.usage, expect_reasoning=True)

    assert len(response.output) >= 2

    molto_reasoning = next(o for o in response.output if isinstance(o, OpenAIReasoningItem))
    assert molto_reasoning.type == "reasoning"
    assert molto_reasoning.id == sdk_reasoning.id
    if expect_summary:
        assert len(molto_reasoning.summary) >= 1
        assert molto_reasoning.summary[0].type == "summary_text"
        assert len(molto_reasoning.summary[0].text) > 0
    else:
        assert molto_reasoning.summary == []

    msg_molto = next(o for o in response.output if isinstance(o, OpenAIOutputItem))
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "completed"
    assert msg_molto.id == sdk_msg.id
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0
