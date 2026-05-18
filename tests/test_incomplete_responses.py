import pytest

from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
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
    assert_usage,
    extract_cassette_response,
    extract_stream_cassette_response,
    make_handler,
    make_molto_client,
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

    cassette_response = extract_cassette_response(
        cassette_path, stream=stream, terminal_event_type="response.incomplete"
    )

    for key, expected in EXPECTED_SETTINGS.items():
        assert cassette_response[key] == expected, (
            f"YAML cassette {key}={cassette_response[key]!r}, expected {expected!r}"
        )
    assert cassette_response["incomplete_details"]["reason"] == "max_output_tokens"

    stream_resp = extract_stream_cassette_response(
        cassette_path, terminal_event_type="response.incomplete"
    )
    expected_molto_response_id = stream_resp["id"]
    expected_molto_msg_id = stream_resp["output"][0]["id"]

    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.incomplete"

        output_item_added = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamOutput)
            and e.type == "response.output_item.added"
        ]
        output_item_done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamOutput) and e.type == "response.output_item.done"
        ]
        assert len(output_item_added) >= 1
        assert len(output_item_done) >= 1
        added_id = output_item_added[0].item.id
        done_id = output_item_done[0].item.id
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
            assert delta.item_id == added_id, (
                f"delta item_id {delta.item_id!r} != added item id {added_id!r}"
            )

        text_done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamText) and e.type == "response.output_text.done"
        ]
        assert len(text_done) == 1
        assert text_done[0].item_id == added_id

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    assert response.id == expected_molto_response_id
    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(response, key)
        assert actual == expected, f"Molto response.{key}={actual!r}, expected {expected!r}"

    assert len(response.output) == 1
    msg_molto = response.output[0]
    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "incomplete"
    assert msg_molto.id == expected_molto_msg_id
    if stream:
        assert msg_molto.id == added_id
    assert len(msg_molto.content) == 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0
    assert "financial advisor" in text_molto.text

    assert_usage(response.usage, output_tokens_exact=20)
