
import pytest
from openai.types.responses import (
    Response,
    ResponseIncompleteEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_text import ResponseOutputText

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
from molto.utils.io import json_loads
from molto.utils.test import (
    CASSETTE_DIR,
    STREAM_PARAMS,
    make_handler,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    read_cassette,
)

# Settings expected to round-trip Request → YAML cassette → Response.
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

    # ── (1) Verify the YAML cassette response carries every setting ────
    cassette_data = read_cassette(cassette_path, stream=stream)
    if stream:
        incomplete_frames = [
            line
            for line in cassette_data
            if line.startswith("data: ") and '"type":"response.incomplete"' in line
        ]
        assert len(incomplete_frames) == 1
        cassette_response = json_loads(incomplete_frames[0][len("data: ") :])["response"]
    else:
        cassette_response = cassette_data

    for key, expected in EXPECTED_SETTINGS.items():
        assert cassette_response[key] == expected, (
            f"YAML cassette {key}={cassette_response[key]!r}, expected {expected!r}"
        )
    assert cassette_response["incomplete_details"]["reason"] == "max_output_tokens"

    client = make_openai_client(handler, load_stream_cassette=stream)
    response = await client.responses.create(**req_body)
    if stream:
        events = [r async for r in response]
        assert ResponseTextDeltaEvent in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert ResponseIncompleteEvent in {type(e) for e in events}
        assert isinstance(events[-1], ResponseIncompleteEvent)
        response = events[-1].response
    assert isinstance(response, Response)

    # ── (2) OpenAI SDK response — parsed from the YAML cassette ────────
    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(response, key)
        assert actual == expected, (
            f"OpenAI SDK response.{key}={actual!r}, expected {expected!r}"
        )

    assert len(response.output) == 1
    msg = response.output[0]
    assert isinstance(msg, ResponseOutputMessage)
    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.status == "incomplete"
    assert len(msg.content) == 1

    text_content = msg.content[0]
    assert isinstance(text_content, ResponseOutputText)
    assert text_content.type == "output_text"
    assert len(text_content.text) > 0
    assert "financial advisor" in text_content.text

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens == 20
    assert (
        response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens
    )

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

    # ── (3) Molto response — must agree with the YAML/SDK values ───────
    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(response, key)
        assert actual == expected, (
            f"Molto response.{key}={actual!r}, expected {expected!r}"
        )

    assert len(response.output) == 1
    msg_molto = response.output[0]
    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "incomplete"
    assert msg_molto.id is not None
    assert msg_molto.id.startswith("msg_")
    if stream:
        assert msg_molto.id == added_id
    assert len(msg_molto.content) == 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0
    assert "financial advisor" in text_molto.text

    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens == 20
    assert (
        response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens
    )
