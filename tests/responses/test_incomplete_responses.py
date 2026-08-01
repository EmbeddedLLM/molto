import pytest
from openai.types.responses import ResponseOutputMessage

from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesStreamText,
)
from tests.support import (
    STREAM_PARAMS,
    assert_stream_text_events,
    assert_usage,
    run_responses_parity,
)

EXPECTED_SETTINGS = {
    "max_output_tokens": 20,
    "instructions": "You are a financial advisor for the user.",
}


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_max_output_tokens_incomplete_round_trip_through_sdk(stream: bool, cassette_manager):
    """Compare incomplete Responses output against SDK-parsed cassettes.

    Example:
        # max_output_tokens=20 returns status "incomplete" with reason "max_output_tokens".
    """

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

    parity = await run_responses_parity(cassette_manager, req_body, terminal="response.incomplete")
    sdk_response = parity.sdk_response
    response = parity.molto_response

    assert sdk_response.status == "incomplete"
    assert sdk_response.incomplete_details is not None
    assert sdk_response.incomplete_details.reason == "max_output_tokens"
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(sdk_response, key)
        assert actual == expected, f"OpenAI SDK response.{key}={actual!r}, expected {expected!r}"

    sdk_msg = next((o for o in sdk_response.output if isinstance(o, ResponseOutputMessage)), None)
    assert sdk_msg is not None
    assert sdk_msg.status == "incomplete"

    if stream:
        assert OpenAIResponsesStreamText in {type(e) for e in parity.molto_events}
        event_types = [e.type for e in parity.molto_events]
        assert "response.output_text.delta" in event_types
        assert_stream_text_events(parity.molto_events)

    assert response.id == sdk_response.id
    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(response, key)
        assert actual == expected, f"Molto response.{key}={actual!r}, expected {expected!r}"

    for msg_molto in response.output:
        assert isinstance(msg_molto, OpenAIOutputItem)
        assert msg_molto.type == "message"
        assert msg_molto.role == "assistant"
        assert msg_molto.status == "incomplete"
        assert msg_molto.id == sdk_msg.id
        assert len(msg_molto.content) >= 1

        text_molto = msg_molto.content[0]
        assert isinstance(text_molto, OpenAIOutputTextContent)
        assert text_molto.type == "output_text"
        assert len(text_molto.text) > 0
        assert "financial advisor" in text_molto.text

    assert_usage(response.usage, output_tokens_exact=20)
