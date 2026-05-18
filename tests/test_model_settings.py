import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_text import ResponseOutputText

from molto.providers import OpenAIResponsesProvider
from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAIResponsesStreamText,
    OpenAIWebSearchToolCall,
    _OpenAIStreamEvent,
)
from molto.utils.test import (
    CASSETTE_DIR,
    STREAM_PARAMS,
    assert_stream_text_events,
    assert_usage,
    extract_cassette_response,
    extract_output_ids,
    extract_stream_cassette_response,
    make_handler,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
)

# Settings expected to round-trip Request → YAML cassette → Response (SDK/Molto).
# Every value here MUST exist in the YAML cassette response for the test to be
# meaningful — otherwise the SDK/Molto assertions are checking values that came
# from the request IR rather than from the recorded API response.
EXPECTED_SETTINGS = {
    "instructions": "The user would like to know about the weather in Taiwan today.",
    "max_output_tokens": 400,
    "max_tool_calls": 1,
    "parallel_tool_calls": True,
    "prompt_cache_key": "my_cache_key",
    "prompt_cache_retention": "24h",
    "safety_identifier": "safety",
    "service_tier": "flex",
    "store": False,
    "temperature": 0.45,
    "tool_choice": "required",
    "top_logprobs": 1,
    "top_p": 0.55,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "truncation": "disabled",
}


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_model_settings(stream: bool):

    req_body = dict(
        model="gpt-5.1",
        input=[
            {"role": "user", "content": "Who are you?"},
        ],
        instructions=EXPECTED_SETTINGS["instructions"],
        max_output_tokens=EXPECTED_SETTINGS["max_output_tokens"],
        max_tool_calls=EXPECTED_SETTINGS["max_tool_calls"],
        parallel_tool_calls=EXPECTED_SETTINGS["parallel_tool_calls"],
        prompt_cache_key=EXPECTED_SETTINGS["prompt_cache_key"],
        prompt_cache_retention=EXPECTED_SETTINGS["prompt_cache_retention"],
        safety_identifier=EXPECTED_SETTINGS["safety_identifier"],
        store=EXPECTED_SETTINGS["store"],
        service_tier=EXPECTED_SETTINGS["service_tier"],
        temperature=EXPECTED_SETTINGS["temperature"],
        tool_choice=EXPECTED_SETTINGS["tool_choice"],
        top_logprobs=EXPECTED_SETTINGS["top_logprobs"],
        top_p=EXPECTED_SETTINGS["top_p"],
        truncation=EXPECTED_SETTINGS["truncation"],
        include=[
            "reasoning.encrypted_content",
            "code_interpreter_call.outputs",
            "web_search_call.action.sources",
            "message.output_text.logprobs",
        ],
        tools=[{"type": "web_search_preview"}, {"type": "code_interpreter"}],
        stream=stream,
    )

    cassette_path = CASSETTE_DIR / "openai" / "responses" / "model_settings.yaml"
    handler = make_handler(
        cassette_path,
        req_body,
        [
            "model",
            "input",
            "instructions",
            "max_output_tokens",
            "max_tool_calls",
            "parallel_tool_calls",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "service_tier",
            "temperature",
            "top_logprobs",
        ],
    )

    # ── (1) Verify the YAML cassette response carries every setting ────
    cassette_response = extract_cassette_response(cassette_path, stream=stream)

    for key, expected in EXPECTED_SETTINGS.items():
        assert cassette_response[key] == expected, (
            f"YAML cassette {key}={cassette_response[key]!r}, expected {expected!r}"
        )

    expected_response_id = cassette_response["id"]
    ids = extract_output_ids(cassette_response, "web_search_call", "message")
    expected_ws_id = ids["web_search_call"]
    expected_msg_id = ids["message"]

    stream_resp = extract_stream_cassette_response(cassette_path)
    expected_molto_response_id = stream_resp["id"]
    stream_ids = extract_output_ids(stream_resp, "web_search_call", "message")
    expected_molto_ws_id = stream_ids["web_search_call"]
    expected_molto_msg_id = stream_ids["message"]

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

    # ── (2) Response-level fields parsed by the OpenAI SDK from the YAML ─
    assert response.id == expected_response_id
    assert response.status == "completed"
    assert response.incomplete_details is None
    assert response.model == "gpt-5.1-2025-11-13"
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(response, key)
        assert actual == expected, f"OpenAI SDK response.{key}={actual!r}, expected {expected!r}"
    assert response.reasoning.effort == "none"

    # Output structure: web_search_call + message
    assert len(response.output) == 2

    # Web search call
    ws = response.output[0]
    assert isinstance(ws, ResponseFunctionWebSearch)
    assert ws.type == "web_search_call"
    assert ws.status == "completed"
    assert ws.id == expected_ws_id
    assert ws.action.type == "search"
    assert ws.action.query == "weather: Taiwan"
    assert ws.action.queries == ["weather: Taiwan"]
    assert ws.action.sources is not None
    assert len(ws.action.sources) == 1
    assert ws.action.sources[0].type == "api"
    assert ws.action.sources[0].name == "oai-weather"

    # Message output
    msg = response.output[1]
    assert isinstance(msg, ResponseOutputMessage)
    assert msg.type == "message"
    assert msg.id == expected_msg_id
    assert msg.role == "assistant"
    assert msg.status == "completed"
    assert len(msg.content) == 1

    text_content = msg.content[0]
    assert isinstance(text_content, ResponseOutputText)
    assert text_content.type == "output_text"
    assert len(text_content.text) > 0

    # Logprobs (top_logprobs=1 means 1 alternative per token)
    assert text_content.logprobs is not None
    assert len(text_content.logprobs) > 0
    for lp in text_content.logprobs:
        assert len(lp.top_logprobs) == 1
        assert isinstance(lp.token, str)
        assert isinstance(lp.logprob, float)
        assert isinstance(lp.bytes, list)

    # Usage
    assert response.usage is not None
    assert_usage(response.usage, expect_reasoning=True)
    assert response.usage.input_tokens_details is not None
    assert response.usage.input_tokens_details.cached_tokens >= 0

    # ── Molto client (always streams internally) ───────────────────────
    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.output_text.delta" in {e.type for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.completed"

        added_id = assert_stream_text_events(events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    # ── (3) Response-level fields produced by the Molto pipeline ───────
    # These must match the YAML/SDK values — i.e. Request → IR → Response
    # round-trips without drift.
    assert response.status == "completed"
    assert response.id == expected_molto_response_id
    assert response.incomplete_details is None
    assert response.model == "openai/" + req_body["model"]
    for key, expected in EXPECTED_SETTINGS.items():
        actual = getattr(response, key)
        assert actual == expected, f"Molto response.{key}={actual!r}, expected {expected!r}"
    assert response.reasoning is not None
    assert response.reasoning.effort == "medium"

    # Tools echoed from request
    assert len(response.tools) == 2
    assert response.tools[0].type == "web_search"
    assert response.tools[1].type == "code_interpreter"

    # Completed at from API response
    assert response.completed_at is not None
    assert isinstance(response.completed_at, int)
    assert response.completed_at > 0

    # Output structure: web_search_call + message
    assert len(response.output) == 2

    # Web search call
    ws_molto = response.output[0]
    assert isinstance(ws_molto, OpenAIWebSearchToolCall)
    assert ws_molto.type == "web_search_call"
    assert ws_molto.status == "completed"
    assert ws_molto.id == expected_molto_ws_id
    assert ws_molto.action.type == "search"
    assert ws_molto.action.query == "weather: Taiwan"
    assert ws_molto.action.queries == ["weather: Taiwan"]
    assert ws_molto.action.sources is not None
    assert len(ws_molto.action.sources) == 1
    assert ws_molto.action.sources[0].type == "api"
    assert ws_molto.action.sources[0].name == "oai-weather"

    # Message output
    msg_molto = response.output[1]
    assert isinstance(msg_molto, OpenAIOutputItem)
    assert msg_molto.type == "message"
    assert msg_molto.role == "assistant"
    assert msg_molto.status == "completed"
    assert msg_molto.id == expected_molto_msg_id
    if stream:
        assert msg_molto.id == added_id
    assert len(msg_molto.content) >= 1

    text_molto = msg_molto.content[0]
    assert isinstance(text_molto, OpenAIOutputTextContent)
    assert text_molto.type == "output_text"
    assert len(text_molto.text) > 0

    # Logprobs (top_logprobs=1 means 1 alternative per token)
    assert text_molto.logprobs is not None
    assert len(text_molto.logprobs) > 0

    # Usage
    assert response.usage is not None
    assert_usage(response.usage, expect_reasoning=True)
    assert response.usage.input_tokens_details is not None
    assert response.usage.input_tokens_details.cached_tokens >= 0


@pytest.mark.parametrize("max_tool_calls", [None, 1, 5])
async def test_max_tool_calls_wired_into_usage_limits(max_tool_calls: int | None):
    """`max_tool_calls` must reach pydantic-ai's `UsageLimits.tool_calls_limit`."""
    req = OpenAIResponsesRequest(
        model="openai/gpt-4.1-nano",
        input=[{"role": "user", "content": "hi"}],
        max_tool_calls=max_tool_calls,
        tools=[
            {
                "type": "function",
                "name": "noop",
                "description": "noop",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }
        ],
    )
    ir = req.to_ir()
    assert ir.max_tool_calls == max_tool_calls

    run_settings = await OpenAIResponsesProvider().parse_run_settings(ir)
    assert run_settings["usage_limits"].tool_calls_limit == max_tool_calls
