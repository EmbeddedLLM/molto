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
    OpenAIResponsesStreamOutput,
    OpenAIResponsesStreamText,
    OpenAIWebSearchToolCall,
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
    cassette_data = read_cassette(cassette_path, stream=stream)
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

    for key, expected in EXPECTED_SETTINGS.items():
        assert cassette_response[key] == expected, (
            f"YAML cassette {key}={cassette_response[key]!r}, expected {expected!r}"
        )

    expected_response_id = cassette_response["id"]
    cassette_ws = [o for o in cassette_response["output"] if o["type"] == "web_search_call"]
    assert len(cassette_ws) == 1
    expected_ws_id = cassette_ws[0]["id"]
    cassette_msg = [o for o in cassette_response["output"] if o["type"] == "message"]
    assert len(cassette_msg) == 1
    expected_msg_id = cassette_msg[0]["id"]

    stream_cassette_data = read_cassette(cassette_path, stream=True)
    stream_completed = [
        line
        for line in stream_cassette_data
        if line.startswith("data: ") and '"type":"response.completed"' in line
    ]
    assert len(stream_completed) == 1
    stream_resp = json_loads(stream_completed[0][len("data: ") :])["response"]
    expected_molto_response_id = stream_resp["id"]
    stream_ws = [o for o in stream_resp["output"] if o["type"] == "web_search_call"]
    expected_molto_ws_id = stream_ws[0]["id"]
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
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert (
        response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens
    )
    assert response.usage.input_tokens_details is not None
    assert response.usage.input_tokens_details.cached_tokens >= 0
    assert response.usage.output_tokens_details is not None
    assert response.usage.output_tokens_details.reasoning_tokens >= 0

    # ── Molto client (always streams internally) ───────────────────────
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
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert (
        response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens
    )
    assert response.usage.input_tokens_details is not None
    assert response.usage.input_tokens_details.cached_tokens >= 0
    assert response.usage.output_tokens_details is not None
    assert response.usage.output_tokens_details.reasoning_tokens >= 0


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
