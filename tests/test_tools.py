import httpx
import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
)

from molto.types.openai.responses import (
    OpenAIFunctionToolCall,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAIResponsesStreamOutput,
    OpenAIResponsesStreamText,
    OpenAIResponsesWebSearchFilters,
    OpenAIResponsesWebSearchTool,
    OpenAIResponsesWebSearchUserLocation,
    OpenAIWebSearchActionSearch,
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


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_tool_choice_required(stream: bool):

    req_body = dict(
        model="gpt-4.1-nano",
        input=[{"role": "user", "content": "What number am I thinking of?"}],
        tools=[
            {
                "type": "function",
                "name": "guess_number",
                "description": "Guess a number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer", "description": "The guessed number."},
                    },
                    "required": ["number"],
                    "additionalProperties": False,
                },
            },
        ],
        tool_choice="required",
        stream=stream,
    )

    cassette_path = CASSETTE_DIR / "openai" / "responses" / "tool_choice-required.yaml"

    cassette_data = read_cassette(cassette_path, stream=stream)
    if stream:
        completed_frames = [
            line
            for line in cassette_data
            if line.startswith("data: ") and '"type":"response.completed"' in line
        ]
        assert len(completed_frames) == 1
        cassette_resp = json_loads(completed_frames[0][len("data: ") :])["response"]
    else:
        cassette_resp = cassette_data
    expected_fc = cassette_resp["output"][0]
    expected_response_id = cassette_resp["id"]
    expected_fc_id = expected_fc["id"]
    expected_call_id = expected_fc["call_id"]
    expected_name = expected_fc["name"]

    # Molto always reads the stream cassette internally
    stream_cassette_data = read_cassette(cassette_path, stream=True)
    completed_frames = [
        line
        for line in stream_cassette_data
        if line.startswith("data: ") and '"type":"response.completed"' in line
    ]
    assert len(completed_frames) == 1
    stream_resp = json_loads(completed_frames[0][len("data: ") :])["response"]
    stream_fc = stream_resp["output"][0]
    expected_molto_response_id = stream_resp["id"]
    expected_molto_fc_id = stream_fc["id"]
    expected_molto_call_id = stream_fc["call_id"]

    def handler(request: httpx.Request, load_stream_cassette: bool) -> httpx.Response:
        try:
            assert request.method == "POST"
            body = json_loads(request.content)
            for attr in ["model", "input", "tool_choice"]:
                assert body.get(attr) == req_body[attr], (
                    f"Expected {attr} to be {req_body[attr]!r}, got {body.get(attr)!r}"
                )
            assert isinstance(body.get("tools"), list)
            assert len(body["tools"]) == 1
            assert body["tools"][0].get("name") == "guess_number"
            assert len(body.get("input", [])) == 1
            assert body.get("stream", False) is load_stream_cassette
            data = read_cassette(cassette_path, stream=load_stream_cassette)
            if load_stream_cassette:
                return httpx.Response(
                    200,
                    content="".join(data).encode(),
                    headers={"content-type": "text/event-stream; charset=utf-8"},
                )
            return httpx.Response(200, json=data)
        except AssertionError as e:
            return httpx.Response(
                400,
                json={"error": f"Assertions failed: {e!r}"},
            )
        except Exception as e:
            return httpx.Response(
                400,
                json={"error": f"Invalid request: {e!r}"},
            )

    client = make_openai_client(handler, load_stream_cassette=stream)
    response = await client.responses.create(**req_body)
    if stream:
        events = [r async for r in response]
        assert ResponseFunctionCallArgumentsDeltaEvent in {type(e) for e in events}
        assert "response.function_call_arguments.delta" in {e.type for e in events}
        assert isinstance(events[-1], ResponseCompletedEvent)
        response = events[-1].response
    assert isinstance(response, Response)
    assert isinstance(response.output, list)
    assert isinstance(response.output[0], ResponseFunctionToolCall)
    assert response.output[0].name == expected_name
    assert response.id == expected_response_id
    assert response.output[0].id == expected_fc_id
    assert response.output[0].call_id == expected_call_id

    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.function_call_arguments.delta" in {e.type for e in events}
        assert isinstance(events[-1], OpenAIResponsesStream)
        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)
    assert isinstance(response.output, list)
    assert isinstance(response.output[0], OpenAIFunctionToolCall)
    assert response.output[0].name == expected_name
    assert response.id == expected_molto_response_id
    assert response.output[0].id == expected_molto_fc_id
    assert response.output[0].call_id == expected_molto_call_id


# Expected web_search tool settings — these MUST exist in both the YAML cassette
# (the response body for non-stream and the SSE `response.created`/`response.completed`
# frames for stream) AND in the parsed response object on both the OpenAI SDK side
# (parsed from the YAML) and the Molto side (rebuilt from the request IR).
EXPECTED_WEB_SEARCH_TOOL = {
    "type": "web_search",
    "search_context_size": "high",
    "filters": {
        "allowed_domains": ["wikipedia.org", "example.com"],
    },
    "user_location": {
        "type": "approximate",
        "city": "San Francisco",
        "country": "US",
        "region": "California",
        "timezone": "America/Los_Angeles",
    },
}


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_builtin_tools_web_search(stream: bool):
    """Verify every field of `OpenAIResponsesWebSearchTool` (type,
    search_context_size, filters.allowed_domains, full user_location) is:

    1. present verbatim in the cassette YAML (the recorded API response),
    2. parsed correctly by the OpenAI SDK from that YAML, and
    3. parsed correctly by the Molto pipeline (request IR → from_ir → response).

    All three must agree.
    """

    req_body = dict(
        model="gpt-5-nano",
        input=[{"role": "user", "content": "Hello!"}],
        tool_choice="required",
        tools=[EXPECTED_WEB_SEARCH_TOOL],
        include=["reasoning.encrypted_content"],
        stream=stream,
    )

    cassette_path = (
        CASSETTE_DIR / "openai" / "responses" / "tools_choice-required-builtin-tools.yaml"
    )
    handler = make_handler(
        cassette_path,
        req_body,
        ["model", "input", "tool_choice"],
    )

    # ── (1) Verify the YAML cassette itself has all settings ───────────
    cassette_data = read_cassette(cassette_path, stream=stream)
    if stream:
        completed_frames = [
            line
            for line in cassette_data
            if line.startswith("data: ") and '"type":"response.completed"' in line
        ]
        assert len(completed_frames) == 1
        completed_payload = json_loads(completed_frames[0][len("data: ") :])
        cassette_response = completed_payload["response"]
        cassette_tools = cassette_response["tools"]
    else:
        cassette_response = cassette_data
        cassette_tools = cassette_data["tools"]

    assert isinstance(cassette_tools, list)
    assert len(cassette_tools) == 1
    yaml_tool = cassette_tools[0]
    assert yaml_tool["type"] == EXPECTED_WEB_SEARCH_TOOL["type"]
    assert yaml_tool["search_context_size"] == EXPECTED_WEB_SEARCH_TOOL["search_context_size"]
    assert yaml_tool["filters"] == EXPECTED_WEB_SEARCH_TOOL["filters"]
    assert yaml_tool["user_location"] == EXPECTED_WEB_SEARCH_TOOL["user_location"]

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

    # ── (2) OpenAI SDK client — parses the YAML directly ───────────────
    client = make_openai_client(handler, load_stream_cassette=stream)
    response = await client.responses.create(**req_body)
    if stream:
        events = [r async for r in response]
        assert isinstance(events[-1], ResponseCompletedEvent)
        sdk_event_types = {e.type for e in events}
        assert "response.web_search_call.in_progress" in sdk_event_types
        assert "response.web_search_call.searching" in sdk_event_types
        assert "response.web_search_call.completed" in sdk_event_types
        response = events[-1].response
    assert isinstance(response, Response)
    assert response.id == expected_response_id
    assert isinstance(response.output, list)

    # web_search_call output item
    sdk_ws_items = [o for o in response.output if isinstance(o, ResponseFunctionWebSearch)]
    assert len(sdk_ws_items) == 1
    sdk_ws = sdk_ws_items[0]
    assert sdk_ws.type == "web_search_call"
    assert sdk_ws.status == "completed"
    assert sdk_ws.id == expected_ws_id
    assert sdk_ws.action.type == "search"
    assert isinstance(sdk_ws.action.query, str)
    assert len(sdk_ws.action.query) > 0
    assert isinstance(sdk_ws.action.queries, list)
    assert len(sdk_ws.action.queries) >= 1
    assert all(isinstance(q, str) and len(q) > 0 for q in sdk_ws.action.queries)
    # `web_search_call.action.sources` was not in `include`, so sources are absent
    assert sdk_ws.action.sources is None

    # Tool definition echoed in the response — must come from the YAML
    assert len(response.tools) == 1
    sdk_tool = response.tools[0]
    assert sdk_tool.type == EXPECTED_WEB_SEARCH_TOOL["type"]
    assert sdk_tool.search_context_size == EXPECTED_WEB_SEARCH_TOOL["search_context_size"]
    assert sdk_tool.filters is not None
    assert (
        sdk_tool.filters.allowed_domains == EXPECTED_WEB_SEARCH_TOOL["filters"]["allowed_domains"]
    )
    assert sdk_tool.user_location is not None
    assert sdk_tool.user_location.type == EXPECTED_WEB_SEARCH_TOOL["user_location"]["type"]
    assert sdk_tool.user_location.city == EXPECTED_WEB_SEARCH_TOOL["user_location"]["city"]
    assert sdk_tool.user_location.country == EXPECTED_WEB_SEARCH_TOOL["user_location"]["country"]
    assert sdk_tool.user_location.region == EXPECTED_WEB_SEARCH_TOOL["user_location"]["region"]
    assert (
        str(sdk_tool.user_location.timezone)
        == EXPECTED_WEB_SEARCH_TOOL["user_location"]["timezone"]
    )

    # SDK message ID
    sdk_msg_items = [o for o in response.output if isinstance(o, ResponseOutputMessage)]
    assert len(sdk_msg_items) == 1
    assert sdk_msg_items[0].id == expected_msg_id

    # ── (3) Molto client — request IR → from_ir → response ─────────────
    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.completed"

        # The molto stream surfaces the web_search_call as a completed output item
        ws_done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamOutput)
            and e.type == "response.output_item.done"
            and isinstance(e.item, OpenAIWebSearchToolCall)
        ]
        assert len(ws_done) == 1
        assert ws_done[0].item.status == "completed"
        assert ws_done[0].item.id == expected_molto_ws_id

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)
    assert response.id == expected_molto_response_id
    assert isinstance(response.output, list)

    # Web search call is fully parsed into the typed pydantic model
    molto_ws_items = [o for o in response.output if isinstance(o, OpenAIWebSearchToolCall)]
    assert len(molto_ws_items) == 1
    molto_ws = molto_ws_items[0]
    assert molto_ws.type == "web_search_call"
    assert molto_ws.status == "completed"
    assert molto_ws.id == expected_molto_ws_id

    assert isinstance(molto_ws.action, OpenAIWebSearchActionSearch)
    assert molto_ws.action.type == "search"
    assert isinstance(molto_ws.action.query, str)
    assert len(molto_ws.action.query) > 0
    assert isinstance(molto_ws.action.queries, list)
    assert len(molto_ws.action.queries) >= 1
    assert all(isinstance(q, str) and len(q) > 0 for q in molto_ws.action.queries)
    assert molto_ws.action.sources is None

    # Tool definition surfaced on the molto response — must agree with the YAML
    assert len(response.tools) == 1
    molto_tool = response.tools[0]
    assert isinstance(molto_tool, OpenAIResponsesWebSearchTool)
    assert molto_tool.type == EXPECTED_WEB_SEARCH_TOOL["type"]
    assert molto_tool.search_context_size == EXPECTED_WEB_SEARCH_TOOL["search_context_size"]

    assert isinstance(molto_tool.filters, OpenAIResponsesWebSearchFilters)
    assert (
        molto_tool.filters.allowed_domains
        == EXPECTED_WEB_SEARCH_TOOL["filters"]["allowed_domains"]
    )

    assert isinstance(molto_tool.user_location, OpenAIResponsesWebSearchUserLocation)
    assert molto_tool.user_location.type == EXPECTED_WEB_SEARCH_TOOL["user_location"]["type"]
    assert molto_tool.user_location.city == EXPECTED_WEB_SEARCH_TOOL["user_location"]["city"]
    assert molto_tool.user_location.country == EXPECTED_WEB_SEARCH_TOOL["user_location"]["country"]
    assert molto_tool.user_location.region == EXPECTED_WEB_SEARCH_TOOL["user_location"]["region"]
    assert (
        str(molto_tool.user_location.timezone)
        == EXPECTED_WEB_SEARCH_TOOL["user_location"]["timezone"]
    )

    # A trailing assistant message is also produced
    assistant_msgs = [o for o in response.output if getattr(o, "type", None) == "message"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].id == expected_molto_msg_id
    assert assistant_msgs[0].role == "assistant"
    assert assistant_msgs[0].status == "completed"
    assert len(assistant_msgs[0].content) >= 1
    assert len(assistant_msgs[0].content[0].text) > 0
