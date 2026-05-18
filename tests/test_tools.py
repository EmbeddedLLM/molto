import pytest

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
from molto.utils.test import (
    CASSETTE_DIR,
    STREAM_PARAMS,
    extract_cassette_response,
    extract_output_ids,
    extract_stream_cassette_response,
    make_handler,
    make_molto_client,
    prepend_model_prefix,
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
    handler = make_handler(cassette_path, req_body, ["model", "input", "tool_choice"])

    cassette_resp = extract_cassette_response(cassette_path, stream=stream)
    expected_fc = cassette_resp["output"][0]
    expected_name = expected_fc["name"]

    stream_resp = extract_stream_cassette_response(cassette_path)
    stream_fc = stream_resp["output"][0]
    expected_molto_response_id = stream_resp["id"]
    expected_molto_fc_id = stream_fc["id"]
    expected_molto_call_id = stream_fc["call_id"]

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
    2. parsed correctly by the Molto pipeline (request IR → from_ir → response).
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

    cassette_response = extract_cassette_response(cassette_path, stream=stream)
    cassette_tools = cassette_response["tools"]
    assert isinstance(cassette_tools, list)
    assert len(cassette_tools) == 1
    yaml_tool = cassette_tools[0]
    assert yaml_tool["type"] == EXPECTED_WEB_SEARCH_TOOL["type"]
    assert yaml_tool["search_context_size"] == EXPECTED_WEB_SEARCH_TOOL["search_context_size"]
    assert yaml_tool["filters"] == EXPECTED_WEB_SEARCH_TOOL["filters"]
    assert yaml_tool["user_location"] == EXPECTED_WEB_SEARCH_TOOL["user_location"]

    stream_resp = extract_stream_cassette_response(cassette_path)
    expected_molto_response_id = stream_resp["id"]
    stream_ids = extract_output_ids(stream_resp, "web_search_call", "message")
    expected_molto_ws_id = stream_ids["web_search_call"]
    expected_molto_msg_id = stream_ids["message"]

    client = make_molto_client(handler)
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert events[-1].type == "response.completed"

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

    assistant_msgs = [o for o in response.output if getattr(o, "type", None) == "message"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].id == expected_molto_msg_id
    assert assistant_msgs[0].role == "assistant"
    assert assistant_msgs[0].status == "completed"
    assert len(assistant_msgs[0].content) >= 1
    assert len(assistant_msgs[0].content[0].text) > 0
