import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
)

from molto.types.openai.responses import (
    OpenAIFunctionToolCall,
    OpenAIResponsesStreamOutput,
    OpenAIResponsesWebSearchFilters,
    OpenAIResponsesWebSearchTool,
    OpenAIResponsesWebSearchUserLocation,
    OpenAIWebSearchActionSearch,
    OpenAIWebSearchToolCall,
)
from tests.support import (
    STREAM_PARAMS,
    assert_events_before_terminal,
    run_responses_parity,
)


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_required_function_tool_round_trip_through_sdk(stream: bool, cassette_manager):
    """Compare required function-tool output against SDK-parsed cassettes.

    Example:
        # stream=True records/replays the default stream cassette and yields a function_call.
    """

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
                "strict": True,
            },
        ],
        tool_choice="required",
        stream=stream,
    )

    parity = await run_responses_parity(cassette_manager, req_body)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    assert sdk_response.status == "completed"
    if stream:
        sdk_event_types = [e.type for e in parity.sdk_events]
        assert "response.function_call_arguments.delta" in sdk_event_types
        assert_events_before_terminal(parity.sdk_events, "response.function_call_arguments.delta")

    sdk_fc = next(o for o in sdk_response.output if isinstance(o, ResponseFunctionToolCall))
    assert isinstance(sdk_fc.name, str) and len(sdk_fc.name) > 0
    assert isinstance(sdk_fc.id, str) and len(sdk_fc.id) > 0
    assert isinstance(sdk_fc.call_id, str) and len(sdk_fc.call_id) > 0

    if stream:
        event_types = [e.type for e in parity.molto_events]
        assert "response.function_call_arguments.delta" in event_types

    fc = next(o for o in response.output if isinstance(o, OpenAIFunctionToolCall))
    assert fc.name == sdk_fc.name
    assert response.id == sdk_response.id
    assert fc.id == sdk_fc.id
    assert fc.call_id == sdk_fc.call_id


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
async def test_web_search_tool_round_trip_through_sdk(stream: bool, cassette_manager):
    """Verify native web-search tool parity against SDK-parsed cassettes.

    Example:
        # stream=True records/replays the default stream cassette and yields a web_search_call.
    """

    req_body = dict(
        model="gpt-5-nano",
        input=[{"role": "user", "content": "Hello!"}],
        tool_choice="required",
        tools=[EXPECTED_WEB_SEARCH_TOOL],
        include=["reasoning.encrypted_content"],
        stream=stream,
    )

    parity = await run_responses_parity(cassette_manager, req_body)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    assert sdk_response.status == "completed"

    # Validate the SDK parsed the tool configuration from the cassette
    sdk_tools = sdk_response.tools
    assert isinstance(sdk_tools, list) and len(sdk_tools) == 1
    sdk_tool = sdk_tools[0]
    assert sdk_tool.type == EXPECTED_WEB_SEARCH_TOOL["type"]
    assert sdk_tool.search_context_size == EXPECTED_WEB_SEARCH_TOOL["search_context_size"]
    assert (
        sdk_tool.filters.allowed_domains == EXPECTED_WEB_SEARCH_TOOL["filters"]["allowed_domains"]
    )
    sdk_loc = sdk_tool.user_location
    assert sdk_loc.city == EXPECTED_WEB_SEARCH_TOOL["user_location"]["city"]
    assert sdk_loc.country == EXPECTED_WEB_SEARCH_TOOL["user_location"]["country"]
    assert sdk_loc.region == EXPECTED_WEB_SEARCH_TOOL["user_location"]["region"]

    sdk_ws = next(o for o in sdk_response.output if isinstance(o, ResponseFunctionWebSearch))
    sdk_msg = next(o for o in sdk_response.output if isinstance(o, ResponseOutputMessage))
    assert sdk_ws.type == "web_search_call"
    assert sdk_ws.status == "completed"
    assert isinstance(sdk_ws.id, str) and len(sdk_ws.id) > 0

    if stream:
        ws_done = [
            e
            for e in parity.molto_events
            if isinstance(e, OpenAIResponsesStreamOutput)
            and e.type == "response.output_item.done"
            and isinstance(e.item, OpenAIWebSearchToolCall)
        ]
        assert len(ws_done) == 1
        assert ws_done[0].item.status == "completed"
        assert ws_done[0].item.id == sdk_ws.id

    assert response.id == sdk_response.id
    assert isinstance(response.output, list)

    molto_ws_items = [o for o in response.output if isinstance(o, OpenAIWebSearchToolCall)]
    assert len(molto_ws_items) == 1
    molto_ws = molto_ws_items[0]
    assert molto_ws.type == "web_search_call"
    assert molto_ws.status == "completed"
    assert molto_ws.id == sdk_ws.id

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
    assert assistant_msgs[0].id == sdk_msg.id
    assert assistant_msgs[0].role == "assistant"
    assert assistant_msgs[0].status == "completed"
    assert len(assistant_msgs[0].content) >= 1
    assert len(assistant_msgs[0].content[0].text) > 0
