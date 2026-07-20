import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
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
from molto.utils.test import (
    STREAM_PARAMS,
    assert_events_before_terminal,
    assert_output_item_order,
    assert_stream_events_match,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    stream_reference_cassette_suffix,
)


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_tool_choice_required(stream: bool, cassette_manager):
    """Compare required function-tool output against SDK-parsed cassettes.

    Example:
        # stream=False records/replays the default non-stream cassette and returns a function call.
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

    assert_attrs = ["model", "input", "tool_choice"]

    # -- OpenAI SDK (parametrized stream) --------------------------------
    sdk_client = make_openai_client(
        cassette_manager,
        expected_request_body=req_body,
        assert_attrs=assert_attrs,
    )
    sdk_response = await sdk_client.responses.create(**req_body)
    if stream:
        sdk_events = [e async for e in sdk_response]
        sdk_event_types = [e.type for e in sdk_events]
        assert isinstance(sdk_events[-1], ResponseCompletedEvent)
        assert sdk_event_types[-1] == "response.completed"
        assert "response.function_call_arguments.delta" in sdk_event_types

        assert_output_item_order(sdk_events)

        assert_events_before_terminal(sdk_events, "response.function_call_arguments.delta")

        sdk_response = sdk_events[-1].response
    assert isinstance(sdk_response, Response)
    assert sdk_response.status == "completed"

    sdk_fc = next(o for o in sdk_response.output if isinstance(o, ResponseFunctionToolCall))
    assert isinstance(sdk_fc.name, str) and len(sdk_fc.name) > 0
    assert isinstance(sdk_fc.id, str) and len(sdk_fc.id) > 0
    assert isinstance(sdk_fc.call_id, str) and len(sdk_fc.call_id) > 0

    stream_ref_suffix = stream_reference_cassette_suffix(stream)

    # -- Stream SDK reference (matches Molto's internal streaming) -------
    stream_req_body = {**req_body, "stream": True}
    ref_client = make_openai_client(
        cassette_manager,
        cassette_suffix=stream_ref_suffix,
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    ref_events = [e async for e in await ref_client.responses.create(**stream_req_body)]
    assert isinstance(ref_events[-1], ResponseCompletedEvent)
    sdk_ref = ref_events[-1].response
    sdk_ref_fc = next(o for o in sdk_ref.output if isinstance(o, ResponseFunctionToolCall))

    # -- Molto client ----------------------------------------------------
    client = make_molto_client(
        cassette_manager,
        cassette_suffix=stream_ref_suffix,
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        event_types = [e.type for e in events]
        assert OpenAIResponsesStreamText in {type(e) for e in events}
        assert "response.function_call_arguments.delta" in event_types
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert event_types[-1] == "response.completed"

        assert_output_item_order(events)

        assert_stream_events_match(ref_events, events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)

    fc = next(o for o in response.output if isinstance(o, OpenAIFunctionToolCall))
    assert fc.name == sdk_ref_fc.name
    assert response.id == sdk_ref.id
    assert fc.id == sdk_ref_fc.id
    assert fc.call_id == sdk_ref_fc.call_id


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
async def test_native_tools_web_search(stream: bool, cassette_manager):
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

    assert_attrs = ["model", "input", "tool_choice"]

    # -- OpenAI SDK (parametrized stream) --------------------------------
    sdk_client = make_openai_client(
        cassette_manager,
        expected_request_body=req_body,
        assert_attrs=assert_attrs,
    )
    sdk_response = await sdk_client.responses.create(**req_body)
    if stream:
        sdk_events = [e async for e in sdk_response]
        sdk_event_types = [e.type for e in sdk_events]
        assert isinstance(sdk_events[-1], ResponseCompletedEvent)
        assert sdk_event_types[-1] == "response.completed"

        assert_output_item_order(sdk_events)

        sdk_response = sdk_events[-1].response
    assert isinstance(sdk_response, Response)
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
    assert sdk_ws.type == "web_search_call"
    assert sdk_ws.status == "completed"
    assert isinstance(sdk_ws.id, str) and len(sdk_ws.id) > 0

    stream_ref_suffix = stream_reference_cassette_suffix(stream)

    # -- Stream SDK reference (matches Molto's internal streaming) -------
    stream_req_body = {**req_body, "stream": True}
    ref_client = make_openai_client(
        cassette_manager,
        cassette_suffix=stream_ref_suffix,
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    ref_events = [e async for e in await ref_client.responses.create(**stream_req_body)]
    assert isinstance(ref_events[-1], ResponseCompletedEvent)
    sdk_ref = ref_events[-1].response
    sdk_ref_ws = next(o for o in sdk_ref.output if isinstance(o, ResponseFunctionWebSearch))
    sdk_ref_msg = next(o for o in sdk_ref.output if isinstance(o, ResponseOutputMessage))

    # -- Molto client ----------------------------------------------------
    client = make_molto_client(
        cassette_manager,
        cassette_suffix=stream_ref_suffix,
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    response = await client.chat(OpenAIResponsesRequest(**prepend_model_prefix(req_body)))
    if stream:
        events: list[_OpenAIStreamEvent] = [r async for r in response]
        event_types = [e.type for e in events]
        assert isinstance(events[-1], OpenAIResponsesStream)
        assert event_types[-1] == "response.completed"

        ws_done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamOutput)
            and e.type == "response.output_item.done"
            and isinstance(e.item, OpenAIWebSearchToolCall)
        ]
        assert len(ws_done) == 1
        assert ws_done[0].item.status == "completed"
        assert ws_done[0].item.id == sdk_ref_ws.id

        assert_output_item_order(events)

        assert_stream_events_match(ref_events, events)

        response = events[-1].response
    assert isinstance(response, OpenAIResponsesResponse)
    assert response.id == sdk_ref.id
    assert isinstance(response.output, list)

    molto_ws_items = [o for o in response.output if isinstance(o, OpenAIWebSearchToolCall)]
    assert len(molto_ws_items) == 1
    molto_ws = molto_ws_items[0]
    assert molto_ws.type == "web_search_call"
    assert molto_ws.status == "completed"
    assert molto_ws.id == sdk_ref_ws.id

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
    assert assistant_msgs[0].id == sdk_ref_msg.id
    assert assistant_msgs[0].role == "assistant"
    assert assistant_msgs[0].status == "completed"
    assert len(assistant_msgs[0].content) >= 1
    assert len(assistant_msgs[0].content[0].text) > 0
