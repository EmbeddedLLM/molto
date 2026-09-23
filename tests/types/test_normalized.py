import pytest
from pydantic import TypeAdapter, ValidationError

from molto.types.normalized import (
    AssistantInput,
    CacheBreakpoint,
    FunctionTool,
    FunctionToolChoice,
    InputItem,
    NormalizedRequest,
    PartStartEvent,
    SystemPrompt,
    TextDelta,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolReturn,
    UserInput,
)


def test_user_input_preserves_message_boundary_around_text_parts():
    first = UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")])
    second = UserInput(content=[TextInput(content="Follow-up")])

    request = NormalizedRequest(input=[first, second])

    assert request.input == [first, second]


def test_text_input_is_not_a_top_level_input_item():
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        NormalizedRequest(input=[TextInput(content="Unwrapped")])


def test_text_output_requires_an_assistant_message_boundary():
    """Ensures assistant history cannot discard its message boundary at construction."""
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        NormalizedRequest(input=[TextOutput(content="Unwrapped")])


def test_assistant_input_preserves_message_boundaries_and_ordered_parts():
    """Ensures assistant messages retain mixed content, adjacent text, IDs, and empty bodies."""
    first = AssistantInput(
        id="msg_1",
        content=[
            Thinking(content="Plan", summary=""),
            TextOutput(content="First", id="output_part_1"),
            TextOutput(content="Second"),
            ToolCall(tool_name="weather", args="{}", tool_call_id="call_1"),
            TextOutput(content="After call"),
        ],
    )
    second = AssistantInput(content=[TextOutput(content="Follow-up")])
    empty = AssistantInput(content=[])

    request = NormalizedRequest(input=[first, second, empty])

    assert request.input == [first, second, empty]


@pytest.mark.parametrize(
    "values",
    [
        {"content": [], "unknown": True},
        {"content": "Unwrapped"},
        {"content": [{"content_type": "text-input", "content": "Wrong role"}]},
        {"content": [{"content_type": "text-output", "content": "Hi", "extra_fields": {}}]},
    ],
)
def test_assistant_input_rejects_invalid_container_and_parts(values):
    """Ensures assistant containers accept only ordered, strictly validated output parts."""
    with pytest.raises(ValidationError):
        AssistantInput(**values)


def test_cache_breakpoints_include_nested_assistant_text_in_order():
    """Ensures shared cache traversal visits assistant text without flattening message parts."""
    system_marker = CacheBreakpoint(ttl="1h")
    assistant_marker = CacheBreakpoint(ttl="5m")
    request = NormalizedRequest(
        input=[
            SystemPrompt(content="Rules", cache_breakpoint=system_marker),
            AssistantInput(
                content=[
                    Thinking(content="Plan", summary=""),
                    TextOutput(content="Cached", cache_breakpoint=assistant_marker),
                    ToolCall(tool_name="weather", args="{}", tool_call_id="call_1"),
                    TextOutput(content="Uncached"),
                ]
            ),
        ]
    )

    assert list(request.iter_cache_breakpoints()) == [system_marker, assistant_marker]


def test_request_rejects_top_level_text():
    with pytest.raises(ValidationError):
        NormalizedRequest(input="Unwrapped")


def test_tool_call_and_return_are_valid_conversation_input():
    tool_call = ToolCall(
        tool_name="get_weather",
        args='{"city":"Paris"}',
        tool_call_id="call_1",
    )
    tool_return = ToolReturn(content=[TextInput(content="sunny")], tool_call_id="call_1")

    request = NormalizedRequest(input=[tool_call, tool_return])

    assert request.input == [tool_call, tool_return]
    assert tool_return.tool_name is None
    assert tool_return.outcome == "success"


def test_tool_return_requires_content_parts():
    with pytest.raises(ValidationError, match="valid list"):
        ToolReturn(content="sunny", tool_call_id="call_1")


def test_request_uses_provider_neutral_function_tools():
    tool = FunctionTool(
        name="get_weather",
        description="Get the weather",
        parameters={"type": "object", "properties": {}},
    )
    choice = FunctionToolChoice(name="get_weather")

    request = NormalizedRequest(tools=[tool], tool_choice=choice)

    assert request.tools == [tool]
    assert request.tool_choice == choice
    assert tool.strict is False


@pytest.mark.parametrize(
    "request_data",
    [
        {"unknown": True},
        {"extra_fields": {}},
        {
            "input": [
                {
                    "content_type": "system-prompt",
                    "content": "Be helpful",
                    "unknown": True,
                }
            ]
        },
        {
            "input": [
                {
                    "content_type": "user-input",
                    "content": [{"content_type": "text-input", "content": "Hi"}],
                    "unknown": True,
                }
            ]
        },
        {
            "input": [
                {
                    "content_type": "user-input",
                    "content": [
                        {
                            "content_type": "text-input",
                            "content": "Hi",
                            "metadata": {"application_field": True},
                        }
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "content_type": "user-input",
                    "content": [
                        {
                            "content_type": "image-url-input",
                            "url": "https://example.com/image.png",
                            "unknown": True,
                        }
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "content_type": "tool-return",
                    "content": [{"content_type": "text-input", "content": "sunny"}],
                    "tool_call_id": "call_1",
                    "unknown": True,
                }
            ]
        },
        {"reasoning": {"effort": "high", "unknown": True}},
        {
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                    "unknown": True,
                }
            ]
        },
        {
            "tool_choice": {
                "type": "function",
                "name": "get_weather",
                "unknown": True,
            }
        },
        {"stream_options": {"include_obfuscation": True, "unknown": True}},
        {"prompt_cache": {"mode": "implicit", "unknown": True}},
        {
            "input": [
                {
                    "content_type": "system-prompt",
                    "content": "s",
                    "cache_breakpoint": {"unknown": True},
                }
            ]
        },
        {
            "input": [
                {
                    "content_type": "assistant-input",
                    "content": [
                        {
                            "content_type": "text-output",
                            "content": "a",
                            "cache_breakpoint": {"unknown": True},
                        }
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "content_type": "user-input",
                    "content": [
                        {
                            "content_type": "text-input",
                            "content": "u",
                            "cache_breakpoint": {"unknown": True},
                        }
                    ],
                }
            ]
        },
    ],
)
def test_request_rejects_extra_fields_at_every_strict_boundary(request_data: dict):
    """Asserts that unknown nested settings are rejected before translation can lose them."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        NormalizedRequest(**request_data)


@pytest.mark.parametrize(
    "item",
    [
        {
            "content_type": "thinking",
            "content": "",
            "summary": "summary",
            "provider_field": "value",
        },
        {
            "content_type": "text-output",
            "content": "Hello",
            "provider_field": "value",
        },
        {
            "content_type": "tool-call",
            "tool_name": "get_weather",
            "args": "{}",
            "tool_call_id": "call_1",
            "provider_field": "value",
        },
    ],
)
def test_request_rejects_extra_fields_on_raw_output_history(item: dict):
    """Ensures nested assistant history rejects provider extensions from raw dictionaries."""
    with pytest.raises(ValidationError, match="provider_field"):
        NormalizedRequest(input=[{"content_type": "assistant-input", "content": [item]}])


@pytest.mark.parametrize(
    "item",
    [
        Thinking(content="", summary="summary", provider_field="value"),
        TextOutput(content="Hello", provider_field="value"),
        ToolCall(
            tool_name="get_weather",
            args="{}",
            tool_call_id="call_1",
            provider_field="value",
        ),
    ],
)
def test_request_rejects_extra_fields_on_preconstructed_output_history(item):
    """Ensures output models with extensions cannot bypass assistant history validation."""
    with pytest.raises(ValidationError, match="provider_field"):
        AssistantInput(content=[item])


def test_output_types_preserve_extra_fields():
    text = TextOutput(content="Hello", provider_field="value")
    delta = TextDelta(content_delta="Hello", provider_field="value")
    event = PartStartEvent(index=0, part=text, provider_event_field="value")

    assert text.extra_fields == {"provider_field": "value"}
    assert delta.extra_fields == {"provider_field": "value"}
    assert event.extra_fields == {"provider_event_field": "value"}
    assert event.part.extra_fields == {"provider_field": "value"}


def test_function_tool_parameters_remain_an_open_json_schema():
    request = NormalizedRequest(
        tools=[
            FunctionTool(
                name="get_weather",
                parameters={
                    "type": "object",
                    "provider_keyword": {"enabled": True},
                },
            )
        ]
    )

    assert request.tools[0].parameters["provider_keyword"] == {"enabled": True}


def test_user_input_preserves_ordered_text_and_image_parts():
    message = TypeAdapter(InputItem).validate_python(
        {
            "content_type": "user-input",
            "content": [
                {"content_type": "text-input", "content": "before"},
                {
                    "content_type": "image-url-input",
                    "url": "data:image/png;base64,AA==",
                    "detail": "high",
                },
                {"content_type": "text-input", "content": "after"},
            ],
        }
    )

    assert isinstance(message, UserInput)
    assert [part.content_type for part in message.content] == [
        "text-input",
        "image-url-input",
        "text-input",
    ]
    assert message.content[0].content == "before"
    assert message.content[1].url == "data:image/png;base64,AA=="
    assert message.content[1].detail == "high"
    assert message.content[2].content == "after"


@pytest.mark.parametrize(
    "values",
    [
        {"content": "raw", "summary": "summary"},
        {"content": "raw", "summary": "", "signature": "opaque-state"},
    ],
)
def test_thinking_rejects_contradictory_representations(values: dict) -> None:
    with pytest.raises(ValidationError):
        Thinking(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"content_delta": "raw", "summary_delta": "summary"},
        {"content_delta": "raw", "signature_delta": "opaque-state"},
    ],
)
def test_thinking_delta_rejects_contradictory_representations(values: dict) -> None:
    with pytest.raises(ValidationError):
        ThinkingDelta(**values)


def test_thinking_keeps_protocol_specific_valid_states() -> None:
    assert Thinking(content="", summary="summary").signature is None
    assert Thinking(content="", summary="", signature="opaque-state").signature == "opaque-state"
    assert ThinkingDelta(summary_delta="summary").signature_delta is None
    assert ThinkingDelta(signature_delta="opaque-state").summary_delta is None
