import pytest
from pydantic import TypeAdapter, ValidationError

from molto.types.normalized import (
    FunctionTool,
    FunctionToolChoice,
    InputItem,
    NormalizedRequest,
    TextInput,
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


def test_tool_call_and_return_are_valid_conversation_input():
    tool_call = ToolCall(
        tool_name="get_weather",
        args='{"city":"Paris"}',
        tool_call_id="call_1",
    )
    tool_return = ToolReturn(content="sunny", tool_call_id="call_1")

    request = NormalizedRequest(input=[tool_call, tool_return])

    assert request.input == [tool_call, tool_return]
    assert tool_return.tool_name is None
    assert tool_return.outcome == "success"


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
