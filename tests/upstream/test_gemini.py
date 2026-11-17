"""Supplementary synthetic Gemini history cases; provider validation is separate."""

import pytest

from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    NormalizedRequest,
    TextOutput,
    Thinking,
    ToolCall,
)
from molto.upstream.gemini import GeminiChatCompletionsClient


@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("explicit_reasoning", [False, True])
def test_signature_prefix_does_not_split_pending_reasoning(tool, explicit_reasoning):
    """Ensures signature-only prefixes join pending reasoning while explicit messages stay separate."""
    reasoning = Thinking(content="", summary="Consider this")
    content = (
        ToolCall(tool_call_id="call_1", tool_name="lookup", args="{}")
        if tool
        else TextOutput(content="Answer")
    )
    request = NormalizedRequest(
        model="synthetic",
        input=[
            AssistantInput(content=[reasoning]) if explicit_reasoning else reasoning,
            Thinking(content="", summary="", signature="opaque"),
            AssistantInput(content=[content]),
            AssistantInput(content=[TextOutput(content="Separate")]),
        ],
    )
    original = request.model_copy(deep=True)
    extension = {"extra_content": {"google": {"thought_signature": "opaque"}}}
    message = {
        "role": "assistant",
        **(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                        **extension,
                    }
                ]
            }
            if tool
            else {"content": "Answer", **extension}
        ),
    }
    expected = [message, {"role": "assistant", "content": "Separate"}]
    if explicit_reasoning:
        expected.insert(0, {"role": "assistant", "reasoning_content": "Consider this"})
    else:
        message["reasoning_content"] = "Consider this"
    assert (
        GeminiChatCompletionsClient._native_request(request).model_dump()["messages"] == expected
    )
    assert request == original


@pytest.mark.parametrize("tool", [False, True])
def test_signature_filtering_preserves_reasoning_order_validation(tool):
    """Ensures removing signature-only parts never admits reasoning after text or calls."""
    content = (
        ToolCall(tool_call_id="call_1", tool_name="lookup", args="{}")
        if tool
        else TextOutput(content="Answer")
    )
    request = NormalizedRequest(
        model="synthetic",
        input=[
            AssistantInput(
                content=[
                    Thinking(content="", summary="", signature="opaque"),
                    content,
                    Thinking(content="", summary="Too late"),
                ]
            )
        ],
    )
    with pytest.raises(UnsupportedInput, match="reasoning order"):
        GeminiChatCompletionsClient._native_request(request)
