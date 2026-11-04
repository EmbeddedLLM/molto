"""Synthetic provider history boundaries; these cases do not certify live models."""

from copy import deepcopy

import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.responses import ResponseReasoningItem
from pydantic import ValidationError

from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest


@pytest.mark.parametrize("reasoning", ["", "  Think\n", [], ["", "  Think\n", "\t", ""]])
def test_assistant_reasoning_preserves_native_shape(reasoning):
    """Ensures SDK output reasoning keeps its exact string or ordered array in history."""
    message = ChatCompletionMessage.model_validate(
        {"role": "assistant", "content": "Answer", "reasoning": reasoning}
    ).model_dump(exclude_unset=True)
    request = OpenAIChatCompletionsCreateRequest(model="model", messages=[message])
    expected = {"model": "model", "messages": [message]}
    assert request.model_dump() == expected
    assert request.model_dump() == expected


@pytest.mark.parametrize("reasoning", [None, 1, {}, [1], [None], [["nested"]]])
def test_assistant_reasoning_rejects_invalid_values(reasoning):
    """Ensures the reasoning extension cannot accept nulls or non-string array elements."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="model", messages=[{"role": "assistant", "reasoning": reasoning}]
        )


@pytest.mark.parametrize("role", ["user", "system", "developer", "tool"])
def test_reasoning_remains_assistant_only(role):
    """Ensures reasoning does not weaken validation for unrelated message roles."""
    message = {"role": role, "content": "Text", "reasoning": "Thought"}
    if role == "tool":
        message["tool_call_id"] = "call_1"
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(model="model", messages=[message])


@pytest.mark.parametrize(
    "call",
    [
        {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a":2}'}},
        {"id": "call_2", "type": "custom", "custom": {"name": "code", "input": " x\n"}},
    ],
)
def test_tool_history_is_eager_and_repeatable(call):
    """Ensures function and custom calls survive repeated serialization without lazy iterators."""
    message = {"role": "assistant", "content": None, "tool_calls": [call]}
    request = OpenAIChatCompletionsCreateRequest(model="model", messages=[message])
    expected = {"model": "model", "messages": [message]}
    assert request.model_dump(warnings="error") == expected
    assert request.model_dump(warnings="error") == expected


@pytest.mark.parametrize("kind", ["function", "custom"])
@pytest.mark.parametrize(
    "corruption",
    ["index", "missing_id", "missing_name", "missing_payload", "extra", "nested_extra"],
)
def test_tool_history_rejects_corruption_at_construction(kind, corruption):
    """Ensures malformed call history fails before transport instead of disappearing on dump."""
    payload = "arguments" if kind == "function" else "input"
    call = {"id": "call_1", "type": kind, kind: {"name": "tool", payload: "{}"}}
    if corruption == "index":
        call["index"] = 0
    elif corruption == "missing_id":
        del call["id"]
    elif corruption == "missing_name":
        del call[kind]["name"]
    elif corruption == "missing_payload":
        del call[kind][payload]
    elif corruption == "extra":
        call["unexpected"] = True
    else:
        call[kind]["unexpected"] = True
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="model", messages=[{"role": "assistant", "tool_calls": [call]}]
        )


def google_state(value):
    return {"google": {"thought_signature": value}}


@pytest.mark.parametrize("role", ["user", "system", "developer", "assistant", "tool"])
@pytest.mark.parametrize("parts", [False, True])
def test_google_message_and_part_attachments_remain_independent(role, parts):
    """Ensures opaque message and individual part values remain at their original attachments."""
    message = {"role": role, "content": "Text", "extra_content": google_state(" message\n")}
    if parts:
        message["content"] = [
            {"type": "text", "text": "First", "extra_content": google_state("part-1")},
            {"type": "text", "text": "Second", "extra_content": google_state("part-2")},
        ]
    if role == "tool":
        message["tool_call_id"] = "call_1"
        message["extra_content"]["google"]["parts"] = [
            {
                "file_data": {
                    "mime_type": "application/pdf",
                    "display_name": "report",
                    "file_uri": "gs://example/report.pdf",
                }
            }
        ]
    original = deepcopy(message)
    request = OpenAIChatCompletionsCreateRequest(model="model", messages=[message])
    assert request.model_dump()["messages"] == [original]
    assert request.model_dump()["messages"] == [original]
    assert message == original


def test_google_parallel_calls_preserve_distinct_opaque_state():
    """Ensures each parallel call retains its own signature, identity and argument bytes."""
    message = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": None,
            "extra_content": google_state("message"),
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "add", "arguments": f'{{"a":{i}}}'},
                    "extra_content": google_state(f"call-{i}"),
                }
                for i in range(2)
            ],
        }
    ).model_dump(exclude_unset=True)
    request = OpenAIChatCompletionsCreateRequest(model="model", messages=[message])
    assert request.model_dump(warnings="error")["messages"] == [message]
    assert request.model_dump(warnings="error")["messages"] == [message]


@pytest.mark.parametrize(
    "extension", [{"other": {}}, {"google": {}, "other": {}}, {"google": "bad"}, None]
)
def test_google_extension_rejects_unknown_namespaces(extension):
    """Ensures the extension is a narrow Google envelope rather than arbitrary request extras."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="model",
            messages=[{"role": "user", "content": "Text", "extra_content": extension}],
        )


def test_openrouter_reasoning_format_round_trips_sdk_history():
    """Ensures SDK-retained reasoning format survives repeated native history serialization."""
    item = ResponseReasoningItem.model_validate(
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": " Think\n"}],
            "status": "completed",
            "format": "unknown",
        }
    ).model_dump(exclude_unset=True)
    request = OpenAIResponsesCreateRequest(model="model", input=[item])
    assert request.model_dump()["input"] == [item]
    assert request.model_dump()["input"] == [item]


def test_assistant_null_tools_placeholder_preserves_native_history():
    """Ensures Fireworks' observed null tools placeholder survives repeated native serialization."""
    message = {"role": "assistant", "content": "Answer", "tools": None}
    parsed = ChatCompletionMessage.model_validate(message).model_dump(exclude_unset=True)
    request = OpenAIChatCompletionsCreateRequest(model="model", messages=[parsed])
    assert request.model_dump()["messages"] == [message]
    assert request.model_dump()["messages"] == [message]


@pytest.mark.parametrize("tools", [[], {}, [{"type": "function"}], "", False, 1])
def test_assistant_tools_placeholder_rejects_populated_values(tools):
    """Ensures the observed null placeholder cannot accept tool definitions or unrelated values."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="model", messages=[{"role": "assistant", "content": "Answer", "tools": tools}]
        )


@pytest.mark.parametrize("role", ["user", "system", "developer", "tool"])
def test_tools_placeholder_remains_assistant_only(role):
    """Ensures the Fireworks placeholder does not weaken unrelated message schemas."""
    message = {"role": role, "content": "Text", "tools": None}
    if role == "tool":
        message["tool_call_id"] = "call_1"
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(model="model", messages=[message])


@pytest.mark.parametrize("field", ["tools", "tool_calls"])
def test_null_tools_placeholder_has_no_translated_tool_semantics(field):
    """Ensures empty assistant metadata cannot introduce normalized tool definitions."""
    from molto.adapters.openai import OpenAIChatCompletionsAdapter
    from molto.providers import PROVIDER_CAPABILITIES
    from molto.types.normalized import AssistantInput, TextOutput

    request = OpenAIChatCompletionsCreateRequest(
        model="model", messages=[{"role": "assistant", "content": "Answer", field: None}]
    )
    normalized = OpenAIChatCompletionsAdapter(
        PROVIDER_CAPABILITIES["fireworks"]
    ).normalize_request(request)
    assert normalized.tools is None
    assert normalized.input == [AssistantInput(content=[TextOutput(content="Answer")])]


def test_function_call_null_outer_name_preserves_native_history():
    """Ensures Fireworks' outer null name never replaces the function's actual tool name."""
    call = {
        "id": "call_1",
        "type": "function",
        "name": None,
        "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
    }
    message = {"role": "assistant", "tool_calls": [call]}
    request = OpenAIChatCompletionsCreateRequest(model="model", messages=[message])
    assert request.model_dump()["messages"] == [message]
    assert request.model_dump()["messages"] == [message]


@pytest.mark.parametrize("name", ["add", "", [], {}, False, 1])
def test_function_call_outer_name_rejects_populated_values(name):
    """Ensures the known null placeholder cannot introduce competing function identities."""
    call = {
        "id": "call_1",
        "type": "function",
        "name": name,
        "function": {"name": "add", "arguments": "{}"},
    }
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsCreateRequest(
            model="model", messages=[{"role": "assistant", "tool_calls": [call]}]
        )


def test_null_tool_calls_preserves_native_history():
    """Ensures Fireworks' emitted empty call placeholder survives exact native continuation."""
    body = {
        "model": "test",
        "messages": [{"role": "assistant", "content": "done", "tool_calls": None}],
    }
    request = OpenAIChatCompletionsCreateRequest.model_validate(body)
    assert request.model_dump(exclude_none=True)["messages"] == body["messages"]
