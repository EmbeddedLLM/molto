from collections.abc import AsyncIterator

import pytest

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.normalized import (
    FunctionToolChoice,
    InputTokenDetails,
    ModelResponseStreamEvent,
    OutputTokenDetails,
    PartDeltaEvent,
    PartStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolReturn,
    Usage,
    UserInput,
)

_CAPABILITIES = PROVIDER_CAPABILITIES["anthropic"]


def _request(**overrides) -> AnthropicMessagesCreateRequest:
    body = {
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "m",
        "stream": True,
    }
    return AnthropicMessagesCreateRequest(**(body | overrides))


def _normalize(request: AnthropicMessagesCreateRequest, capabilities: ProviderCapabilities):
    return AnthropicMessagesAdapter(capabilities).normalize_request(request)


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("Be brief.", ["Be brief."]),
        ([{"type": "text", "text": "One"}, {"type": "text", "text": "Two"}], ["One", "Two"]),
    ],
    ids=["string", "blocks"],
)
def test_normalizes_top_level_system_prompts(system, expected):
    normalized = _normalize(_request(system=system), _CAPABILITIES)

    assert normalized.input[: len(expected)] == [SystemPrompt(content=text) for text in expected]


def test_normalizes_mid_conversation_system_message():
    request = _request(
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": [{"type": "text", "text": "New rules."}]},
            {"role": "assistant", "content": "Understood."},
        ]
    )

    normalized = _normalize(request, _CAPABILITIES)

    assert normalized.input == [
        UserInput(content=[TextInput(content="Hi")]),
        SystemPrompt(content="New rules."),
        TextOutput(content="Understood."),
    ]


def test_rejects_non_text_system_message_blocks():
    request = _request(
        messages=[
            {"role": "user", "content": "Hi"},
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Rules."},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.test/a.png"},
                    },
                ],
            },
        ]
    )

    with pytest.raises(ValueError, match=r"system content blocks: \['image'\]"):
        _normalize(request, _CAPABILITIES)


def test_rejects_unsupported_user_content_block():
    request = _request(
        messages=[
            {
                "role": "user",
                "content": [{"type": "tool_use", "id": "c1", "name": "T", "input": {}}],
            }
        ]
    )

    with pytest.raises(ValueError, match="Unsupported Anthropic user content block: tool_use"):
        _normalize(request, _CAPABILITIES)


def test_rejects_tool_result_after_user_content():
    request = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "tool_result", "tool_use_id": "c1", "content": "done"},
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match="tool results must precede user content"):
        _normalize(request, _CAPABILITIES)


def test_rejects_non_text_tool_result_content():
    request = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://example.test/a.png"},
                            },
                        ],
                    }
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match=r"tool result content blocks: \['image'\]"):
        _normalize(request, _CAPABILITIES)


def test_normalizes_tool_result_before_user_content():
    request = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1"},
                    {"type": "text", "text": "And?"},
                ],
            }
        ]
    )

    normalized = _normalize(request, _CAPABILITIES)

    assert normalized.input == [
        ToolReturn(content=[TextInput(content="")], tool_call_id="c1"),
        UserInput(content=[TextInput(content="And?")]),
    ]


def test_rejects_unsupported_tool_type():
    request = _request(tools=[{"type": "bash_20250124", "name": "bash"}])

    with pytest.raises(ValueError, match="Unsupported Anthropic tool type: 'bash_20250124'"):
        _normalize(request, _CAPABILITIES)


def test_rejects_unsupported_custom_tool_fields():
    request = _request(
        tools=[
            {
                "name": "T",
                "input_schema": {"type": "object", "properties": {}},
                "defer_loading": True,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"custom tool fields: \['cache_control', 'defer_loading'\]",
    ):
        _normalize(request, _CAPABILITIES)


@pytest.mark.parametrize(
    ("tool_choice", "expected_choice", "expected_parallel"),
    [
        ({"type": "auto"}, None, None),
        ({"type": "none"}, "none", None),
        ({"type": "any"}, "required", None),
        ({"type": "tool", "name": "First"}, FunctionToolChoice(name="First"), None),
        ({"type": "auto", "disable_parallel_tool_use": True}, None, False),
    ],
    ids=["auto", "none", "any", "tool", "auto-sequential"],
)
def test_normalizes_tool_choice(tool_choice, expected_choice, expected_parallel):
    normalized = _normalize(_request(tool_choice=tool_choice), _CAPABILITIES)

    assert normalized.tool_choice == expected_choice
    assert normalized.parallel_tool_calls is expected_parallel


async def _stream(*events: ModelResponseStreamEvent) -> AsyncIterator[ModelResponseStreamEvent]:
    for event in events:
        yield event


def _adapter(capabilities: ProviderCapabilities = _CAPABILITIES):
    adapter = AnthropicMessagesAdapter(capabilities)
    request = adapter.normalize_request(_request())
    return adapter, request


_MESSAGE_START = ResponseStartEvent(id="msg_123", created_at=1, model="m")


@pytest.mark.parametrize(
    "delta",
    [ThinkingDelta(content_delta="raw"), ThinkingDelta(summary_delta="summary")],
    ids=["raw", "summary"],
)
async def test_projection_collapses_either_thinking_representation(delta: ThinkingDelta) -> None:
    adapter, request = _adapter()

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _MESSAGE_START,
                PartStartEvent(index=0, part=Thinking(content="", summary="")),
                PartDeltaEvent(index=0, delta=delta),
            ),
            request,
        )
    ]

    deltas = [event for event in events if event.type == "content_block_delta"]
    assert len(deltas) == 1
    assert deltas[0].delta.type == "thinking_delta"
    assert deltas[0].delta.thinking == (delta.content_delta or delta.summary_delta)


async def test_projection_skips_empty_thinking_delta() -> None:
    adapter, request = _adapter()

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _MESSAGE_START,
                PartStartEvent(index=0, part=Thinking(content="", summary="")),
                PartDeltaEvent(index=0, delta=ThinkingDelta()),
            ),
            request,
        )
    ]

    assert [event.type for event in events] == ["message_start", "content_block_start"]


@pytest.mark.parametrize(
    "delta",
    [ToolCallDelta(tool_name_delta="Extra"), ToolCallDelta(tool_call_id="call_new")],
    ids=["name", "id"],
)
async def test_projection_rejects_tool_identity_updates(delta: ToolCallDelta) -> None:
    adapter, request = _adapter()

    with pytest.raises(ValueError, match="cannot update a tool name or tool call ID"):
        [
            event
            async for event in adapter.project_events(
                _stream(
                    _MESSAGE_START,
                    PartStartEvent(
                        index=0,
                        part=ToolCall(tool_name="First", args="", tool_call_id="call_one"),
                    ),
                    PartDeltaEvent(index=0, delta=delta),
                ),
                request,
            )
        ]


async def test_projection_maps_terminal_usage() -> None:
    adapter, request = _adapter()

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _MESSAGE_START,
                ResponseEndEvent(
                    id="msg_123",
                    created_at=1,
                    model="m",
                    termination_reason="stop",
                    usage=Usage(
                        input_tokens=10,
                        input_tokens_details=InputTokenDetails(
                            cached_tokens=3, cache_write_tokens=2
                        ),
                        output_tokens=7,
                        output_tokens_details=OutputTokenDetails(reasoning_tokens=4),
                        total_tokens=17,
                    ),
                ),
            ),
            request,
        )
    ]

    usage = next(event for event in events if event.type == "message_delta").usage
    assert usage.input_tokens == 5
    assert usage.cache_creation_input_tokens == 2
    assert usage.cache_read_input_tokens == 3
    assert usage.output_tokens == 7
    assert usage.output_tokens_details.thinking_tokens == 4
    assert events[-1].type == "message_stop"
