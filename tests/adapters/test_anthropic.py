from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic
from anthropic import NotFoundError as AnthropicNotFoundError
from anthropic.lib.streaming._messages import accumulate_event

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesStreamEvent,
    AnthropicMessagesStreamEventAdapter,
)
from molto.types.exceptions import ModelNotFoundError, UnsupportedInput
from molto.types.normalized import (
    AssistantInput,
    CacheBreakpoint,
    FunctionToolChoice,
    ImageUrlInput,
    InputTokenDetails,
    ModelResponseStreamEvent,
    OutputTokenDetails,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    PromptCache,
    ResponseEndEvent,
    ResponseStartEvent,
    SystemPrompt,
    TextDelta,
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
from molto.utils.io import read_yaml

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


def test_normalization_rejects_unsigned_summary_history() -> None:
    request = _request(
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Summary"}],
            }
        ]
    )

    with pytest.raises(UnsupportedInput, match="summary requires a signature"):
        _normalize(request, _CAPABILITIES)


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
    """Ensures system messages and surrounding message boundaries survive normalization."""
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
        AssistantInput(content=[TextOutput(content="Understood.")]),
    ]


def test_normalizes_assistant_messages_with_ordered_parts_and_boundaries():
    """Ensures Messages keeps mixed blocks, adjacent text, empty and reasoning-only messages."""
    normalized = _normalize(
        _request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Plan", "signature": "sig_1"},
                        {
                            "type": "text",
                            "text": "First",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "Second"},
                        {"type": "tool_use", "id": "call_1", "name": "weather", "input": {}},
                        {"type": "text", "text": "After call"},
                    ],
                },
                {"role": "assistant", "content": "Follow-up"},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "Next", "signature": "sig_2"}],
                },
                {"role": "assistant", "content": []},
            ]
        ),
        _CAPABILITIES,
    )

    assert normalized.input == [
        AssistantInput(
            content=[
                Thinking(content="", summary="Plan", signature="sig_1"),
                TextOutput(content="First", cache_breakpoint=CacheBreakpoint()),
                TextOutput(content="Second"),
                ToolCall(tool_name="weather", args="{}", tool_call_id="call_1"),
                TextOutput(content="After call"),
            ]
        ),
        AssistantInput(content=[TextOutput(content="Follow-up")]),
        AssistantInput(content=[Thinking(content="", summary="Next", signature="sig_2")]),
        AssistantInput(content=[]),
    ]
    assert list(normalized.iter_cache_breakpoints()) == [CacheBreakpoint()]


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

    with pytest.raises(UnsupportedInput, match=r"system content blocks: \['image'\]"):
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

    with pytest.raises(
        UnsupportedInput, match="Unsupported Anthropic user content block: tool_use"
    ):
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

    with pytest.raises(UnsupportedInput, match="tool results must precede user content"):
        _normalize(request, _CAPABILITIES)


def test_normalizes_image_tool_result_content():
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

    normalized = _normalize(request, _CAPABILITIES)

    assert normalized.input == [
        ToolReturn(
            content=[
                TextInput(content="ok"),
                ImageUrlInput(url="https://example.test/a.png"),
            ],
            tool_call_id="c1",
        )
    ]


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

    with pytest.raises(UnsupportedInput, match="Unsupported Anthropic tool type: 'bash_20250124'"):
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


_EPHEMERAL = {"type": "ephemeral"}


def test_normalizes_top_level_cache_control_as_implicit_prompt_cache():
    """Ensures automatic cache intent and lifetimes survive cross-provider normalization."""
    for cache_control, expected in (
        (_EPHEMERAL, PromptCache(mode="implicit")),
        ({"type": "ephemeral", "ttl": "1h"}, PromptCache(mode="implicit", ttl="1h")),
    ):
        normalized = _normalize(_request(cache_control=cache_control), _CAPABILITIES)
        assert normalized.prompt_cache == expected


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {"system": [{"type": "text", "text": "s", "cache_control": _EPHEMERAL}]},
            id="system",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "u", "cache_control": _EPHEMERAL}],
                    }
                ]
            },
            id="user-text",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://example.com/a.png"},
                                "cache_control": _EPHEMERAL,
                            }
                        ],
                    }
                ]
            },
            id="image",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "c1",
                                "content": "result",
                                "cache_control": _EPHEMERAL,
                            }
                        ],
                    }
                ]
            },
            id="tool-result",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "a", "cache_control": _EPHEMERAL}],
                    }
                ]
            },
            id="assistant-text",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "c1",
                                "name": "lookup",
                                "input": {},
                                "cache_control": _EPHEMERAL,
                            }
                        ],
                    }
                ]
            },
            id="tool-use",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "thought",
                                "signature": "opaque-state",
                                "cache_control": _EPHEMERAL,
                            }
                        ],
                    }
                ]
            },
            id="thinking",
        ),
    ],
)
def test_content_block_cache_control(overrides):
    """Ensures portable markers survive and unrepresentable cache boundaries are rejected."""
    block = (overrides.get("system") or overrides["messages"][0]["content"])[0]
    if block["type"] in {"tool_result", "tool_use", "thinking"}:
        with pytest.raises(
            UnsupportedInput, match="cache_control is not supported for translation"
        ):
            _normalize(_request(**overrides), _CAPABILITIES)
    else:
        normalized = _normalize(_request(**overrides), _CAPABILITIES)
        assert list(normalized.iter_cache_breakpoints()) == [CacheBreakpoint()]
        assert normalized.prompt_cache == PromptCache(mode="explicit")


def test_rejects_untranslated_request_fields():
    with pytest.raises(
        UnsupportedInput, match=r"Anthropic Messages request fields.*stop_sequences"
    ):
        _normalize(_request(stop_sequences=["END"]), _CAPABILITIES)


def test_rejects_untranslated_output_config_fields():
    with pytest.raises(UnsupportedInput, match=r"output_config fields.*format"):
        _normalize(
            _request(output_config={"format": {"type": "json_schema", "schema": {}}}),
            _CAPABILITIES,
        )


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


async def _native_stream(
    *events: AnthropicMessagesStreamEvent,
) -> AsyncIterator[AnthropicMessagesStreamEvent]:
    for event in events:
        yield event


def _adapter(capabilities: ProviderCapabilities = _CAPABILITIES):
    adapter = AnthropicMessagesAdapter(capabilities)
    request = adapter.normalize_request(_request())
    return adapter, request


_MESSAGE_START = ResponseStartEvent(id="msg_123", created_at=1, model="m")
_MESSAGE_END = ResponseEndEvent(id="msg_123", created_at=1, model="m", termination_reason="stop")


@pytest.mark.parametrize(
    ("capabilities", "delta"),
    [
        (PROVIDER_CAPABILITIES["vllm"], ThinkingDelta(content_delta="raw")),
        (_CAPABILITIES, ThinkingDelta(summary_delta="summary")),
    ],
    ids=["raw", "summary"],
)
async def test_projection_collapses_either_thinking_representation(
    capabilities: ProviderCapabilities, delta: ThinkingDelta
) -> None:
    adapter, request = _adapter(capabilities)

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


async def test_projection_supports_official_empty_summary_start() -> None:
    adapter, request = _adapter()

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _MESSAGE_START,
                PartStartEvent(index=0, part=Thinking(content="", summary="")),
                PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Summary")),
                PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="opaque-state")),
                PartEndEvent(
                    index=0,
                    part=Thinking(content="", summary="Summary", signature="opaque-state"),
                ),
            ),
            request,
        )
    ]

    start = next(event for event in events if event.type == "content_block_start")
    deltas = [event.delta for event in events if event.type == "content_block_delta"]
    assert start.content_block.signature == ""
    assert [delta.type for delta in deltas] == ["thinking_delta", "signature_delta"]


async def test_projection_rejects_reasoning_outside_capability() -> None:
    adapter, request = _adapter()

    with pytest.raises(ValueError, match="cannot contain raw reasoning"):
        [
            event
            async for event in adapter.project_events(
                _stream(
                    _MESSAGE_START,
                    PartStartEvent(index=0, part=Thinking(content="", summary="")),
                    PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="raw")),
                ),
                request,
            )
        ]


@pytest.mark.parametrize("signature", [None, ""])
async def test_projection_preserves_unsigned_summary_completion(signature: str | None) -> None:
    """Ensures missing output signatures do not discard streamed reasoning summaries."""
    adapter, request = _adapter()

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _MESSAGE_START,
                PartStartEvent(
                    index=0, part=Thinking(content="", summary="", signature=signature)
                ),
                PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Summary")),
                PartEndEvent(
                    index=0, part=Thinking(content="", summary="Summary", signature=signature)
                ),
                _MESSAGE_END,
            ),
            request,
        )
    ]

    deltas = [event.delta for event in events if event.type == "content_block_delta"]
    assert [delta.type for delta in deltas] == ["thinking_delta"]
    assert deltas[0].thinking == "Summary"
    response = await adapter.project_response(_native_stream(*events), request)
    assert response.content[0].thinking == "Summary"
    assert response.content[0].signature == ""


async def test_projection_preserves_non_empty_part_starts() -> None:
    # A part start may already carry an initial snapshot; the content block
    # preserves it instead of waiting for deltas that may not exist.
    adapter, request = _adapter()

    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _MESSAGE_START,
                PartStartEvent(
                    index=0,
                    part=Thinking(content="", summary="Preface", signature="sig-1"),
                ),
                PartEndEvent(
                    index=0,
                    part=Thinking(content="", summary="Preface", signature="sig-1"),
                ),
                PartStartEvent(index=1, part=TextOutput(content="Answer")),
                PartEndEvent(index=1, part=TextOutput(content="Answer")),
            ),
            request,
        )
    ]

    starts = [event for event in events if event.type == "content_block_start"]
    assert starts[0].content_block.thinking == "Preface"
    assert starts[0].content_block.signature == "sig-1"
    assert starts[1].content_block.text == "Answer"


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


async def test_response_projection_collects_projected_events() -> None:
    adapter, request = _adapter()
    response = await adapter.project_response(
        adapter.project_events(
            _stream(
                _MESSAGE_START,
                PartStartEvent(
                    index=0,
                    part=Thinking(content="", summary="Plan", signature="initial-state"),
                ),
                PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta=" carefully")),
                PartDeltaEvent(
                    index=0,
                    delta=ThinkingDelta(signature_delta="opaque-state"),
                ),
                PartEndEvent(
                    index=0,
                    part=Thinking(
                        content="",
                        summary="Plan carefully",
                        signature="opaque-state",
                    ),
                ),
                PartStartEvent(index=1, part=TextOutput(content="Hel")),
                PartDeltaEvent(index=1, delta=TextDelta(content_delta="lo")),
                PartEndEvent(index=1, part=TextOutput(content="Hello")),
                PartStartEvent(
                    index=2,
                    part=ToolCall(tool_name="weather", args="", tool_call_id="call_1"),
                ),
                PartDeltaEvent(index=2, delta=ToolCallDelta(args_delta='{"city":')),
                PartDeltaEvent(index=2, delta=ToolCallDelta(args_delta='"Paris"}')),
                PartEndEvent(
                    index=2,
                    part=ToolCall(
                        tool_name="weather",
                        args='{"city":"Paris"}',
                        tool_call_id="call_1",
                    ),
                ),
                ResponseEndEvent(
                    id="msg_123",
                    created_at=1,
                    model="m",
                    termination_reason="tool_call",
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
        ),
        request,
    )

    assert response.id == "msg_123"
    assert response.model == "m"
    assert response.role == "assistant"
    assert response.stop_reason == "tool_use"
    assert response.stop_sequence is None
    assert [block.type for block in response.content] == ["thinking", "text", "tool_use"]
    thinking = response.content[0]
    assert thinking.type == "thinking"
    assert thinking.thinking == "Plan carefully"
    assert thinking.signature == "opaque-state"
    text = response.content[1]
    assert text.type == "text"
    assert text.text == "Hello"
    tool = response.content[2]
    assert tool.type == "tool_use"
    assert tool.id == "call_1"
    assert tool.name == "weather"
    assert tool.input == {"city": "Paris"}
    assert response.usage.input_tokens == 5
    assert response.usage.cache_creation_input_tokens == 2
    assert response.usage.cache_read_input_tokens == 3
    assert response.usage.output_tokens == 7
    assert response.usage.output_tokens_details is not None
    assert response.usage.output_tokens_details.thinking_tokens == 4
    assert response.usage.service_tier == "standard"


@pytest.mark.parametrize(
    ("arguments", "expected_input"),
    [
        ("{", {}),
        ('{"city": "Par', {}),
        ('{"city":"Paris","units":"cel', {"city": "Paris"}),
        ('{"cities":["Paris","Lon', {"cities": ["Paris"]}),
    ],
)
async def test_response_projection_preserves_truncated_tool_input_like_sdk(
    arguments, expected_input
) -> None:
    adapter, request = _adapter()
    events = [
        event
        async for event in adapter.project_events(
            _stream(
                _MESSAGE_START,
                PartStartEvent(
                    index=0,
                    part=ToolCall(tool_name="weather", args="", tool_call_id="call_1"),
                ),
                PartDeltaEvent(index=0, delta=ToolCallDelta(args_delta=arguments)),
                PartEndEvent(
                    index=0,
                    part=ToolCall(tool_name="weather", args=arguments, tool_call_id="call_1"),
                ),
                ResponseEndEvent(
                    id="msg_123",
                    created_at=1,
                    model="m",
                    termination_reason="max_output_tokens",
                ),
            ),
            request,
        )
    ]
    expected = None
    for event in events:
        expected = accumulate_event(event=event, current_snapshot=expected)

    response = await adapter.project_response(_native_stream(*events), request)

    assert expected is not None
    assert response.stop_reason == expected.stop_reason == "max_tokens"
    assert response.content[0].input == expected.content[0].input == expected_input


@pytest.mark.parametrize("scenario", ["tool_call", "tool_call_parallel"])
async def test_response_projection_matches_sdk_tool_cassette(scenario) -> None:
    cassette = read_yaml(
        Path(__file__).parents[1]
        / "cassettes"
        / "matched"
        / scenario
        / "anthropic-messages-stream-00.yaml"
    )
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(cassette["response"]["sse"]),
        )
    )
    async with AsyncAnthropic(
        api_key="test-key", http_client=httpx.AsyncClient(transport=transport)
    ) as client:
        async with await client.messages.create(**cassette["request"]["body"]) as stream:
            events = [event async for event in stream]
    expected = None
    for event in events:
        expected = accumulate_event(event=event, current_snapshot=expected)
    adapter, request = _adapter()

    response = await adapter.project_response(_native_stream(*events), request)

    assert expected is not None
    # The SDK attaches its parsing buffer and structured-output state to blocks.
    # It also omits terminal thinking-token usage, which Molto preserves.
    exclude = {
        "content": {"__all__": {"__json_buf", "parsed_output"}},
        "usage": {"output_tokens_details"},
    }
    assert response.model_dump(exclude=exclude) == expected.model_dump(exclude=exclude)
    terminal_usage = next(event.usage for event in events if event.type == "message_delta")
    assert response.usage.output_tokens_details == terminal_usage.output_tokens_details


async def test_response_projection_drains_projected_stream() -> None:
    adapter, request = _adapter()
    exhausted = False

    async def events() -> AsyncIterator[ModelResponseStreamEvent]:
        nonlocal exhausted
        yield _MESSAGE_START
        yield _MESSAGE_END
        exhausted = True

    await adapter.project_response(adapter.project_events(events(), request), request)

    assert exhausted


async def test_response_projection_rejects_missing_message_stop() -> None:
    adapter, request = _adapter()

    with pytest.raises(ValueError, match="ended before the message stop event"):
        await adapter.project_response(
            adapter.project_events(_stream(_MESSAGE_START), request),
            request,
        )


async def test_response_projection_rejects_message_stop_before_final_delta() -> None:
    adapter, request = _adapter()
    message_start = AnthropicMessagesStreamEventAdapter.validate_python(
        {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "m",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
    )
    message_stop = AnthropicMessagesStreamEventAdapter.validate_python({"type": "message_stop"})

    with pytest.raises(ValueError, match="message stop before its final delta"):
        await adapter.project_response(
            _native_stream(message_start, message_stop),
            request,
        )


async def test_native_passthrough_preserves_original_error_class() -> None:
    body = {
        "type": "error",
        "error": {"type": "not_found_error", "message": "model: m"},
        "request_id": "req_123",
    }
    response = httpx.Response(
        404,
        json=body,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    normalized_error: ModelNotFoundError
    try:
        raise AnthropicNotFoundError("model: m", response=response, body=body)
    except AnthropicNotFoundError as error:
        try:
            raise ModelNotFoundError(
                error.message, response=error.response, body=error.body
            ) from error
        except ModelNotFoundError as caught:
            normalized_error = caught

    async def failing_events() -> AsyncIterator[ModelResponseStreamEvent]:
        raise normalized_error
        yield

    # Native passthrough never normalizes a request.
    adapter = AnthropicMessagesAdapter(_CAPABILITIES)
    with pytest.raises(AnthropicNotFoundError) as exc_info:
        [event async for event in adapter.project_native_events(failing_events())]

    assert isinstance(exc_info.value.__cause__, ModelNotFoundError)
    assert exc_info.value is not normalized_error.__cause__
