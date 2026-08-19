from collections.abc import AsyncIterator
from time import time

import httpx
import pytest
from pydantic import ValidationError

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesStreamEvent,
    AnthropicMessagesStreamEventAdapter,
)
from molto.types.normalized import (
    FunctionTool,
    FunctionToolChoice,
    ImageUrlInput,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    Reasoning,
    ResponseEndEvent,
    ResponseStartEvent,
    TerminationReason,
    TextInput,
    TextOutput,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolReturn,
    UserInput,
)
from molto.upstream.anthropic import AnthropicMessagesClient

VLLM_CAPABILITIES = PROVIDER_CAPABILITIES["vllm"]


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("vllm", Thinking(content="Reasoning", summary="")),
        ("anthropic", Thinking(content="", summary="Reasoning")),
    ],
)
def test_adapter_classifies_ambiguous_thinking_history(
    provider: str,
    expected: Thinking,
) -> None:
    request = AnthropicMessagesCreateRequest(
        max_tokens=100,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Reasoning"},
                    {"type": "text", "text": "Answer"},
                ],
            }
        ],
        model="m",
        stream=True,
    )

    normalized = AnthropicMessagesAdapter(PROVIDER_CAPABILITIES[provider]).normalize_request(
        request
    )

    assert normalized.input == [expected, TextOutput(content="Answer")]


async def test_events_merge_start_usage_without_mutating_started_parts(
    monkeypatch: pytest.MonkeyPatch,
):
    native_events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 5,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "output_tokens": 0,
                    "service_tier": "priority",
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]

    async def stub_native_events(
        native_request: AnthropicMessagesCreateRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        for event in native_events:
            yield AnthropicMessagesStreamEventAdapter.validate_python(event)

    before = int(time())
    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [
            event
            async for event in client.events(
                NormalizedRequest(
                    input=[UserInput(content=[TextInput(content="Hello")])],
                    max_output_tokens=10,
                    model="model",
                    stream=True,
                )
            )
        ]

    part_start = next(event for event in events if isinstance(event, PartStartEvent))
    part_end = next(event for event in events if isinstance(event, PartEndEvent))
    response_start = next(event for event in events if isinstance(event, ResponseStartEvent))
    response_end = next(event for event in events if isinstance(event, ResponseEndEvent))
    assert before <= response_start.created_at <= int(time())
    assert response_end.created_at == response_start.created_at
    assert response_start.service_tier == "priority"
    assert response_end.service_tier == "priority"
    assert response_end.termination_reason == "stop"
    assert response_end.stop_sequence is None
    assert part_start.part.content == ""
    assert part_end.part.content == "Hi"
    assert response_end.usage is not None
    assert response_end.usage.input_tokens == 10
    assert response_end.usage.input_tokens_details.cached_tokens == 3
    assert response_end.usage.input_tokens_details.cache_write_tokens == 2


@pytest.mark.parametrize(
    ("capabilities", "thinking_delta", "thinking"),
    [
        (
            VLLM_CAPABILITIES,
            ThinkingDelta(content_delta="Thought"),
            Thinking(content="Thought", summary="", signature="replacement-state"),
        ),
        (
            PROVIDER_CAPABILITIES["anthropic"],
            ThinkingDelta(summary_delta="Thought"),
            Thinking(content="", summary="Thought", signature="replacement-state"),
        ),
    ],
)
async def test_client_maps_streamed_signatures_to_ir(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: ProviderCapabilities,
    thinking_delta: ThinkingDelta,
    thinking: Thinking,
):
    native_events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "initial-state",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Thought"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "replacement-state"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]

    async def stub_native_events(
        native_request: AnthropicMessagesCreateRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        for event in native_events:
            yield AnthropicMessagesStreamEventAdapter.validate_python(event)

    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(http_client, capabilities=capabilities)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [
            event
            async for event in client.events(
                NormalizedRequest(
                    input=[UserInput(content=[TextInput(content="Hello")])],
                    max_output_tokens=10,
                    model="model",
                    stream=True,
                )
            )
        ]

    part_start = next(event for event in events if isinstance(event, PartStartEvent))
    text_delta = next(
        event
        for event in events
        if isinstance(event, PartDeltaEvent)
        and isinstance(event.delta, ThinkingDelta)
        and event.delta.signature_delta is None
    )
    signature_delta = next(
        event
        for event in events
        if isinstance(event, PartDeltaEvent)
        and isinstance(event.delta, ThinkingDelta)
        and event.delta.signature_delta is not None
    )
    part_end = next(event for event in events if isinstance(event, PartEndEvent))
    assert isinstance(part_start.part, Thinking)
    assert part_start.part.signature == "initial-state"
    assert text_delta.delta == thinking_delta
    assert isinstance(signature_delta.delta, ThinkingDelta)
    assert signature_delta.delta.signature_delta == "replacement-state"
    assert isinstance(part_end.part, Thinking)
    assert part_end.part == thinking


def test_adapter_rejects_omitted_thinking_display():
    request = AnthropicMessagesCreateRequest(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="model",
        stream=True,
        thinking={"type": "adaptive", "display": "omitted"},
    )

    with pytest.raises(ValueError, match="summarized display"):
        AnthropicMessagesAdapter(VLLM_CAPABILITIES).normalize_request(request)


@pytest.mark.parametrize(
    ("thinking", "output_config", "effort"),
    [
        ({"type": "adaptive", "display": "summarized"}, None, "high"),
        (
            {"type": "adaptive", "display": "summarized"},
            {"effort": "medium"},
            "medium",
        ),
        (
            {"type": "enabled", "budget_tokens": 2048, "display": "summarized"},
            None,
            "low",
        ),
        ({"type": "disabled"}, None, "none"),
    ],
)
def test_adapter_maps_thinking_configuration_to_reasoning_effort(
    thinking: dict,
    output_config: dict | None,
    effort: str,
):
    request = AnthropicMessagesCreateRequest(
        max_tokens=20_000,
        messages=[
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Prior thought"},
                    {"type": "text", "text": "Prior answer"},
                ],
            },
            {"role": "user", "content": "Follow-up"},
        ],
        model="model",
        output_config=output_config,
        stream=True,
        thinking=thinking,
    )

    normalized_request = AnthropicMessagesAdapter(VLLM_CAPABILITIES).normalize_request(request)

    assert normalized_request.reasoning is not None
    assert normalized_request.reasoning.effort == effort


@pytest.mark.parametrize(
    ("model", "effort", "thinking", "output_config"),
    [
        (
            "deepseek-v4-flash",
            "high",
            {"type": "adaptive", "display": "summarized"},
            {"effort": "high"},
        ),
        (
            "claude-sonnet-4-6",
            "minimal",
            {"type": "adaptive", "display": "summarized"},
            {"effort": "low"},
        ),
        (
            "claude-opus-4-6",
            "xhigh",
            {"type": "adaptive", "display": "summarized"},
            {"effort": "max"},
        ),
        (
            "claude-opus-4-7",
            "xhigh",
            {"type": "adaptive", "display": "summarized"},
            {"effort": "xhigh"},
        ),
        (
            "claude-opus-5",
            "max",
            {"type": "adaptive", "display": "summarized"},
            {"effort": "max"},
        ),
        (
            "claude-sonnet-4-5-20250929",
            "medium",
            {"type": "enabled", "budget_tokens": 10_000, "display": "summarized"},
            None,
        ),
    ],
)
def test_client_selects_model_supported_thinking_configuration(
    model: str,
    effort: str,
    thinking: dict,
    output_config: dict | None,
):
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[UserInput(content=[TextInput(content="Hello")])],
            max_output_tokens=40_000,
            model=model,
            reasoning=Reasoning(effort=effort),
            stream=True,
        )
    )

    assert body["thinking"] == thinking
    assert body.get("output_config") == output_config


def test_client_projects_user_text_as_content_block_list():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[
                UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")]),
                UserInput(content=[TextInput(content="Follow-up")]),
                TextOutput(content="Answer"),
            ],
            max_output_tokens=2048,
            model="compatible-model",
            stream=True,
        )
    )

    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "Follow-up"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
    ]


def test_adapter_preserves_user_message_and_part_boundaries():
    request = AnthropicMessagesCreateRequest(
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "text", "text": "Part two"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Follow-up"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
        ],
        model="compatible-model",
        stream=True,
    )

    normalized = AnthropicMessagesAdapter(VLLM_CAPABILITIES).normalize_request(request)

    assert normalized.input == [
        UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")]),
        UserInput(content=[TextInput(content="Follow-up")]),
        TextOutput(content="Answer"),
    ]


def test_adapter_normalizes_image_blocks_as_data_urls():
    request = AnthropicMessagesCreateRequest(
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AA==",
                        },
                    },
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.test/b.png"},
                    },
                    {"type": "text", "text": "after"},
                ],
            }
        ],
        model="compatible-model",
        stream=True,
    )

    normalized = AnthropicMessagesAdapter(VLLM_CAPABILITIES).normalize_request(request)

    assert normalized.input == [
        UserInput(
            content=[
                TextInput(content="before"),
                ImageUrlInput(url="data:image/png;base64,AA=="),
                ImageUrlInput(url="https://example.test/b.png"),
                TextInput(content="after"),
            ]
        )
    ]


def test_request_rejects_file_image_source():
    with pytest.raises(ValidationError):
        AnthropicMessagesCreateRequest(
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "file", "file_id": "file_123"}}
                    ],
                }
            ],
            model="compatible-model",
            stream=True,
        )


def test_client_projects_image_parts_as_image_blocks():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[
                UserInput(
                    content=[
                        TextInput(content="before"),
                        ImageUrlInput(url="data:image/png;base64,AA=="),
                        ImageUrlInput(url="https://example.test/b.png"),
                        TextInput(content="after"),
                    ]
                )
            ],
            max_output_tokens=2048,
            model="compatible-model",
            stream=True,
        )
    )

    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
                },
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.test/b.png"},
                },
                {"type": "text", "text": "after"},
            ],
        }
    ]


@pytest.mark.parametrize("detail", ["low", "high", "original"])
def test_client_rejects_unrepresentable_image_detail(detail: str):
    request = NormalizedRequest(
        input=[
            UserInput(content=[ImageUrlInput(url="data:image/png;base64,AA==", detail=detail)])
        ],
        max_output_tokens=2048,
        model="compatible-model",
        stream=True,
    )

    with pytest.raises(ValueError, match="cannot represent image detail"):
        AnthropicMessagesClient._request_body(request)


def test_client_rejects_non_base64_data_url_image():
    request = NormalizedRequest(
        input=[UserInput(content=[ImageUrlInput(url="data:image/png,%89NP")])],
        max_output_tokens=2048,
        model="compatible-model",
        stream=True,
    )

    with pytest.raises(ValueError, match="base64 data URL"):
        AnthropicMessagesClient._request_body(request)


@pytest.mark.parametrize(
    ("effort", "budget_tokens"),
    [
        ("minimal", 1024),
        ("low", 2048),
        ("medium", 10_000),
        ("high", 16_384),
        ("xhigh", 32_768),
        ("max", 32_768),
    ],
)
def test_client_maps_effort_to_extended_thinking_budget(
    effort: str,
    budget_tokens: int,
):
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[UserInput(content=[TextInput(content="Hello")])],
            max_output_tokens=40_000,
            model="claude-sonnet-4-5-20250929",
            reasoning=Reasoning(effort=effort),
            stream=True,
        )
    )

    assert body["thinking"] == {
        "type": "enabled",
        "budget_tokens": budget_tokens,
        "display": "summarized",
    }
    assert "output_config" not in body


def test_client_enables_adaptive_thinking_for_reasoning_history():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[Thinking(content="Prior thought", summary=""), TextOutput(content="Answer")],
            max_output_tokens=2048,
            model="compatible-model",
            stream=True,
        )
    )

    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}


def test_client_uses_summary_when_reasoning_history_omits_raw_content():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[
                Thinking(content="", summary="Prior summary", signature="opaque-state"),
                TextOutput(content="Prior answer"),
            ],
            max_output_tokens=2048,
            model="compatible-model",
            stream=True,
        )
    )

    assert body["messages"][0]["content"][0] == {
        "type": "thinking",
        "thinking": "Prior summary",
        "signature": "opaque-state",
    }


def test_client_disables_thinking_without_output_effort():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[UserInput(content=[TextInput(content="Hello")])],
            max_output_tokens=2048,
            model="compatible-model",
            reasoning=Reasoning(effort="none"),
            stream=True,
        )
    )

    assert body["thinking"] == {"type": "disabled"}
    assert "output_config" not in body


def test_client_rejects_extended_thinking_budget_at_output_limit():
    with pytest.raises(ValueError, match="must be less than max_output_tokens"):
        AnthropicMessagesClient._request_body(
            NormalizedRequest(
                input=[UserInput(content=[TextInput(content="Hello")])],
                max_output_tokens=10_000,
                model="claude-sonnet-4-5-20250929",
                reasoning=Reasoning(effort="medium"),
                stream=True,
            )
        )


def test_adapter_maps_request_signatures_to_ir():
    request = AnthropicMessagesCreateRequest(
        max_tokens=10,
        messages=[
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Prior thought",
                        "signature": "opaque-state",
                    },
                    {"type": "text", "text": "Prior answer"},
                ],
            },
            {"role": "user", "content": "Follow-up"},
        ],
        model="model",
        stream=True,
    )

    normalized_request = AnthropicMessagesAdapter(VLLM_CAPABILITIES).normalize_request(request)

    assert isinstance(normalized_request.input, list)
    thinking = next(item for item in normalized_request.input if isinstance(item, Thinking))
    assert thinking.signature == "opaque-state"


async def test_adapter_maps_same_provider_opaque_reasoning_state():
    request = AnthropicMessagesCreateRequest(
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello"}],
        model="model",
        stream=True,
    )
    adapter = AnthropicMessagesAdapter(PROVIDER_CAPABILITIES["openai"])
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="msg_123", model="model")
        yield PartStartEvent(
            index=0,
            part=Thinking(content="", summary="", signature="initial-state"),
        )
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="Thought"))
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="replacement-state"))
        yield PartEndEvent(
            index=0,
            part=Thinking(content="", summary="Thought", signature="replacement-state"),
        )
        yield ResponseEndEvent(
            id="msg_123",
            created_at=1,
            model="model",
            termination_reason="stop",
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]
    thinking_start = next(
        event
        for event in events
        if event.type == "content_block_start" and event.content_block.type == "thinking"
    )
    signature_delta = next(
        event
        for event in events
        if event.type == "content_block_delta" and event.delta.type == "signature_delta"
    )
    thinking_delta = next(
        event
        for event in events
        if event.type == "content_block_delta" and event.delta.type == "thinking_delta"
    )
    assert thinking_delta.delta.thinking == "Thought"
    assert thinking_start.content_block.signature == "initial-state"
    assert signature_delta.delta.signature == "replacement-state"


async def test_client_forwards_same_provider_opaque_reasoning_state(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_requests: list[AnthropicMessagesCreateRequest] = []

    async def stub_native_events(
        native_request: AnthropicMessagesCreateRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        captured_requests.append(native_request)
        if False:
            yield

    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [
            event
            async for event in client.events(
                NormalizedRequest(
                    input=[
                        Thinking(content="Prior thought", summary="", signature="opaque-state"),
                        TextOutput(content="Prior answer"),
                    ],
                    max_output_tokens=10,
                    model="model",
                    stream=True,
                )
            )
        ]

    assert events == []
    content = captured_requests[0].model_dump()["messages"][0]["content"]
    assert content[0]["signature"] == "opaque-state"


@pytest.mark.parametrize(
    ("termination_reason", "stop_reason"),
    [
        ("max_output_tokens", "max_tokens"),
        ("model_context_window_exceeded", "model_context_window_exceeded"),
        ("pause", "pause_turn"),
        ("refusal", "refusal"),
        ("stop", "end_turn"),
        ("stop_sequence", "stop_sequence"),
        ("tool_call", "tool_use"),
    ],
)
async def test_adapter_uses_standard_service_tier_and_projects_termination_reason(
    termination_reason: TerminationReason,
    stop_reason: str,
):
    request = AnthropicMessagesCreateRequest(
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello"}],
        model="model",
        stream=True,
    )
    adapter = AnthropicMessagesAdapter(VLLM_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="msg_123", model="model", service_tier="auto")
        yield ResponseEndEvent(
            id="msg_123",
            created_at=1,
            model="model",
            service_tier="auto",
            termination_reason=termination_reason,
            stop_sequence="END" if termination_reason == "stop_sequence" else None,
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]
    message_start = events[0]
    message_delta = events[1]
    assert message_start.type == "message_start"
    assert message_start.message.usage.service_tier == "standard"
    assert message_delta.type == "message_delta"
    assert message_delta.delta.stop_reason == stop_reason
    assert message_delta.delta.stop_sequence == (
        "END" if termination_reason == "stop_sequence" else None
    )


async def test_adapter_coerces_unrepresentable_content_filter_termination_to_end_turn():
    request = AnthropicMessagesCreateRequest(
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello"}],
        model="model",
        stream=True,
    )
    adapter = AnthropicMessagesAdapter(VLLM_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseEndEvent(
            id="msg_123",
            created_at=1,
            model="model",
            termination_reason="content_filter",
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    message_delta = next(event for event in events if event.type == "message_delta")
    assert message_delta.delta.stop_reason == "end_turn"


def test_adapter_normalizes_function_tools_and_tool_history():
    request = AnthropicMessagesCreateRequest(
        max_tokens=2048,
        messages=[
            {"role": "user", "content": "Run both tools."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running them."},
                    {
                        "type": "tool_use",
                        "id": "call_one",
                        "name": "First",
                        "input": {"value": 1},
                        "caller": {"type": "direct"},
                    },
                    {
                        "type": "tool_use",
                        "id": "call_two",
                        "name": "Second",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_one",
                        "content": "one",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_two",
                        "content": [{"type": "text", "text": "two"}],
                        "is_error": True,
                    },
                ],
            },
        ],
        model="model",
        stream=True,
        tool_choice={"type": "any", "disable_parallel_tool_use": True},
        tools=[
            {
                "name": "First",
                "description": "First tool",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "Second",
                "input_schema": {"type": "object", "properties": {}},
                "strict": True,
            },
        ],
    )

    normalized = AnthropicMessagesAdapter(VLLM_CAPABILITIES).normalize_request(request)

    assert normalized.input == [
        UserInput(content=[TextInput(content="Run both tools.")]),
        TextOutput(content="Running them."),
        ToolCall(tool_name="First", args='{"value":1}', tool_call_id="call_one"),
        ToolCall(tool_name="Second", args="{}", tool_call_id="call_two"),
        ToolReturn(
            content=[TextInput(content="one")],
            tool_call_id="call_one",
        ),
        ToolReturn(
            content=[TextInput(content="two")],
            tool_call_id="call_two",
            outcome="failed",
        ),
    ]
    assert normalized.tools == [
        FunctionTool(
            name="First",
            description="First tool",
            parameters={"type": "object", "properties": {}},
        ),
        FunctionTool(
            name="Second",
            parameters={"type": "object", "properties": {}},
            strict=True,
        ),
    ]
    assert normalized.tool_choice == "required"
    assert normalized.parallel_tool_calls is False


def test_adapter_rejects_server_generated_tool_history():
    request = AnthropicMessagesCreateRequest(
        max_tokens=2048,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_one",
                        "name": "First",
                        "input": {},
                        "caller": {
                            "type": "code_execution_20250825",
                            "tool_id": "server_tool",
                        },
                    }
                ],
            }
        ],
        model="model",
        stream=True,
    )

    with pytest.raises(ValueError, match="directly invoked"):
        AnthropicMessagesAdapter(VLLM_CAPABILITIES).normalize_request(request)


def test_client_projects_function_tools_and_tool_history_without_caller():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[
                UserInput(content=[TextInput(content="Run both tools.")]),
                TextOutput(content="Running them."),
                ToolCall(tool_name="First", args='{"value":1}', tool_call_id="call_one"),
                ToolCall(tool_name="Second", args="{}", tool_call_id="call_two"),
                ToolReturn(content=[TextInput(content="one")], tool_call_id="call_one"),
                ToolReturn(
                    content=[TextInput(content="two")],
                    tool_call_id="call_two",
                    outcome="failed",
                ),
            ],
            max_output_tokens=2048,
            model="model",
            parallel_tool_calls=False,
            stream=True,
            tool_choice=FunctionToolChoice(name="First"),
            tools=[
                FunctionTool(
                    name="First",
                    description="First tool",
                    parameters={"type": "object", "properties": {}},
                )
            ],
        )
    )

    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Run both tools."}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Running them."},
                {
                    "type": "tool_use",
                    "id": "call_one",
                    "name": "First",
                    "input": {"value": 1},
                },
                {
                    "type": "tool_use",
                    "id": "call_two",
                    "name": "Second",
                    "input": {},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_one",
                    "content": [{"type": "text", "text": "one"}],
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_two",
                    "content": [{"type": "text", "text": "two"}],
                    "is_error": True,
                },
            ],
        },
    ]
    assert body["tools"] == [
        {
            "name": "First",
            "description": "First tool",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    assert body["tool_choice"] == {
        "type": "tool",
        "name": "First",
        "disable_parallel_tool_use": True,
    }


@pytest.mark.parametrize("tool_choice", ["required", FunctionToolChoice(name="First")])
def test_client_rejects_required_tool_choice_without_tools(tool_choice):
    with pytest.raises(ValueError, match="needs at least one tool"):
        AnthropicMessagesClient._request_body(
            NormalizedRequest(model="model", tool_choice=tool_choice)
        )


def test_client_omits_parallel_tool_choice_without_tools():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            max_output_tokens=10,
            model="model",
            parallel_tool_calls=False,
        )
    )

    assert "tool_choice" not in body


@pytest.mark.parametrize("outcome", ["denied", "interrupted"])
def test_client_rejects_unrepresentable_tool_result_outcomes(outcome):
    request = NormalizedRequest(
        input=[
            ToolReturn(
                content=[TextInput(content="result")],
                tool_call_id="call_one",
                outcome=outcome,
            )
        ],
        model="model",
    )

    with pytest.raises(ValueError, match="cannot preserve denied or interrupted"):
        AnthropicMessagesClient._request_body(request)


async def test_client_normalizes_streamed_tool_call(monkeypatch: pytest.MonkeyPatch):
    native_events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_one",
                "name": "First",
                "input": {},
                "caller": {"type": "direct"},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"value"'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": ":1}"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]

    async def stub_native_events(
        native_request: AnthropicMessagesCreateRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        for event in native_events:
            yield AnthropicMessagesStreamEventAdapter.validate_python(event)

    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [
            event
            async for event in client.events(
                NormalizedRequest(
                    input=[UserInput(content=[TextInput(content="Run it.")])],
                    max_output_tokens=10,
                    model="model",
                    stream=True,
                )
            )
        ]

    assert events[1] == PartStartEvent(
        index=0,
        part=ToolCall(tool_name="First", args="", tool_call_id="call_one"),
    )
    assert events[2:4] == [
        PartDeltaEvent(index=0, delta=ToolCallDelta(args_delta='{"value"')),
        PartDeltaEvent(index=0, delta=ToolCallDelta(args_delta=":1}")),
    ]
    assert events[4] == PartEndEvent(
        index=0,
        part=ToolCall(tool_name="First", args='{"value":1}', tool_call_id="call_one"),
    )
    assert isinstance(events[-1], ResponseEndEvent)
    assert events[-1].termination_reason == "tool_call"


async def test_adapter_projects_tool_call_with_direct_caller():
    request = AnthropicMessagesCreateRequest(
        max_tokens=10,
        messages=[{"role": "user", "content": "Run it."}],
        model="model",
        stream=True,
    )
    adapter = AnthropicMessagesAdapter(VLLM_CAPABILITIES)
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="msg_123", model="model")
        yield PartStartEvent(
            index=0,
            part=ToolCall(tool_name="First", args='{"value"', tool_call_id="call_one"),
        )
        yield PartDeltaEvent(index=0, delta=ToolCallDelta(args_delta=":1}"))
        yield PartEndEvent(
            index=0,
            part=ToolCall(tool_name="First", args='{"value":1}', tool_call_id="call_one"),
        )
        yield ResponseEndEvent(
            id="msg_123",
            created_at=1,
            model="model",
            termination_reason="tool_call",
        )

    events = [
        event async for event in adapter.project_events(normalized_events(), normalized_request)
    ]

    tool_start = next(event for event in events if event.type == "content_block_start")
    assert tool_start.content_block.model_dump(exclude_unset=True) == {
        "type": "tool_use",
        "id": "call_one",
        "name": "First",
        "input": {},
        "caller": {"type": "direct"},
    }
    assert [
        event.delta.partial_json for event in events if event.type == "content_block_delta"
    ] == ['{"value"', ":1}"]
    message_delta = next(event for event in events if event.type == "message_delta")
    assert message_delta.delta.stop_reason == "tool_use"
