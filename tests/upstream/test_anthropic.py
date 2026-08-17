from collections.abc import AsyncIterator
from time import time

import httpx
import pytest

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesStreamEvent,
    AnthropicMessagesStreamEventAdapter,
)
from molto.types.normalized import (
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
    UserInput,
)
from molto.upstream.anthropic import AnthropicMessagesClient


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
        client = AnthropicMessagesClient(http_client)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [
            event
            async for event in client.events(
                NormalizedRequest(input="Hello", max_output_tokens=10, model="model", stream=True)
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


async def test_client_maps_streamed_signatures_to_ir(monkeypatch: pytest.MonkeyPatch):
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
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Thought"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "opaque-state"},
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
        client = AnthropicMessagesClient(http_client)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        events = [
            event
            async for event in client.events(
                NormalizedRequest(
                    input="Hello",
                    max_output_tokens=10,
                    model="model",
                    stream=True,
                )
            )
        ]

    part_start = next(event for event in events if isinstance(event, PartStartEvent))
    signature_delta = next(
        event
        for event in events
        if isinstance(event, PartDeltaEvent)
        and isinstance(event.delta, ThinkingDelta)
        and event.delta.signature_delta is not None
    )
    part_end = next(event for event in events if isinstance(event, PartEndEvent))
    assert isinstance(part_start.part, Thinking)
    assert part_start.part.signature == ""
    assert isinstance(signature_delta.delta, ThinkingDelta)
    assert signature_delta.delta.signature_delta == "opaque-state"
    assert isinstance(part_end.part, Thinking)
    assert part_end.part.signature == "opaque-state"


def test_adapter_rejects_omitted_thinking_display():
    request = AnthropicMessagesCreateRequest(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="model",
        stream=True,
        thinking={"type": "adaptive", "display": "omitted"},
    )

    with pytest.raises(ValueError, match="summarized display"):
        AnthropicMessagesAdapter().normalize_request(request)


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

    normalized_request = AnthropicMessagesAdapter().normalize_request(request)

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
            input="Hello",
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

    normalized = AnthropicMessagesAdapter().normalize_request(request)

    assert normalized.input == [
        UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")]),
        UserInput(content=[TextInput(content="Follow-up")]),
        TextOutput(content="Answer"),
    ]


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
            input="Hello",
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


def test_client_disables_thinking_without_output_effort():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input="Hello",
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
                input="Hello",
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

    normalized_request = AnthropicMessagesAdapter().normalize_request(request)

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
    adapter = AnthropicMessagesAdapter()
    normalized_request = adapter.normalize_request(request)

    async def normalized_events():
        yield ResponseStartEvent(id="msg_123", model="model")
        yield PartStartEvent(index=0, part=Thinking(content="", summary=""))
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(content_delta="Thought"))
        yield PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="opaque-state"))
        yield PartEndEvent(
            index=0,
            part=Thinking(content="Thought", summary="", signature="opaque-state"),
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
    signature_delta = next(
        event
        for event in events
        if event.type == "content_block_delta" and event.delta.type == "signature_delta"
    )
    assert signature_delta.delta.signature == "opaque-state"


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
        client = AnthropicMessagesClient(http_client)
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
    adapter = AnthropicMessagesAdapter()
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
    adapter = AnthropicMessagesAdapter()
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
