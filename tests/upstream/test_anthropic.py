from collections.abc import AsyncIterator
from time import time

import httpx
import pytest
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic import NotFoundError as AnthropicNotFoundError
from pydantic import ValidationError

from molto.adapters.anthropic import AnthropicMessagesAdapter
from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.anthropic_messages import (
    AnthropicMessagesCreateRequest,
    AnthropicMessagesResponse,
    AnthropicMessagesStreamEvent,
    AnthropicMessagesStreamEventAdapter,
)
from molto.types.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ResourceNotFoundError,
    UnsupportedInput,
)
from molto.types.normalized import (
    AssistantInput,
    CacheBreakpoint,
    FunctionTool,
    FunctionToolChoice,
    ImageUrlInput,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    PromptCache,
    Reasoning,
    ResponseEndEvent,
    ResponseStartEvent,
    StreamOptions,
    SystemPrompt,
    TerminationReason,
    TextDelta,
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


async def test_native_response_uses_anthropic_sdk_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "model",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AnthropicMessagesClient(
            http_client,
            capabilities=PROVIDER_CAPABILITIES["anthropic"],
            api_key="test-api-key",
        )
        request = AnthropicMessagesCreateRequest(
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
            model="model",
            stream=False,
        )
        response = await client.native_response(request)

    assert type(response) is AnthropicMessagesResponse
    assert response.id == "msg_123"


@pytest.mark.parametrize(
    ("provider", "signature", "expected"),
    [
        ("vllm", None, Thinking(content="Reasoning", summary="")),
        (
            "anthropic",
            "opaque-state",
            Thinking(content="", summary="Reasoning", signature="opaque-state"),
        ),
    ],
)
def test_adapter_classifies_ambiguous_thinking_history(
    provider: str,
    signature: str | None,
    expected: Thinking,
) -> None:
    """Ensures Messages thinking is classified inside its original assistant message."""
    thinking_block = {"type": "thinking", "thinking": "Reasoning"}
    if signature is not None:
        thinking_block["signature"] = signature
    request = AnthropicMessagesCreateRequest(
        max_tokens=100,
        messages=[
            {
                "role": "assistant",
                "content": [
                    thinking_block,
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

    assert normalized.input == [AssistantInput(content=[expected, TextOutput(content="Answer")])]


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
    ("capabilities", "thinking_delta", "thinking", "preserves_signature"),
    [
        (
            VLLM_CAPABILITIES,
            ThinkingDelta(content_delta="Thought"),
            Thinking(content="Thought", summary=""),
            False,
        ),
        (
            PROVIDER_CAPABILITIES["anthropic"],
            ThinkingDelta(summary_delta="Thought"),
            Thinking(content="", summary="Thought", signature="replacement-state"),
            True,
        ),
    ],
)
async def test_client_maps_streamed_signatures_to_ir(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: ProviderCapabilities,
    thinking_delta: ThinkingDelta,
    thinking: Thinking,
    preserves_signature: bool,
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
    signature_deltas = [
        event
        for event in events
        if isinstance(event, PartDeltaEvent)
        and isinstance(event.delta, ThinkingDelta)
        and event.delta.signature_delta is not None
    ]
    part_end = next(event for event in events if isinstance(event, PartEndEvent))
    assert isinstance(part_start.part, Thinking)
    assert part_start.part.signature == ("initial-state" if preserves_signature else None)
    assert text_delta.delta == thinking_delta
    assert [event.delta.signature_delta for event in signature_deltas] == (
        ["replacement-state"] if preserves_signature else []
    )
    assert isinstance(part_end.part, Thinking)
    assert part_end.part == thinking


_MESSAGE_START = {
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
}
_MESSAGE_END_EVENTS = [
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
]


async def _collect_client_events(
    monkeypatch: pytest.MonkeyPatch,
    native_events: list[dict],
    capabilities: ProviderCapabilities = VLLM_CAPABILITIES,
) -> list:
    async def stub_native_events(
        native_request: AnthropicMessagesCreateRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        for event in native_events:
            yield AnthropicMessagesStreamEventAdapter.validate_python(event)

    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(http_client, capabilities=capabilities)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        return [
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


@pytest.mark.parametrize(
    ("capabilities", "content_block", "expected_part"),
    [
        (
            PROVIDER_CAPABILITIES["anthropic"],
            {"type": "thinking", "thinking": "", "signature": "initial-state"},
            Thinking(content="", summary="", signature="initial-state"),
        ),
        (
            VLLM_CAPABILITIES,
            {"type": "thinking", "thinking": "Preface", "signature": ""},
            Thinking(content="Preface", summary=""),
        ),
    ],
)
async def test_client_starts_non_empty_thinking_block_immediately(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: ProviderCapabilities,
    content_block: dict,
    expected_part: Thinking,
) -> None:
    # A thinking block that already carries text or a signature must not wait
    # for deltas that may never arrive if the stream stalls or fails.
    async def stub_native_events(
        native_request: AnthropicMessagesCreateRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        yield AnthropicMessagesStreamEventAdapter.validate_python(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": content_block,
            }
        )

    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(http_client, capabilities=capabilities)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        stream = client.events(
            NormalizedRequest(
                input=[UserInput(content=[TextInput(content="Hello")])],
                max_output_tokens=10,
                model="model",
                stream=True,
            )
        )
        first = await anext(stream)
        await stream.aclose()

    assert first == PartStartEvent(index=0, part=expected_part)


async def test_client_defers_completely_empty_thinking_block_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only a completely empty thinking block is deferred; with no further events
    # it produces nothing at all.
    async def stub_native_events(
        native_request: AnthropicMessagesCreateRequest,
    ) -> AsyncIterator[AnthropicMessagesStreamEvent]:
        yield AnthropicMessagesStreamEventAdapter.validate_python(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }
        )

    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(http_client, capabilities=VLLM_CAPABILITIES)
        monkeypatch.setattr(client, "native_events", stub_native_events)
        stream = client.events(
            NormalizedRequest(
                input=[UserInput(content=[TextInput(content="Hello")])],
                max_output_tokens=10,
                model="model",
                stream=True,
            )
        )
        with pytest.raises(StopAsyncIteration):
            await anext(stream)


@pytest.mark.parametrize(
    "delta",
    [
        {"type": "thinking_delta", "thinking": ""},
        {"type": "signature_delta", "signature": ""},
    ],
    ids=["thinking", "signature"],
)
async def test_client_omits_thinking_block_after_empty_deltas(
    monkeypatch: pytest.MonkeyPatch,
    delta: dict,
) -> None:
    # Empty deltas carry nothing: they must not materialize a deferred thinking
    # block, so the block is still omitted at block stop.
    native_events = [
        _MESSAGE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        *_MESSAGE_END_EVENTS,
    ]

    events = await _collect_client_events(
        monkeypatch, native_events, PROVIDER_CAPABILITIES["anthropic"]
    )

    assert [
        event
        for event in events
        if isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent))
    ] == []


async def test_client_omits_completely_empty_thinking_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A thinking block with no text, signature, or deltas carries nothing to
    # translate, so its part lifecycle is omitted from the IR and the surviving
    # parts are reindexed to stay dense.
    native_events = [
        _MESSAGE_START,
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
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Bye"},
        },
        {"type": "content_block_stop", "index": 2},
        *_MESSAGE_END_EVENTS,
    ]

    events = await _collect_client_events(
        monkeypatch, native_events, PROVIDER_CAPABILITIES["anthropic"]
    )

    part_events = [
        event
        for event in events
        if isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent))
    ]
    assert part_events == [
        PartStartEvent(index=0, part=TextOutput(content="")),
        PartDeltaEvent(index=0, delta=TextDelta(content_delta="Hi")),
        PartEndEvent(index=0, part=TextOutput(content="Hi")),
        PartStartEvent(index=1, part=TextOutput(content="")),
        PartDeltaEvent(index=1, delta=TextDelta(content_delta="Bye")),
        PartEndEvent(index=1, part=TextOutput(content="Bye")),
    ]


async def test_client_preserves_signature_only_thinking_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A thinking block without text deltas still carries its opaque signature.
    native_events = [
        _MESSAGE_START,
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
            "delta": {"type": "signature_delta", "signature": "final-state"},
        },
        {"type": "content_block_stop", "index": 0},
        *_MESSAGE_END_EVENTS,
    ]

    events = await _collect_client_events(
        monkeypatch, native_events, PROVIDER_CAPABILITIES["anthropic"]
    )

    part_events = [
        event
        for event in events
        if isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent))
    ]
    assert part_events == [
        PartStartEvent(index=0, part=Thinking(content="", summary="", signature="initial-state")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(signature_delta="final-state")),
        PartEndEvent(index=0, part=Thinking(content="", summary="", signature="final-state")),
    ]


async def test_client_rejects_unsigned_summary_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_events = [
        _MESSAGE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Summary"},
        },
        {"type": "content_block_stop", "index": 0},
    ]

    with pytest.raises(ValueError, match="summary requires a signature"):
        await _collect_client_events(
            monkeypatch, native_events, PROVIDER_CAPABILITIES["anthropic"]
        )


async def test_client_completes_thinking_block_without_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A thinking block whose text is complete at block start still gets a
    # balanced part lifecycle: the deferred start is flushed at block stop.
    native_events = [
        _MESSAGE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "Settled thought",
                "signature": "sig-1",
            },
        },
        {"type": "content_block_stop", "index": 0},
        *_MESSAGE_END_EVENTS,
    ]

    events = await _collect_client_events(monkeypatch, native_events)

    part_events = [
        event
        for event in events
        if isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent))
    ]
    assert part_events == [
        PartStartEvent(index=0, part=Thinking(content="Settled thought", summary="")),
        PartEndEvent(index=0, part=Thinking(content="Settled thought", summary="")),
    ]


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
        output_config=output_config,
        stream=True,
        thinking=thinking,
    )

    normalized_request = AnthropicMessagesAdapter(
        PROVIDER_CAPABILITIES["anthropic"]
    ).normalize_request(request)

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
    """Ensures Messages projection retains user parts and assistant message boundaries."""
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[
                UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")]),
                UserInput(content=[TextInput(content="Follow-up")]),
                AssistantInput(content=[TextOutput(content="Answer")]),
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


def test_client_projects_prompt_cache_as_automatic_cache_control():
    """Ensures automatic caching follows mode and lifetime intent, never a caller key."""

    def body(prompt_cache: PromptCache) -> dict[str, object]:
        return AnthropicMessagesClient._request_body(
            NormalizedRequest(
                max_output_tokens=2048,
                model="compatible-model",
                stream=True,
                prompt_cache=prompt_cache,
            )
        )

    assert body(PromptCache(mode="implicit"))["cache_control"] == {"type": "ephemeral"}
    assert body(PromptCache(ttl="1h"))["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert body(PromptCache(key="turn-1", ttl="30m"))["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert body(PromptCache(mode="implicit", ttl="5m"))["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }
    # Explicit mode asks for caller-marked breakpoints, which Messages only represents on
    # individual content blocks, so projecting no automatic cache control is faithful.
    assert "cache_control" not in body(PromptCache(mode="explicit", ttl="30m"))
    assert "cache_control" not in body(PromptCache(key="turn-1"))
    assert "cache_control" not in body(PromptCache())


def test_client_preserves_assistant_message_boundaries_and_ordered_blocks():
    """Ensures Messages retains adjacent containers, markers, and block ordering."""
    request = NormalizedRequest(
        model="compatible-model",
        max_output_tokens=2048,
        input=[
            AssistantInput(content=[]),
            AssistantInput(
                content=[
                    TextOutput(content=""),
                    Thinking(content="Thought", summary=""),
                    TextOutput(content="Cached", cache_breakpoint=CacheBreakpoint()),
                    ToolCall(tool_name="lookup", tool_call_id="call_1", args="{}"),
                    TextOutput(content="After tool"),
                ]
            ),
            AssistantInput(content=[TextOutput(content="Next message")]),
        ],
    )

    assert AnthropicMessagesClient._request_body(request)["messages"] == [
        {"role": "assistant", "content": []},
        {
            "role": "assistant",
            "content": [
                # An empty text part carries no content and Messages rejects an
                # empty block, so it is omitted.
                {"type": "thinking", "thinking": "Thought"},
                {"type": "text", "text": "Cached", "cache_control": {"type": "ephemeral"}},
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
                {"type": "text", "text": "After tool"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Next message"}]},
    ]


def test_adapter_preserves_user_message_and_part_boundaries():
    """Ensures Messages normalization retains user parts and assistant containers."""
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
        AssistantInput(content=[TextOutput(content="Answer")]),
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

    with pytest.raises(UnsupportedInput, match="cannot represent image detail"):
        AnthropicMessagesClient._request_body(request)


def test_client_rejects_non_base64_data_url_image():
    request = NormalizedRequest(
        input=[UserInput(content=[ImageUrlInput(url="data:image/png,%89NP")])],
        max_output_tokens=2048,
        model="compatible-model",
        stream=True,
    )

    with pytest.raises(UnsupportedInput, match="base64 data URL"):
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
    """Ensures nested assistant reasoning enables adaptive thinking for supported models."""
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[
                AssistantInput(
                    content=[
                        Thinking(content="Prior thought", summary=""),
                        TextOutput(content="Answer"),
                    ]
                )
            ],
            max_output_tokens=2048,
            model="compatible-model",
            stream=True,
        )
    )

    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}


def test_client_uses_summary_when_reasoning_history_omits_raw_content():
    """Ensures nested signed summaries preserve text and signature in thinking blocks."""
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[
                AssistantInput(
                    content=[
                        Thinking(content="", summary="Prior summary", signature="opaque-state"),
                        TextOutput(content="Prior answer"),
                    ]
                ),
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


def test_client_rejects_reasoning_history_outside_capability() -> None:
    request = NormalizedRequest(
        input=[Thinking(content="", summary="Summary", signature="opaque-state")],
        max_output_tokens=100,
        model="model",
        stream=True,
    )

    with pytest.raises(UnsupportedInput, match="does not match the provider"):
        AnthropicMessagesClient._request_body(request, capabilities=VLLM_CAPABILITIES)


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
    with pytest.raises(UnsupportedInput, match="must be less than max_output_tokens"):
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
    """Ensures request signatures remain attached to thinking inside the assistant container."""
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

    normalized_request = AnthropicMessagesAdapter(
        PROVIDER_CAPABILITIES["anthropic"]
    ).normalize_request(request)

    assert isinstance(normalized_request.input, list)
    assistant = normalized_request.input[1]
    assert isinstance(assistant, AssistantInput)
    thinking = assistant.content[0]
    assert isinstance(thinking, Thinking)
    assert thinking.summary == "Prior thought"
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
    """Ensures Messages preserves opaque continuation in nested request and streamed output."""
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
                        AssistantInput(
                            content=[
                                Thinking(content="Prior thought", summary=""),
                                TextOutput(content="Prior answer"),
                            ]
                        ),
                    ],
                    max_output_tokens=10,
                    model="model",
                    stream=True,
                )
            )
        ]

    assert events == []
    content = captured_requests[0].model_dump()["messages"][0]["content"]
    assert "signature" not in content[0]


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
    """Ensures Messages normalization retains grouped assistant text and function calls."""
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
        AssistantInput(
            content=[
                TextOutput(content="Running them."),
                ToolCall(tool_name="First", args='{"value":1}', tool_call_id="call_one"),
                ToolCall(tool_name="Second", args="{}", tool_call_id="call_two"),
            ]
        ),
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

    with pytest.raises(UnsupportedInput, match="directly invoked"):
        AnthropicMessagesAdapter(VLLM_CAPABILITIES).normalize_request(request)


def test_client_projects_function_tools_and_tool_history_without_caller():
    """Ensures grouped assistant tool history projects without synthesizing caller metadata."""
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            input=[
                UserInput(content=[TextInput(content="Run both tools.")]),
                AssistantInput(
                    content=[
                        TextOutput(content="Running them."),
                        ToolCall(tool_name="First", args='{"value":1}', tool_call_id="call_one"),
                        ToolCall(tool_name="Second", args="{}", tool_call_id="call_two"),
                    ]
                ),
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
    with pytest.raises(UnsupportedInput, match="needs at least one tool"):
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

    with pytest.raises(UnsupportedInput, match="cannot preserve denied or interrupted"):
        AnthropicMessagesClient._request_body(request)


def test_client_accepts_disabled_stream_obfuscation():
    body = AnthropicMessagesClient._request_body(
        NormalizedRequest(
            max_output_tokens=10,
            model="model",
            stream=True,
            stream_options=StreamOptions(include_obfuscation=False),
        )
    )

    assert body["stream"] is True


@pytest.mark.parametrize(
    ("normalized_request", "error"),
    [
        (NormalizedRequest(model="model", max_tool_calls=1), "max_tool_calls"),
        (
            NormalizedRequest(model="model", prompt_cache=PromptCache(retention="24h")),
            "prompt cache retention",
        ),
        (
            NormalizedRequest(
                model="model",
                stream_options=StreamOptions(include_obfuscation=True),
            ),
            "stream obfuscation",
        ),
        (
            NormalizedRequest(model="model", reasoning=Reasoning(summary="concise")),
            "concise reasoning summaries",
        ),
        (
            NormalizedRequest(
                input=[
                    ToolReturn(
                        tool_name="lookup",
                        content=[TextInput(content="result")],
                        tool_call_id="call_123",
                    )
                ],
                model="model",
            ),
            "tool return names",
        ),
    ],
)
def test_client_rejects_unrepresentable_normalized_request_fields(
    normalized_request: NormalizedRequest,
    error: str,
):
    with pytest.raises(UnsupportedInput, match=error):
        AnthropicMessagesClient._request_body(normalized_request)


def test_anthropic_client_ignores_responses_history_item_ids():
    """Ensures Messages omits optional item and message IDs while preserving call identities."""

    def request(item_id: str | None) -> NormalizedRequest:
        return NormalizedRequest(
            input=[
                AssistantInput(
                    content=[
                        Thinking(content="thought", summary="", id=item_id),
                        TextOutput(content="answer"),
                        ToolCall(
                            id=item_id, tool_call_id="call_123", tool_name="lookup", args="{}"
                        ),
                    ],
                    id=item_id,
                ),
                ToolReturn(
                    id=item_id,
                    content=[TextInput(content="result")],
                    tool_call_id="call_123",
                ),
            ],
            max_output_tokens=100,
            model="model",
        )

    assert AnthropicMessagesClient._request_body(
        request("responses_item_id")
    ) == AnthropicMessagesClient._request_body(request(None))


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


async def test_native_events_normalize_known_sdk_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "type": "error",
                "error": {"type": "not_found_error", "message": "model: m"},
                "request_id": "req_123",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AnthropicMessagesClient(
            http_client,
            capabilities=PROVIDER_CAPABILITIES["anthropic"],
            api_key="test-api-key",
        )
        request = AnthropicMessagesCreateRequest(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            stream=True,
        )
        with pytest.raises(ModelNotFoundError) as exc_info:
            [event async for event in client.native_events(request)]

    error = exc_info.value
    assert error.response.status_code == 404
    assert isinstance(error.__cause__, AnthropicAPIStatusError)


@pytest.mark.parametrize(
    ("sdk_error_type", "status", "message"),
    [
        (AnthropicAPIStatusError, 400, "prompt is too long: 2 > 1"),
        (AnthropicBadRequestError, 400, "max_tokens is invalid"),
        (AnthropicAPIStatusError, 401, "Invalid key"),
        (AnthropicAPIStatusError, 404, "model: m"),
    ],
)
async def test_error_recognition_requires_sdk_class_and_message(sdk_error_type, status, message):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    error = sdk_error_type(message, response=response, body=None)
    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(
            http_client, capabilities=PROVIDER_CAPABILITIES["anthropic"]
        )
        with pytest.raises(AnthropicAPIStatusError) as exc_info:
            client._handle_error(error)
    assert exc_info.value is error


@pytest.mark.parametrize("status", [400, 401, 404, 422])
@pytest.mark.parametrize(
    ("sdk_error_type", "message", "normalized_type"),
    [
        (AnthropicBadRequestError, "prompt is too long: 2 > 1", ContextLengthExceededError),
        (AnthropicAuthenticationError, "Invalid key", AuthenticationError),
        (AnthropicNotFoundError, "model: m", ModelNotFoundError),
        (AnthropicNotFoundError, "Resource missing", ResourceNotFoundError),
    ],
)
async def test_error_recognition_uses_sdk_class(sdk_error_type, status, message, normalized_type):
    class ProviderError(sdk_error_type):
        pass

    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    error = ProviderError(message, response=response, body=None)
    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(
            http_client, capabilities=PROVIDER_CAPABILITIES["anthropic"]
        )
        with pytest.raises(normalized_type) as exc_info:
            client._handle_error(error)
    assert exc_info.value.__cause__ is error
    assert exc_info.value.response is response
    assert exc_info.value.response.status_code == status
    assert exc_info.value.body is error.body


@pytest.mark.parametrize(
    "error",
    [
        TypeError("Messages request did not return a stream."),
        ValueError('"Could not resolve authentication method.'),
    ],
)
async def test_missing_credentials_recognition_requires_type_and_message(error):
    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(
            http_client, capabilities=PROVIDER_CAPABILITIES["anthropic"]
        )
        with pytest.raises(type(error)) as exc_info:
            client._handle_error(error)
    assert exc_info.value is error


async def test_missing_credentials_recognizes_type_error_subclasses():
    class CredentialsError(TypeError):
        pass

    error = CredentialsError('"Could not resolve authentication method.')
    async with httpx.AsyncClient() as http_client:
        client = AnthropicMessagesClient(
            http_client, capabilities=PROVIDER_CAPABILITIES["anthropic"]
        )
        with pytest.raises(AuthenticationError) as exc_info:
            client._handle_error(error)
    assert exc_info.value.__cause__ is error
    assert exc_info.value.response is None


@pytest.mark.parametrize("mode", ["implicit", "explicit"])
def test_cache_breakpoints_preserve_system_parts_and_inherit_lifetime(mode):
    """Ensures inherited cache lifetimes apply consistently without merging instruction parts."""
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[
            SystemPrompt(content="Cached instructions", cache_breakpoint=CacheBreakpoint()),
            SystemPrompt(content="Uncached instructions"),
            UserInput(content=[TextInput(content="Question", cache_breakpoint=CacheBreakpoint())]),
        ],
        prompt_cache=PromptCache(mode=mode, ttl="30m"),
    )
    body = AnthropicMessagesClient._request_body(request)
    control = {"type": "ephemeral", "ttl": "1h"}
    assert body["system"] == [
        {"type": "text", "text": "Cached instructions", "cache_control": control},
        {"type": "text", "text": "Uncached instructions"},
    ]
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Question", "cache_control": control}],
        }
    ]
    assert body.get("cache_control") == (control if mode == "implicit" else None)


def test_rejects_cache_prefix_changes_from_hoisted_instructions():
    """Asserts that instruction hoisting cannot silently change a marked cache prefix."""
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[
            UserInput(content=[TextInput(content="Question", cache_breakpoint=CacheBreakpoint())]),
            SystemPrompt(content="Later instructions"),
        ],
    )
    with pytest.raises(UnsupportedInput, match="hoisting"):
        AnthropicMessagesClient._request_body(request)


@pytest.mark.parametrize(
    ("count", "mode", "unmarked_tail"),
    [
        (4, "explicit", True),
        (5, "explicit", False),
        (3, "implicit", True),
        (4, "implicit", True),
        (4, "implicit", False),
    ],
)
def test_cache_breakpoint_count_is_left_to_upstream(count, mode, unmarked_tail):
    """Ensures every marker and unmarked tail survives while Messages enforces count limits."""
    content = [TextInput(content=str(i), cache_breakpoint=CacheBreakpoint()) for i in range(count)]
    if unmarked_tail:
        content.append(TextInput(content="Uncached tail"))
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[UserInput(content=content)],
        prompt_cache=PromptCache(mode=mode),
    )
    body = AnthropicMessagesClient._request_body(request)
    expected = [
        {"type": "text", "text": str(i), "cache_control": {"type": "ephemeral"}}
        for i in range(count)
    ]
    if unmarked_tail:
        expected.append({"type": "text", "text": "Uncached tail"})
    assert body["messages"][0]["content"] == expected
    assert body.get("cache_control") == ({"type": "ephemeral"} if mode == "implicit" else None)


@pytest.mark.parametrize("ttls", [("1h", "5m"), ("5m", "1h")])
def test_block_lifetimes_must_decrease_in_prompt_order(ttls):
    """Asserts that lifetime-order validation preserves prompt order instead of rearranging it."""
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[
            UserInput(
                content=[
                    TextInput(content=str(i), cache_breakpoint=CacheBreakpoint(ttl=ttl))
                    for i, ttl in enumerate(ttls)
                ]
            )
        ],
        prompt_cache=PromptCache(mode="explicit"),
    )
    if ttls == ("1h", "5m"):
        body = AnthropicMessagesClient._request_body(request)
        assert [part["cache_control"]["ttl"] for part in body["messages"][0]["content"]] == list(
            ttls
        )
    else:
        with pytest.raises(UnsupportedInput, match="must precede"):
            AnthropicMessagesClient._request_body(request)


def test_automatic_and_final_block_cache_lifetimes_must_match():
    """Asserts that conflicting final explicit and automatic cache lifetimes are rejected."""
    request = NormalizedRequest(
        model="m",
        max_output_tokens=100,
        input=[
            UserInput(
                content=[
                    TextInput(content="Question", cache_breakpoint=CacheBreakpoint(ttl="1h")),
                ]
            )
        ],
        prompt_cache=PromptCache(mode="implicit", ttl="5m"),
    )
    with pytest.raises(UnsupportedInput, match="must match"):
        AnthropicMessagesClient._request_body(request)
