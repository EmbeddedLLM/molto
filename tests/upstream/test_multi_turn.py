from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from molto.providers import PROVIDER_CAPABILITIES
from molto.types.normalized import (
    AssistantInput,
    FunctionTool,
    ModelResponseStreamEvent,
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    Reasoning,
    ResponseEndEvent,
    ResponseStartEvent,
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
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from molto.utils.io import json_loads, read_yaml

CLIENTS = [OpenAIResponsesClient, OpenAIChatCompletionsClient, AnthropicMessagesClient]
REASONING_CLIENTS = [OpenAIResponsesClient, AnthropicMessagesClient]
CASSETTE_DIR = Path(__file__).parents[1] / "cassettes" / "upstream_ir"
CLIENT_SETTINGS = {
    OpenAIResponsesClient: ("openai", "responses"),
    OpenAIChatCompletionsClient: ("openai", "chat_completions"),
    AnthropicMessagesClient: ("anthropic", "messages"),
}


@asynccontextmanager
async def _replay_client(client_type, scenario: str, turns: int):
    provider, endpoint = CLIENT_SETTINGS[client_type]
    cassettes = [
        read_yaml(CASSETTE_DIR / scenario / f"{provider}-{endpoint}-stream-{turn:02}.yaml")
        for turn in range(turns)
    ]
    consumed = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal consumed
        assert consumed < len(cassettes), "Unexpected extra provider request"
        cassette = cassettes[consumed]
        assert request.method == cassette["request"]["method"]
        assert request.url.path == cassette["request"]["path"]
        # Exact history matching proves previous IR parts, call IDs, results, and
        # opaque reasoning snapshots reach the next turn without being changed.
        assert json_loads(request.content) == cassette["request"]["body"]
        assert cassette["response"]["status_code"] == 200
        consumed += 1
        return httpx.Response(
            200,
            text="".join(cassette["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        client = client_type(
            http_client,
            api_key="test-key",
            capabilities=PROVIDER_CAPABILITIES[provider],
            extra_body=(
                {"include": ["reasoning.encrypted_content"], "store": False}
                if client_type is OpenAIResponsesClient and scenario == "tool_use_reasoning"
                else None
            ),
        )
        yield client, cassettes[0]["request"]["body"]["model"]
    assert consumed == turns, "Not all conversation turns were replayed"


def _assert_ir(events: list[ModelResponseStreamEvent]) -> list[TextOutput | ToolCall | Thinking]:
    """Check lifecycle balance and reconstruct each completed part from its deltas."""
    start, end = events[0], events[-1]
    assert isinstance(start, ResponseStartEvent)
    assert isinstance(end, ResponseEndEvent)
    assert start.id == end.id and start.id
    assert start.model == end.model and start.model
    assert start.created_at == end.created_at
    assert end.stop_sequence is None
    usage = end.usage
    assert usage is not None
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert 0 <= usage.input_tokens_details.cached_tokens <= usage.input_tokens
    assert 0 <= usage.input_tokens_details.cache_write_tokens <= usage.input_tokens
    assert 0 <= usage.output_tokens_details.reasoning_tokens <= usage.output_tokens

    parts = []
    active = None
    for event in events[1:-1]:
        assert isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent))
        assert event.index == len(parts)
        if isinstance(event, PartStartEvent):
            assert active is None
            active = event.part.model_copy(deep=True)
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, TextDelta):
                assert isinstance(active, TextOutput)
                active.content += delta.content_delta
            elif isinstance(delta, ToolCallDelta):
                assert isinstance(active, ToolCall)
                active.tool_name += delta.tool_name_delta or ""
                active.args += delta.args_delta or ""
                if delta.tool_call_id is not None:
                    active.tool_call_id = delta.tool_call_id
            else:
                assert isinstance(delta, ThinkingDelta)
                assert isinstance(active, Thinking)
                assert not delta.content_delta
                active.summary += delta.summary_delta or ""
                if delta.signature_delta is not None:
                    # Opaque state is a replacement snapshot, never a text fragment.
                    active.signature = delta.signature_delta
        else:
            assert active == event.part
            if isinstance(active, TextOutput):
                assert active.content.strip()
            elif isinstance(active, ToolCall):
                assert active.tool_call_id
                assert active.tool_name == "read_file"
                args = json_loads(active.args)
                assert isinstance(args, dict) and set(args) == {"path"}
                assert isinstance(args["path"], str) and args["path"]
            else:
                assert isinstance(active, Thinking)
                assert active.content == ""
                assert active.signature
            parts.append(event.part)
            active = None
    assert active is None
    assert parts
    assert end.termination_reason == (
        "tool_call" if any(isinstance(part, ToolCall) for part in parts) else "stop"
    )
    return parts


@pytest.mark.parametrize("client_type", CLIENTS, ids=lambda cls: cls.__name__)
async def test_multi_turn_text_chat(client_type) -> None:
    """Ensures all clients emit coherent text IR and preserve assistant history."""
    async with _replay_client(client_type, "text_chat", turns=2) as (client, model):
        request = NormalizedRequest(
            model=model,
            stream=True,
            max_output_tokens=512,
            input=[UserInput(content=[TextInput(content="Suggest a name for a robot gardener.")])],
        )
        response_ids = set()
        for turn in range(2):
            events = [event async for event in client.events(request)]
            parts = _assert_ir(events)
            assert all(isinstance(part, TextOutput) for part in parts)
            assert events[0].id not in response_ids
            response_ids.add(events[0].id)
            if turn == 0:
                request.input.extend(
                    [
                        AssistantInput(content=parts),
                        UserInput(content=[TextInput(content="Explain why that name fits.")]),
                    ]
                )


async def _assert_tool_conversation(client_type, *, reasoning: bool) -> None:
    scenario = "tool_use_reasoning" if reasoning else "tool_use"
    async with _replay_client(client_type, scenario, turns=3) as (client, model):
        request = NormalizedRequest(
            model=model,
            stream=True,
            max_output_tokens=4096 if reasoning else 512,
            reasoning=Reasoning(effort="low", summary="auto") if reasoning else None,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[
                FunctionTool(
                    name="read_file",
                    description="Read a file by its path.",
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                )
            ],
            input=[
                UserInput(
                    content=[
                        TextInput(
                            content="Read index.txt using read_file. It contains the path of another "
                            "file. Read that file too, then summarize its contents in one sentence."
                        )
                    ]
                )
            ],
        )
        call_ids = set()
        response_ids = set()
        continued_thinking = False
        for turn in range(3):
            events = [event async for event in client.events(request)]
            parts = _assert_ir(events)
            assert events[0].id not in response_ids
            response_ids.add(events[0].id)
            thoughts = [part for part in parts if isinstance(part, Thinking)]
            if not reasoning:
                assert not thoughts
            calls = [part for part in parts if isinstance(part, ToolCall)]
            if turn == 2:
                assert not calls
                assert any(isinstance(part, TextOutput) for part in parts)
                continue

            assert len(calls) == 1
            continued_thinking |= bool(thoughts)
            call = calls[0]
            assert call.tool_call_id not in call_ids
            call_ids.add(call.tool_call_id)
            # These are fixture inputs, not provider-generated answer prose. The
            # second path can only be learned from the preceding tool result.
            assert json_loads(call.args)["path"] == ("index.txt" if turn == 0 else "garden.txt")
            request.input.extend(
                [
                    AssistantInput(content=parts),
                    ToolReturn(
                        tool_call_id=call.tool_call_id,
                        content=[
                            TextInput(
                                content="garden.txt"
                                if turn == 0
                                else "Water the tomatoes every morning and harvest ripe fruit."
                            )
                        ],
                    ),
                ]
            )
        assert continued_thinking is reasoning


@pytest.mark.parametrize("client_type", CLIENTS, ids=lambda cls: cls.__name__)
async def test_multi_turn_tool_use(client_type) -> None:
    """Ensures tool IR preserves JSON arguments, call identity, results, and final text."""
    await _assert_tool_conversation(client_type, reasoning=False)


@pytest.mark.parametrize("client_type", REASONING_CLIENTS, ids=lambda cls: cls.__name__)
async def test_multi_turn_tool_use_with_reasoning(client_type) -> None:
    """Ensures reasoning summaries and opaque state survive a complete tool conversation."""
    await _assert_tool_conversation(client_type, reasoning=True)
