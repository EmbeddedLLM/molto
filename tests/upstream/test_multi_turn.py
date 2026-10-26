from base64 import b64encode
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from molto.providers import PROVIDER_CAPABILITIES
from molto.types.normalized import (
    AssistantInput,
    FunctionTool,
    ImageUrlInput,
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
async def _replay_client(
    client_type, scenario: str, turns: int, *, requests: list[httpx.Request] | None = None
):
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
        if requests is not None:
            requests.append(request)
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


@pytest.mark.parametrize("client_type", CLIENTS, ids=lambda cls: cls.__name__)
async def test_multi_turn_text_and_image_chat(client_type) -> None:
    """Ensures mixed text/image parts stay in one user message across conversation turns."""
    requests = []
    async with _replay_client(client_type, "text_image_chat", turns=2, requests=requests) as (
        client,
        model,
    ):
        request = NormalizedRequest(
            model=model,
            stream=True,
            max_output_tokens=512,
            reasoning=Reasoning(effort="none"),
            input=[],
        )
        response_ids = set()
        for filename, prompt in (
            ("github.png", "Describe this image."),
            ("xlsx.jpg", "How does this image differ from the previous one?"),
        ):
            image_path = Path(__file__).parents[1] / "files" / filename
            media_type = "image/png" if filename.endswith(".png") else "image/jpeg"
            image_url = (
                f"data:{media_type};base64,{b64encode(image_path.read_bytes()).decode('ascii')}"
            )
            request.input.append(
                UserInput(
                    content=[
                        TextInput(content=prompt),
                        ImageUrlInput(url=image_url),
                        TextInput(content="Answer in one sentence."),
                    ]
                )
            )
            expected_input = request.model_copy(deep=True).input
            events = [event async for event in client.events(request)]
            parts = _assert_ir(events)
            assert all(isinstance(part, TextOutput) for part in parts)
            assert events[0].id not in response_ids
            response_ids.add(events[0].id)
            assert request.input == expected_input

            body = json_loads(requests[-1].content)
            messages = body["input"] if client_type is OpenAIResponsesClient else body["messages"]
            assert [message["role"] for message in messages] == [
                "user" if isinstance(item, UserInput) else "assistant" for item in expected_input
            ]
            text_type, image_type = {
                OpenAIResponsesClient: ("input_text", "input_image"),
                OpenAIChatCompletionsClient: ("text", "image_url"),
                AnthropicMessagesClient: ("text", "image"),
            }[client_type]
            for message in messages:
                if message["role"] == "user":
                    assert [part["type"] for part in message["content"]] == [
                        text_type,
                        image_type,
                        text_type,
                    ]
            request.input.append(AssistantInput(content=parts))


@pytest.mark.parametrize("client_type", CLIENTS, ids=lambda cls: cls.__name__)
async def test_multi_turn_consecutive_user_text_inputs(client_type) -> None:
    """Ensures adjacent user messages and text parts retain their boundaries across turns."""
    requests = []
    async with _replay_client(
        client_type, "consecutive_user_text", turns=2, requests=requests
    ) as (
        client,
        model,
    ):
        request = NormalizedRequest(
            model=model,
            stream=True,
            max_output_tokens=512,
            reasoning=Reasoning(effort="none"),
            input=[],
        )
        response_ids = set()
        for prompts in (
            (
                ("Help me plan a small vegetable garden.", "The garden gets morning sun."),
                ("Suggest two plants.", "Keep your answer to one sentence."),
            ),
            (
                ("I can water it every morning.", "I have room for two pots."),
                (
                    "Refine your suggestion using those details.",
                    "Keep your answer to one sentence.",
                ),
            ),
        ):
            request.input.extend(
                UserInput(content=[TextInput(content=text) for text in message])
                for message in prompts
            )
            expected_input = request.model_copy(deep=True).input
            events = [event async for event in client.events(request)]
            parts = _assert_ir(events)
            assert all(isinstance(part, TextOutput) for part in parts)
            assert events[0].id not in response_ids
            response_ids.add(events[0].id)
            assert request.input == expected_input

            body = json_loads(requests[-1].content)
            messages = body["input"] if client_type is OpenAIResponsesClient else body["messages"]
            assert [message["role"] for message in messages] == [
                "user" if isinstance(item, UserInput) else "assistant" for item in expected_input
            ]
            assert [
                [part["text"] for part in message["content"]]
                for message in messages
                if message["role"] == "user"
            ] == [
                [part.content for part in item.content]
                for item in expected_input
                if isinstance(item, UserInput)
            ]
            request.input.append(AssistantInput(content=parts))


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
