import json
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

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
from molto.upstream.openai import OpenAIResponsesClient
from molto.upstream.registry import CLIENTS as PROVIDER_CLIENTS
from molto.utils.io import json_loads, read_yaml
from tests.core.test_recorded_provider_conversations import collect
from tests.core.test_together_conversations import _semantics

# Reusing canonical recordings exercises each declared class's protocol contract;
# these combinations are offline tests, not evidence that those models run there.
CLIENT_SETTINGS = {
    client: (provider, endpoint.replace("/", "_"))
    for (provider, endpoint), client in PROVIDER_CLIENTS.items()
}
CLIENTS = list(CLIENT_SETTINGS)
REASONING_CLIENTS = [
    client
    for client, (provider, endpoint) in CLIENT_SETTINGS.items()
    if endpoint != "chat_completions"
    and client.resolve_capabilities(PROVIDER_CAPABILITIES[provider]).thinking == "summary"
]
CASSETTE_DIR = Path(__file__).parents[1] / "cassettes" / "upstream_ir"


@asynccontextmanager
async def _replay_client(
    client_type, scenario: str, turns: int, *, requests: list[httpx.Request] | None = None
):
    provider, endpoint = CLIENT_SETTINGS[client_type]
    source_provider = "anthropic" if endpoint == "messages" else "openai"
    cassettes = [
        read_yaml(CASSETTE_DIR / scenario / f"{source_provider}-{endpoint}-stream-{turn:02}.yaml")
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
            capabilities=client_type.resolve_capabilities(PROVIDER_CAPABILITIES[provider]),
            base_url="https://api.anthropic.com"
            if endpoint == "messages"
            else "https://api.openai.com/v1",
            extra_body=(
                {"include": ["reasoning.encrypted_content"], "store": False}
                if issubclass(client_type, OpenAIResponsesClient)
                and scenario == "tool_use_reasoning"
                else None
            ),
        )
        yield client, cassettes[0]["request"]["body"]["model"]
    assert consumed == turns, "Not all conversation turns were replayed"


def _assert_ir(
    events: list[ModelResponseStreamEvent],
    *,
    tool_name="read_file",
    source_timestamps=None,
    raw_reasoning=False,
) -> list[TextOutput | ToolCall | Thinking]:
    """Check lifecycle balance and reconstruct each completed part from its deltas."""
    start, end = events[0], events[-1]
    assert isinstance(start, ResponseStartEvent)
    assert isinstance(end, ResponseEndEvent)
    assert start.id == end.id and start.id
    assert start.model == end.model and start.model
    if source_timestamps is None:
        assert start.created_at == end.created_at
    else:
        assert (start.created_at, end.created_at) == source_timestamps
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
                if raw_reasoning:
                    assert not delta.summary_delta and not delta.signature_delta
                    active.content += delta.content_delta or ""
                else:
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
                assert active.tool_name == tool_name
                args = json_loads(active.args)
                assert isinstance(args, dict)
                if tool_name == "read_file":
                    assert set(args) == {"path"}
                    assert isinstance(args["path"], str) and args["path"]
                else:
                    assert set(args) == {"a", "b"}
                    assert all(type(value) is int for value in args.values())
            else:
                assert isinstance(active, Thinking)
                if raw_reasoning:
                    assert active.content and not active.summary and not active.signature
                else:
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
            request.input.append(
                UserInput(
                    content=[
                        TextInput(content=prompt),
                        ImageUrlInput.from_bytes(image_path.read_bytes(), media_type),
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
            messages = (
                body["input"]
                if issubclass(client_type, OpenAIResponsesClient)
                else body["messages"]
            )
            assert [message["role"] for message in messages] == [
                "user" if isinstance(item, UserInput) else "assistant" for item in expected_input
            ]
            text_type, image_type = {
                "responses": ("input_text", "input_image"),
                "chat_completions": ("text", "image_url"),
                "messages": ("text", "image"),
            }[CLIENT_SETTINGS[client_type][1]]
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
            messages = (
                body["input"]
                if issubclass(client_type, OpenAIResponsesClient)
                else body["messages"]
            )
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


RECORDED_ROOT = Path(__file__).parents[1] / "cassettes/provider_recordings/2026-11-04"
RECORDED_GROUPS = [
    "fireworks-chat-text-stream-none-history",
    "fireworks-chat-tools-stream-none-history",
    "fireworks-responses-text-stream-none",
    "fireworks-responses-tools-stream-none",
    "fireworks-messages-text-stream-none",
    "fireworks-messages-tools-stream-none",
    "openrouter-chat-text-stream-none",
    "openrouter-responses-text-stream-none",
    "openrouter-responses-tools-stream-low",
    "openrouter-messages-text-stream-none",
    "openrouter-messages-tools-stream-none",
]


def _caller_inputs(messages):
    """Read only recorded caller input; assistant history must come from actual emitted IR."""
    result = []
    for message in messages:
        if message.get("type") == "function_call_output":
            result.append(
                ToolReturn(
                    tool_call_id=message["call_id"], content=[TextInput(content=message["output"])]
                )
            )
        elif message.get("role") == "tool":
            result.append(
                ToolReturn(
                    tool_call_id=message["tool_call_id"],
                    content=[TextInput(content=message["content"])],
                )
            )
        else:
            assert message["role"] == "user"
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if all(part["type"] == "tool_result" for part in content):
                result.extend(
                    ToolReturn(
                        tool_call_id=part["tool_use_id"],
                        content=[TextInput(content=part["content"])],
                    )
                    for part in content
                )
            else:
                assert all(part["type"] in {"text", "input_text"} for part in content)
                result.append(
                    UserInput(content=[TextInput(content=part["text"]) for part in content])
                )
    return result


def _expected_normalized_wire(recorded, protocol):
    """Derive expected wire copies from native evidence using only known representation changes."""
    body = deepcopy(recorded)
    history = body["input" if protocol == "responses" else "messages"]
    for message in history:
        if message.get("role") in {"user", "tool"} and isinstance(message["content"], str):
            message["content"] = [
                {
                    "type": "input_text" if protocol == "responses" else "text",
                    "text": message["content"],
                }
            ]
        if message.get("role") == "assistant":
            if protocol == "chat/completions":
                for field in ("refusal", "tool_calls", "tools", "reasoning", "reasoning_content"):
                    if field in message and message[field] is None:
                        del message[field]
                if message.get("tool_calls"):
                    if message.get("content") is None:
                        message.pop("content", None)
                    for call in message["tool_calls"]:
                        if call.get("name") is None:
                            call.pop("name", None)
            elif protocol == "responses":
                # Normalized text parts do not carry their enclosing source message ID.
                message.pop("id", None)
                assert message.get("phase") is None
                message.pop("phase", None)
                for part in message["content"]:
                    part.setdefault("annotations", [])
                    part.setdefault("logprobs", [])
            else:
                for part in message["content"]:
                    if part.get("citations") in (None, []):
                        part.pop("citations", None)
                    if part.get("caller") in (None, {"type": "direct"}):
                        part.pop("caller", None)
        if message.get("type") == "reasoning":
            assert message.pop("format", None) == "unknown"
            assert message.pop("status") == "completed"
        if message.get("type") == "function_call_output":
            message["output"] = [{"type": "input_text", "text": message["output"]}]
        if protocol == "messages" and message.get("role") == "user":
            for part in message["content"]:
                if part["type"] == "tool_result" and isinstance(part["content"], str):
                    part["content"] = [{"type": "text", "text": part["content"]}]
    if protocol == "chat/completions":
        body["max_completion_tokens"] = body.pop("max_tokens")
    if protocol != "messages":
        for tool in body.get("tools", []):
            (tool["function"] if protocol == "chat/completions" else tool).setdefault(
                "strict", False
            )
    return body


@pytest.mark.parametrize("group_name", RECORDED_GROUPS)
async def test_recorded_provider_normalized_multi_turn(group_name):
    """Ensures actual provider IR continues text/tool turns with exact independently expected wire bodies."""
    group = RECORDED_ROOT / group_name
    manifest = json.loads((group / "manifest.json").read_text())
    protocol = manifest["upstream_protocol"]
    provider = manifest["provider"]
    client_type = PROVIDER_CLIENTS[(provider, protocol)]
    recordings = [read_yaml(group / turn["files"][0]) for turn in manifest["turns"]]
    history_key = "input" if protocol == "responses" else "messages"
    first_body = recordings[0]["request"]["body"]
    request = NormalizedRequest(
        model=first_body["model"],
        stream=True,
        max_output_tokens=manifest["max_tokens"],
        input=_caller_inputs(first_body[history_key]),
    )
    if first_body.get("tools"):
        tool = first_body["tools"][0]
        tool = tool["function"] if protocol == "chat/completions" else tool
        request.tools = [
            FunctionTool(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema" if protocol == "messages" else "parameters"],
            )
        ]
    calls = 0
    for turn, recording in enumerate(recordings):
        recorded = recording["request"]["body"]
        expected = _expected_normalized_wire(recorded, protocol)
        response = recording["response"]
        assert response["status_code"] == 200

        def handle(http_request, expected=expected, response=response):
            nonlocal calls
            calls += 1
            assert str(http_request.url) == manifest["upstream_url"]
            assert json_loads(http_request.content) == expected
            return httpx.Response(
                200, text="".join(response["sse"]), headers={"content-type": "text/event-stream"}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _, response=response: httpx.Response(
                    200,
                    text="".join(response["sse"]),
                    headers={"content-type": "text/event-stream"},
                )
            )
        ) as transport:
            sdk = (
                AsyncAnthropic(api_key="test", http_client=transport)
                if protocol == "messages"
                else AsyncOpenAI(api_key="test", http_client=transport)
            )
            create = (
                sdk.messages.create
                if protocol == "messages"
                else sdk.responses.create
                if protocol == "responses"
                else sdk.chat.completions.create
            )
            sdk_body = {
                key: value
                for key, value in recorded.items()
                if key not in (manifest.get("extra_body") or {})
            }
            reference, source_events = await collect(
                await create(**sdk_body, extra_body=manifest.get("extra_body")), True, protocol
            )
        controls = {
            key: recorded[key]
            for key in (
                "reasoning",
                "reasoning_effort",
                "thinking",
                "tool_choice",
                "parallel_tool_calls",
                "store",
            )
            if key in recorded
        }
        controls.update(manifest.get("extra_body") or {})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
            client = client_type(
                transport,
                api_key="test",
                capabilities=client_type.resolve_capabilities(PROVIDER_CAPABILITIES[provider]),
                extra_body=controls,
            )
            before = request.model_copy(deep=True)
            events = [event async for event in client.events(request)]
            assert request == before
        timestamps = (
            (int(source_events[0]["response"]["created_at"]), int(reference.created_at))
            if protocol == "responses"
            else None
        )
        parts = _assert_ir(
            events,
            tool_name="add",
            source_timestamps=timestamps,
            raw_reasoning=manifest.get("reasoning_setting") == "low",
        )
        source_id, model, text, tool_calls, reasoning, usage = _semantics(reference, protocol)
        assert (events[0].id, events[0].model) == (source_id, model)
        assert "".join(part.content for part in parts if isinstance(part, TextOutput)) == text
        assert [
            (part.tool_call_id, part.tool_name, json_loads(part.args))
            for part in parts
            if isinstance(part, ToolCall)
        ] == tool_calls
        assert [
            (part.content, part.summary, part.signature)
            for part in parts
            if isinstance(part, Thinking)
        ] == reasoning
        assert (
            events[-1].usage.input_tokens,
            events[-1].usage.output_tokens,
            events[-1].usage.total_tokens,
        ) == usage
        if turn + 1 < len(recordings):
            request.input.append(AssistantInput(content=parts))
            output_items = len(reference.output) if protocol == "responses" else 1
            suffix = recordings[turn + 1]["request"]["body"][history_key][
                len(recorded[history_key]) + output_items :
            ]
            assert suffix
            request.input.extend(_caller_inputs(suffix))
    assert calls == len(recordings)


async def test_recorded_together_alternate_reasoning_rejects_normalized_output():
    """Ensures Together's current alternate reasoning dialect rejects rather than losing SDK-visible text."""
    group = RECORDED_ROOT.parent / "2026-11-03/together-text-stream"
    manifest = json.loads((group / "manifest.json").read_text())
    recording = read_yaml(group / manifest["turns"][0]["files"][0])
    body = recording["request"]["body"]
    expected = _expected_normalized_wire(body, "chat/completions")
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        assert json_loads(request.content) == (body if calls == 1 else expected)
        return httpx.Response(
            200,
            text="".join(recording["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        sdk = AsyncOpenAI(api_key="test", http_client=transport)
        sdk_body = {
            key: value
            for key, value in body.items()
            if key not in (manifest.get("extra_body") or {})
        }
        reference, _ = await collect(
            await sdk.chat.completions.create(**sdk_body, extra_body=manifest["extra_body"]), True
        )
        assert reference.choices[0].message.reasoning_content
        client_type = PROVIDER_CLIENTS[("togetherai", "chat/completions")]
        client = client_type(
            transport,
            api_key="test",
            capabilities=client_type.resolve_capabilities(PROVIDER_CAPABILITIES["togetherai"]),
            extra_body=manifest["extra_body"],
        )
        request = NormalizedRequest(
            model=body["model"],
            stream=True,
            max_output_tokens=manifest["max_tokens"],
            input=_caller_inputs(body["messages"]),
        )
        with pytest.raises(
            ValueError, match="reasoning field 'reasoning_content'.*'reasoning' dialect"
        ):
            _ = [event async for event in client.events(request)]
    assert calls == 2 and transport.is_closed
