from contextlib import asynccontextmanager, nullcontext
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from anthropic.lib.streaming._messages import accumulate_event
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState

from molto import Molto
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
from molto.upstream.bedrock import BedrockChatCompletionsClient, BedrockResponsesClient
from molto.upstream.fireworks import FireworksChatCompletionsClient, FireworksResponsesClient
from molto.upstream.gemini import GeminiChatCompletionsClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from molto.upstream.openrouter import OpenRouterResponsesClient
from molto.upstream.together import TogetherAIChatCompletionsClient, TogetherAIResponsesClient
from molto.utils.io import json_loads, read_yaml

CLIENTS = {
    OpenAIResponsesClient: ("openai", "responses"),
    OpenAIChatCompletionsClient: ("openai", "chat_completions"),
    AnthropicMessagesClient: ("anthropic", "messages"),
    BedrockResponsesClient: ("bedrock", "responses"),
    BedrockChatCompletionsClient: ("bedrock", "chat_completions"),
    FireworksResponsesClient: ("fireworks", "responses"),
    FireworksChatCompletionsClient: ("fireworks", "chat_completions"),
    GeminiChatCompletionsClient: ("gemini", "chat_completions"),
    OpenRouterResponsesClient: ("openrouter", "responses"),
    TogetherAIResponsesClient: ("togetherai", "responses"),
    TogetherAIChatCompletionsClient: ("togetherai", "chat_completions"),
}
# The original scenario prompts belong to these providers' own recordings.
# Hosted captures below use their recorded prompts and available scenarios.
CANONICAL_CLIENTS = [
    client for client, (provider, _) in CLIENTS.items() if provider in {"openai", "anthropic"}
]
REASONING_CLIENTS = [
    client for client in CANONICAL_CLIENTS if CLIENTS[client][1] != "chat_completions"
]
CASSETTE_DIR = Path(__file__).parents[1] / "cassettes" / "upstream_ir"


@asynccontextmanager
async def _replay_client(
    client_type, scenario: str, turns: int, *, requests: list[httpx.Request] | None = None
):
    provider, endpoint = CLIENTS[client_type]
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

    parts = {}
    active_parts = {}
    for event in events[1:-1]:
        assert isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent))
        if isinstance(event, PartStartEvent):
            assert event.index == len(parts) + len(active_parts)
            active_parts[event.index] = event.part.model_copy(deep=True)
            continue
        assert event.index in active_parts
        active = active_parts[event.index]
        if isinstance(event, PartDeltaEvent):
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
                assert isinstance(active, Thinking)
                if raw_reasoning:
                    assert active.content and not active.summary and not active.signature
                else:
                    assert active.content == ""
                    assert active.signature
            parts[event.index] = event.part
            del active_parts[event.index]
    assert not active_parts
    assert parts
    parts = [parts[index] for index in range(len(parts))]
    assert end.termination_reason == (
        "tool_call" if any(isinstance(part, ToolCall) for part in parts) else "stop"
    )
    return parts


@pytest.mark.parametrize("client_type", CANONICAL_CLIENTS, ids=lambda cls: cls.__name__)
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


@pytest.mark.parametrize("client_type", CANONICAL_CLIENTS, ids=lambda cls: cls.__name__)
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


@pytest.mark.parametrize("client_type", CANONICAL_CLIENTS, ids=lambda cls: cls.__name__)
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


@pytest.mark.parametrize("client_type", CANONICAL_CLIENTS, ids=lambda cls: cls.__name__)
async def test_multi_turn_tool_use(client_type) -> None:
    """Ensures tool IR preserves JSON arguments, call identity, results, and final text."""
    await _assert_tool_conversation(client_type, reasoning=False)


@pytest.mark.parametrize("client_type", REASONING_CLIENTS, ids=lambda cls: cls.__name__)
async def test_multi_turn_tool_use_with_reasoning(client_type) -> None:
    """Ensures reasoning summaries and opaque state survive a complete tool conversation."""
    await _assert_tool_conversation(client_type, reasoning=True)


# Only provider-owned recordings belong in this matrix. Missing scenarios are
# absent coverage, never canonical fixtures relabeled as another provider.
PROVIDER_CLIENTS = {
    (provider, endpoint): client
    for client, (provider, endpoint) in CLIENTS.items()
    if client not in CANONICAL_CLIENTS
}
PROVIDER_NATIVE_CONVERSATIONS = [
    pytest.param(
        provider, endpoint, scenario.name, mode, id=f"{provider}-{endpoint}-{scenario.name}-{mode}"
    )
    for provider, endpoint in PROVIDER_CLIENTS
    for scenario in sorted(CASSETTE_DIR.iterdir())
    if scenario.is_dir()
    for mode in ("stream", "non_stream")
    if (scenario / f"{provider}-{endpoint}-{mode}-00.yaml").exists()
]

PROVIDER_RESPONSES_CONVERSATIONS = [
    pytest.param(case.values[0], case.values[2], case.values[3], id=case.id)
    for case in PROVIDER_NATIVE_CONVERSATIONS
    if case.values[1] == "responses"
]


def _provider_turns(provider, scenario, mode, endpoint="responses"):
    return [
        read_yaml(path)
        for path in sorted((CASSETTE_DIR / scenario).glob(f"{provider}-{endpoint}-{mode}-*.yaml"))
    ]


def _sdk_body(body):
    """Keep provider extensions separate from the SDK's native arguments."""
    body = deepcopy(body)
    extension_keys = ("provider",) if "input" in body else ("provider", "reasoning")
    extras = {key: body.pop(key) for key in extension_keys if key in body}
    return body, extras


def _recorded_response(cassette):
    response = cassette["response"]
    if "sse" in response:
        return httpx.Response(
            response["status_code"],
            text="".join(response["sse"]),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        response["status_code"],
        content=response["body_bytes"],
        headers={"content-type": "application/json"},
    )


async def _collect_responses(result, stream):
    if not stream:
        return result, None
    async with result:
        events = [event async for event in result]
    completed = [event.response for event in events if event.type == "response.completed"]
    assert len(completed) == 1
    return completed[0], [event.model_dump() for event in events]


async def _collect_chat(result, stream, *, repair_single_call_index=False):
    if not stream:
        return result, None
    state = ChatCompletionStreamState()
    async with result:
        events = [event async for event in result]
    if repair_single_call_index:
        # The Gemini recordings contain one complete, unindexed call per response.
        # Repair SDK reference copies only; Molto output must already carry indices.
        events = [event.model_copy(deep=True) for event in events]
        tools = [
            tool
            for event in events
            for choice in event.choices
            for tool in choice.delta.tool_calls or []
        ]
        assert len(tools) <= 1
        for tool in tools:
            assert tool.index is None and tool.id
            tool.index = 0
    for event in events:
        state.handle_chunk(event)
    return state.get_final_completion(), [event.model_dump() for event in events]


def _chat_history(message, provider):
    """Reproduce the recording driver's actual SDK-output history."""
    # Fireworks rejects null output-only fields in assistant input.
    history = message.model_dump(exclude_unset=True, exclude_none=provider == "fireworks")
    if provider == "gemini":
        # Gemini rejects null optional message fields; opaque nested nulls stay intact.
        history = {name: value for name, value in history.items() if value is not None}
        assert history.pop("annotations", None) in (None, [])
    history.pop("parsed", None)
    for call in history.get("tool_calls") or []:
        if provider in {"togetherai", "gemini"}:
            call.pop("index", None)
        call["function"].pop("parsed_arguments", None)
    # These providers repeat delta.role; the SDK accumulator concatenates those strings.
    if provider in {"togetherai", "gemini"}:
        assert history["role"] and history["role"].replace("assistant", "") == ""
        history["role"] = "assistant"
    else:
        assert history["role"] == "assistant"
    return history


def _chat_input_body(recorded):
    """Omit empty output-only SDK fields from input copies, preserving source cassettes."""
    body = deepcopy(recorded)
    for message in body["messages"]:
        if message["role"] != "assistant":
            continue
        assert message.pop("annotations", None) in (None, [])
        for field in ("reasoning_content", "tool_calls"):
            if field in message and message[field] is None:
                del message[field]
        for call in message.get("tool_calls", []):
            call.pop("index", None)
    return body


@pytest.mark.parametrize(
    ("provider", "endpoint", "scenario", "mode"), PROVIDER_NATIVE_CONVERSATIONS
)
async def test_provider_native_conversation(provider, endpoint, scenario, mode):
    """Ensures native SDK parity and exact actual-output continuation for recorded provider conversations."""
    cassettes = _provider_turns(provider, scenario, mode, endpoint)
    stream = mode == "stream"
    responses = endpoint == "responses"
    history_key = "input" if responses else "messages"
    path = "/responses" if responses else "/chat/completions"
    collect = _collect_responses if responses else _collect_chat
    history = deepcopy(cassettes[0]["request"]["body"][history_key])
    for turn, cassette in enumerate(cassettes):
        recorded = cassette["request"]["body"]
        base_url = cassette["upstream_url"].removesuffix(path)
        assert history == recorded[history_key]
        wire = {**recorded, history_key: history}
        if not responses:
            wire = _chat_input_body(wire)
        body, extras = _sdk_body(wire)
        calls = []

        def handle(request, cassette=cassette, recorded=wire, calls=calls):
            calls.append(request)
            assert str(request.url) == cassette["upstream_url"]
            assert request.headers["authorization"] == "Bearer test-key"
            assert json_loads(request.content) == recorded
            return _recorded_response(cassette)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
            sdk = AsyncOpenAI(
                api_key="test-key",
                base_url=base_url,
                http_client=transport,
                max_retries=0,
            )
            sdk_result = await (
                sdk.responses.create if responses else sdk.chat.completions.create
            )(**body, extra_body=extras)
            expected, expected_events = await (
                _collect_chat(sdk_result, stream, repair_single_call_index=True)
                if provider == "gemini"
                else collect(sdk_result, stream)
            )
        async with Molto(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        ) as molto:
            actual, actual_events = await collect(
                await getattr(molto, endpoint)(
                    body,
                    provider=provider,
                    api_key="test-key",
                    base_url=base_url,
                    extra_body=extras,
                ),
                stream,
            )
        if stream and responses:
            # Native streaming documents only this omitted-usage normalization.
            for event in expected_events:
                if (
                    event["type"] == "response.completed"
                    and event["response"]["usage"]["input_tokens_details"]["cache_write_tokens"]
                    is None
                ):
                    event["response"]["usage"]["input_tokens_details"]["cache_write_tokens"] = 0
                    expected.usage.input_tokens_details.cache_write_tokens = 0
        assert actual_events == expected_events
        assert actual.model_dump() == expected.model_dump()
        assert len(calls) == 2
        if responses:
            history.extend(item.model_dump(exclude_unset=True) for item in actual.output)
        else:
            history.append(_chat_history(actual.choices[0].message, provider))
        if turn + 1 < len(cassettes):
            next_history = cassettes[turn + 1]["request"]["body"][history_key]
            assert history == next_history[: len(history)]
            suffix = next_history[len(history) :]
            assert suffix and all(
                item.get("role") in {"user", "tool"} or item.get("type") == "function_call_output"
                for item in suffix
            )
            history.extend(deepcopy(suffix))


def _caller_input(items):
    """Read recorded caller data; assistant history must come from actual emitted IR."""
    result = []
    for item in items:
        if item.get("type") == "function_call_output":
            content = item["output"]
            if isinstance(content, str):
                content = [{"type": "input_text", "text": content}]
            result.append(
                ToolReturn(
                    tool_call_id=item["call_id"],
                    content=[TextInput(content=part["text"]) for part in content],
                )
            )
        else:
            assert item["role"] == "user"
            content = item["content"]
            if isinstance(content, str):
                content = [{"type": "input_text", "text": content}]
            assert all(part["type"] == "input_text" for part in content)
            result.append(UserInput(content=[TextInput(content=part["text"]) for part in content]))
    return result


def _expected_responses_wire(recorded):
    """Adjust source copies only for documented normalized representations, never model output."""
    body = deepcopy(recorded)
    for item in body["input"]:
        if item.get("role") == "user" and isinstance(item["content"], str):
            item["content"] = [{"type": "input_text", "text": item["content"]}]
        if item.get("role") == "assistant":
            # IR text parts do not carry their enclosing source message's ID.
            item.pop("id", None)
            # Reconstructed Responses history intentionally omits native phase;
            # the SDK parity path above still preserves the provider's value.
            assert item.get("phase") in (None, "commentary", "final_answer")
            item.pop("phase", None)
            for part in item["content"]:
                part.setdefault("annotations", [])
                part.setdefault("logprobs", [])
        if item.get("type") == "reasoning":
            assert item.pop("format", None) in (None, "unknown")
            item.pop("status", None)
        if item.get("type") == "function_call_output" and isinstance(item["output"], str):
            item["output"] = [{"type": "input_text", "text": item["output"]}]
    for tool in body.get("tools", []):
        assert tool["type"] == "function"
        tool.setdefault("strict", False)
    return body


@pytest.mark.parametrize(
    ("provider", "scenario", "mode"),
    [case for case in PROVIDER_RESPONSES_CONVERSATIONS if case.values[2] == "stream"],
)
async def test_provider_normalized_responses_conversation(provider, scenario, mode):
    """Ensures SDK-visible text, reasoning, calls and usage survive real normalized continuations."""
    cassettes = _provider_turns(provider, scenario, mode)
    first = cassettes[0]["request"]["body"]
    request = NormalizedRequest(
        model=first["model"],
        stream=True,
        max_output_tokens=first["max_output_tokens"],
        input=_caller_input(first["input"]),
    )
    if first.get("tools"):
        request.tools = [
            FunctionTool(**{key: value for key, value in tool.items() if key != "type"})
            for tool in first["tools"]
        ]
    for turn, cassette in enumerate(cassettes):
        recorded = cassette["request"]["body"]
        base_url = cassette["upstream_url"].removesuffix("/responses")
        body, extras = _sdk_body(recorded)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _, cassette=cassette: _recorded_response(cassette)
            )
        ) as transport:
            sdk = AsyncOpenAI(
                api_key="test-key", base_url=base_url, http_client=transport, max_retries=0
            )
            reference, source_events = await _collect_responses(
                await sdk.responses.create(**body, extra_body=extras), True
            )
        expected = _expected_responses_wire(recorded)
        calls = []

        def handle(http_request, expected=expected, cassette=cassette, calls=calls):
            calls.append(http_request)
            assert str(http_request.url) == cassette["upstream_url"]
            assert json_loads(http_request.content) == expected
            return _recorded_response(cassette)

        # Keep captured execution controls; these tests target content and history.
        controls = {
            key: recorded[key]
            for key in (
                "reasoning",
                "tool_choice",
                "parallel_tool_calls",
                "store",
                "provider",
                "include",
            )
            if key in recorded
        }
        client_type = PROVIDER_CLIENTS[provider, "responses"]
        unsigned_summary = any(
            item.type == "reasoning" and item.summary and not item.encrypted_content
            for item in reference.output
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
            client = client_type(
                transport,
                api_key="test-key",
                capabilities=client_type.resolve_capabilities(PROVIDER_CAPABILITIES[provider]),
                base_url=base_url,
                extra_body=controls,
            )
            before = request.model_copy(deep=True)
            if unsigned_summary:
                with pytest.raises(
                    ValueError, match="A Responses reasoning summary requires a signature"
                ):
                    _ = [event async for event in client.events(request)]
                assert len(calls) == 1
                return
            events = [event async for event in client.events(request)]
            assert request == before
        assert len(calls) == 1
        raw = any(item.type == "reasoning" and item.content for item in reference.output)
        parts = _assert_ir(
            events,
            tool_name=request.tools[0].name if request.tools else "read_file",
            source_timestamps=(
                int(source_events[0]["response"]["created_at"]),
                int(reference.created_at),
            ),
            raw_reasoning=raw,
        )
        assert (events[0].id, events[0].model) == (reference.id, reference.model)
        assert (
            "".join(part.content for part in parts if isinstance(part, TextOutput))
            == reference.output_text
        )
        assert [
            (part.tool_call_id, part.tool_name, json_loads(part.args))
            for part in parts
            if isinstance(part, ToolCall)
        ] == [
            (item.call_id, item.name, json_loads(item.arguments))
            for item in reference.output
            if item.type == "function_call"
        ]
        assert [
            (part.content, part.summary, part.signature)
            for part in parts
            if isinstance(part, Thinking)
        ] == [
            (
                "".join(part.text for part in item.content or []),
                "".join(part.text for part in item.summary),
                item.encrypted_content,
            )
            for item in reference.output
            if item.type == "reasoning"
        ]
        assert (
            events[-1].usage.input_tokens,
            events[-1].usage.output_tokens,
            events[-1].usage.total_tokens,
        ) == (
            reference.usage.input_tokens,
            reference.usage.output_tokens,
            reference.usage.total_tokens,
        )
        if turn + 1 < len(cassettes):
            request.input.append(AssistantInput(content=parts))
            suffix = cassettes[turn + 1]["request"]["body"]["input"][
                len(recorded["input"]) + len(reference.output) :
            ]
            assert suffix
            request.input.extend(_caller_input(suffix))


def _responses_caller_input(messages):
    """Read only caller messages; translated assistant history comes from Molto output."""
    result = []
    for message in messages:
        if message["role"] == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            assert message["role"] == "user"
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            assert all(part["type"] == "text" for part in content)
            result.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": part["text"]} for part in content],
                }
            )
    return result


def _expected_chat_wire(recorded):
    """Account for normalized text parts and omitted empty assistant metadata on copies."""
    body = _chat_input_body(recorded)
    for message in body["messages"]:
        if message["role"] in {"user", "tool"} and isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        if message["role"] == "assistant":
            for field in ("audio", "function_call", "refusal", "content"):
                if field in message and message[field] is None:
                    del message[field]
            if message.get("tool_calls") and message.get("content") == "":
                del message["content"]
    return body


@pytest.mark.parametrize(
    ("provider", "endpoint", "scenario", "mode"),
    [
        case
        for case in PROVIDER_NATIVE_CONVERSATIONS
        if case.values[1] == "chat_completions" and case.values[3] == "stream"
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_provider_chat_translated_conversation(provider, endpoint, scenario, mode, stream):
    """Ensures translated Chat preserves SDK semantics and continuation, or rejects opaque state."""
    cassettes = _provider_turns(provider, scenario, mode, endpoint)
    first = cassettes[0]["request"]["body"]
    body = {
        "model": first["model"],
        "stream": stream,
        "max_output_tokens": first["max_completion_tokens"],
        "input": _responses_caller_input(first["messages"]),
    }
    if first.get("tools"):
        body["tools"] = [{"type": "function", **tool["function"]} for tool in first["tools"]]
    for turn, cassette in enumerate(cassettes):
        recorded = cassette["request"]["body"]
        base_url = cassette["upstream_url"].removesuffix("/chat/completions")
        sdk_body, extras = _sdk_body(recorded)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _, cassette=cassette: _recorded_response(cassette)
            )
        ) as transport:
            sdk = AsyncOpenAI(
                api_key="test-key", base_url=base_url, http_client=transport, max_retries=0
            )
            reference, _ = await _collect_chat(
                await sdk.chat.completions.create(**sdk_body, extra_body=extras),
                True,
                repair_single_call_index=provider == "gemini",
            )
        expected = _expected_chat_wire(recorded)
        calls = []

        def handle(request, cassette=cassette, expected=expected, calls=calls):
            calls.append(request)
            assert str(request.url) == cassette["upstream_url"]
            assert json_loads(request.content) == expected
            return _recorded_response(cassette)

        controls = {
            key: recorded[key]
            for key in ("reasoning", "reasoning_effort", "tool_choice", "parallel_tool_calls")
            if key in recorded
        }
        before = deepcopy(body)
        message = reference.choices[0].message
        opaque = any(
            getattr(attachment, "extra_content", None)
            for attachment in (message, *(message.tool_calls or []))
        )
        async with Molto(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        ) as molto:
            with (
                pytest.raises(
                    ValueError, match="Chat Completions extra_content cannot be translated"
                )
                if opaque
                else nullcontext()
            ):
                actual, _ = await _collect_responses(
                    await molto.responses(
                        body,
                        provider=provider,
                        endpoint="chat/completions",
                        api_key="test-key",
                        base_url=base_url,
                        extra_body=controls,
                    ),
                    stream,
                )
        assert body == before
        assert len(calls) == 1
        if opaque:
            return
        assert actual.status == "completed"
        assert (actual.id, actual.model, actual.created_at) == (
            reference.id,
            reference.model,
            reference.created,
        )
        assert actual.output_text == (message.content or "")
        assert [
            (item.call_id, item.name, json_loads(item.arguments))
            for item in actual.output
            if item.type == "function_call"
        ] == [
            (call.id, call.function.name, json_loads(call.function.arguments))
            for call in message.tool_calls or []
        ]
        assert not getattr(message, "reasoning_content", None)
        assert all(item.type in {"message", "function_call"} for item in actual.output)
        assert (
            actual.usage.input_tokens,
            actual.usage.output_tokens,
            actual.usage.total_tokens,
        ) == (
            reference.usage.prompt_tokens,
            reference.usage.completion_tokens,
            reference.usage.total_tokens,
        )
        if turn + 1 < len(cassettes):
            body["input"].extend(item.model_dump(exclude_unset=True) for item in actual.output)
            suffix = cassettes[turn + 1]["request"]["body"]["messages"][
                len(recorded["messages"]) + 1 :
            ]
            assert suffix
            body["input"].extend(_responses_caller_input(suffix))


@pytest.mark.parametrize("provider", ["openrouter", "fireworks"])
@pytest.mark.parametrize("downstream", ["chat_completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_provider_reasoning_public_projection(provider, downstream, stream):
    """Ensures public projections preserve raw reasoning or reject unsigned summaries from actual Responses evidence."""
    cassette = _provider_turns(provider, "tool_use_reasoning", "stream")[0]
    recorded = cassette["request"]["body"]
    base_url = cassette["upstream_url"].removesuffix("/responses")
    sdk_body, extras = _sdk_body(recorded)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: _recorded_response(cassette))
    ) as transport:
        sdk = AsyncOpenAI(
            api_key="test-key", base_url=base_url, http_client=transport, max_retries=0
        )
        reference, _ = await _collect_responses(
            await sdk.responses.create(**sdk_body, extra_body=extras), True
        )
    expected = _expected_responses_wire(recorded)
    body = {
        "model": recorded["model"],
        "messages": deepcopy(recorded["input"]),
        "max_tokens": recorded["max_output_tokens"],
        "stream": stream,
    }
    for message in body["messages"]:
        assert message["role"] == "user"
        if isinstance(message["content"], list):
            for part in message["content"]:
                assert part["type"] == "input_text"
                part["type"] = "text"
    tools = recorded["tools"]
    if downstream == "chat_completions":
        body["tools"] = [
            {
                "type": "function",
                "function": {key: value for key, value in tool.items() if key != "type"},
            }
            for tool in tools
        ]
    else:
        body["tools"] = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in tools
        ]
        expected["stream_options"] = {"include_obfuscation": False}
    controls = {
        key: recorded[key]
        for key in ("reasoning", "tool_choice", "parallel_tool_calls", "store", "provider")
        if key in recorded
    }
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == cassette["upstream_url"]
        assert json_loads(request.content) == expected
        return _recorded_response(cassette)

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as molto:
        if provider == "fireworks":
            assert any(
                item.type == "reasoning" and item.summary and not item.encrypted_content
                for item in reference.output
            )
            with pytest.raises(
                ValueError, match="A Responses reasoning summary requires a signature"
            ):
                result = await getattr(molto, downstream)(
                    body,
                    provider=provider,
                    endpoint="responses",
                    base_url=base_url,
                    api_key="test-key",
                    extra_body=controls,
                )
                if stream:
                    async with result:
                        _ = [event async for event in result]
            assert len(calls) == 1
            return
        result = await getattr(molto, downstream)(
            body,
            provider=provider,
            endpoint="responses",
            base_url=base_url,
            api_key="test-key",
            extra_body=controls,
        )
        if stream:
            actual = None
            state = ChatCompletionStreamState() if downstream == "chat_completions" else None
            async with result:
                async for event in result:
                    if state is not None:
                        state.handle_chunk(event)
                    else:
                        actual = accumulate_event(event=event, current_snapshot=actual)
            if state is not None:
                actual = state.get_final_completion()
        else:
            actual = result
    assert len(calls) == 1
    assert (actual.id, actual.model) == (reference.id, reference.model)
    if downstream == "chat_completions":
        message = actual.choices[0].message
        text = message.content or ""
        reasoning = message.reasoning_content
        calls = [
            (call.id, call.function.name, json_loads(call.function.arguments))
            for call in message.tool_calls or []
        ]
        usage = (actual.usage.prompt_tokens, actual.usage.completion_tokens)
        assert actual.choices[0].finish_reason == "tool_calls"
    else:
        text = "".join(block.text for block in actual.content if block.type == "text")
        reasoning = "".join(block.thinking for block in actual.content if block.type == "thinking")
        assert all(not block.signature for block in actual.content if block.type == "thinking")
        calls = [
            (block.id, block.name, block.input)
            for block in actual.content
            if block.type == "tool_use"
        ]
        usage = (
            actual.usage.input_tokens
            + (actual.usage.cache_read_input_tokens or 0)
            + (actual.usage.cache_creation_input_tokens or 0),
            actual.usage.output_tokens,
        )
        assert actual.stop_reason == "tool_use"
    assert text == reference.output_text
    assert reasoning == "".join(
        part.text for item in reference.output if item.type == "reasoning" for part in item.content
    )
    assert calls == [
        (item.call_id, item.name, json_loads(item.arguments))
        for item in reference.output
        if item.type == "function_call"
    ]
    assert usage == (reference.usage.input_tokens, reference.usage.output_tokens)
