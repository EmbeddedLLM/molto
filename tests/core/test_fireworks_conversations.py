"""Replay independently recorded Fireworks protocol conversations and translated output."""

import json
from contextlib import nullcontext
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from molto import Molto
from molto.utils.io import dumps_sse, json_loads, read_yaml
from tests.core.test_interface import _assert_responses_stream_invariants
from tests.core.test_recorded_provider_conversations import collect, replay_conversation
from tests.core.test_together_conversations import _semantics

ROOT = Path(__file__).parents[1] / "cassettes/provider_recordings/2026-11-04"
GROUPS = [
    path.parent
    for path in sorted(ROOT.glob("fireworks-*/manifest.json"))
    if json.loads(path.read_text())["status"] == "complete"
]


@pytest.mark.parametrize("group", GROUPS, ids=lambda path: path.name)
async def test_fireworks_native_conversation_sdk_parity(group):
    """Ensures native Fireworks output and actual subsequent history match independent SDK replay."""
    manifest = json.loads((group / "manifest.json").read_text())
    await replay_conversation(group, protocol=manifest["upstream_protocol"])


def translated_request(recorded, upstream, downstream, stream):
    """Build only the captured user text and function definitions in another protocol."""
    history = recorded["input" if upstream == "responses" else "messages"]
    assert len(history) == 1 and history[0]["role"] == "user"
    content = history[0]["content"]
    parts = [content] if isinstance(content, str) else [part["text"] for part in content]
    body = {"model": recorded["model"], "stream": stream}
    body["input" if downstream == "responses" else "messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "input_text" if downstream == "responses" else "text", "text": part}
                for part in parts
            ],
        }
    ]
    limit = next(
        recorded[key]
        for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
        if key in recorded
    )
    body["max_output_tokens" if downstream == "responses" else "max_tokens"] = limit
    if recorded.get("tools"):
        tool = recorded["tools"][0]
        function = tool["function"] if upstream == "chat/completions" else tool
        payload = {
            "name": function["name"],
            "description": function["description"],
            "parameters": function["input_schema" if upstream == "messages" else "parameters"],
            "strict": False,
        }
        body["tools"] = [
            {"type": "function", "function": payload}
            if downstream == "chat/completions"
            else {"type": "function", **payload}
            if downstream == "responses"
            else {
                "name": payload["name"],
                "description": payload["description"],
                "input_schema": payload["parameters"],
            }
        ]
    return body


@pytest.mark.parametrize(
    ("group", "downstream"),
    [
        pytest.param(group, downstream, id=f"{group.name}-{downstream}")
        for group in GROUPS
        if json.loads((group / "manifest.json").read_text())["stream"]
        for downstream in ("chat/completions", "responses", "messages")
        if downstream != json.loads((group / "manifest.json").read_text())["upstream_protocol"]
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_fireworks_translated_first_turn_sdk_semantics(group, downstream, stream):
    """Ensures recorded first-turn content, calls, usage and lifecycle survive public translation."""
    manifest = json.loads((group / "manifest.json").read_text())
    upstream = manifest["upstream_protocol"]
    recording = read_yaml(group / manifest["turns"][0]["files"][0])
    recorded = recording["request"]["body"]
    body = translated_request(recorded, upstream, downstream, stream)
    expected = translated_request(recorded, upstream, upstream, True)
    # This offline output smoke keeps captured provider controls as explicit overrides;
    # it does not claim the translated request was itself executed against Fireworks.
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
    expected.update(controls)
    if upstream == "chat/completions":
        expected["max_completion_tokens"] = expected.pop("max_tokens")
        expected["stream_options"] = {"include_usage": True}
    if downstream == "messages" and upstream != "messages":
        expected.setdefault("stream_options", {})["include_obfuscation"] = False
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        assert str(request.url) == manifest["upstream_url"]
        assert json_loads(request.content) == expected
        return httpx.Response(
            200,
            text="".join(recording["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                text="".join(recording["response"]["sse"]),
                headers={"content-type": "text/event-stream"},
            )
        )
    ) as transport:
        sdk = (
            AsyncAnthropic(api_key="test", http_client=transport)
            if upstream == "messages"
            else AsyncOpenAI(api_key="test", http_client=transport)
        )
        create = (
            sdk.messages.create
            if upstream == "messages"
            else sdk.responses.create
            if upstream == "responses"
            else sdk.chat.completions.create
        )
        reference, _ = await collect(await create(**recorded), True, upstream)
    rejection = None
    if upstream == "messages" and any(
        block.type == "thinking" and block.thinking and not block.signature
        for block in reference.content
    ):
        rejection = "An Anthropic reasoning summary requires a signature."
    if upstream == "responses" and any(
        item.type == "reasoning" and item.summary and not item.encrypted_content
        for item in reference.output
    ):
        rejection = "A Responses reasoning summary requires a signature."
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match=rejection) if rejection else nullcontext():
        async with Molto(http_client=transport) as molto:
            result = await getattr(molto, downstream.replace("/", "_"))(
                body, provider="fireworks", endpoint=upstream, api_key="test", extra_body=controls
            )
            actual, events = await collect(result, stream, downstream)
    assert calls == 1
    assert transport.is_closed
    if rejection:
        return
    expected_semantics = _semantics(reference, upstream)
    actual_semantics = _semantics(actual, downstream)
    assert actual_semantics == expected_semantics
    if stream and downstream == "responses":
        _assert_responses_stream_invariants([dumps_sse(event) for event in events])

    if stream and downstream == "messages":
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
        starts = [event["index"] for event in events if event["type"] == "content_block_start"]
        stops = [event["index"] for event in events if event["type"] == "content_block_stop"]
        assert starts == stops == list(range(len(starts)))
    if stream and downstream == "chat/completions":
        assert sum(bool(event["usage"]) for event in events) == 1
        assert (
            sum(
                choice["finish_reason"] is not None
                for event in events
                for choice in event["choices"]
            )
            == 1
        )
