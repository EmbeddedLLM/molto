"""Replay recorded OpenRouter free-model conversations without live provider requests."""

import json
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError
from openai.lib.streaming.chat import ChatCompletionStreamState

from molto import Molto
from molto.utils.io import json_loads, read_yaml
from scripts.record_provider_conversation import (
    accumulation_chunk,
    assistant_history,
    verify_accumulation,
)
from tests.core.test_recorded_provider_conversations import (
    replay_conversation,
    replay_translated_first_turn,
)

ROOT = Path(__file__).parents[1] / "cassettes/provider_recordings/2026-11-04"
GROUPS = [
    path.parent
    for path in sorted(ROOT.glob("openrouter-*/manifest.json"))
    if json.loads(path.read_text())["status"] == "complete"
]


@pytest.mark.parametrize("group", GROUPS, ids=lambda path: path.name)
async def test_openrouter_native_conversation_sdk_parity(group):
    """Ensures native OpenRouter output and actual next-turn history match independent SDK replay."""
    manifest = json.loads((group / "manifest.json").read_text())
    rejection = (
        {"reasoning.str": "string_type", "reasoning.list[str]": "list_type"}
        if group.name
        in (
            "openrouter-chat-text-nonstream-none",
            "openrouter-chat-tools-nonstream-none",
            "openrouter-chat-parallel-nonstream-none",
        )
        else None
    )
    await replay_conversation(
        group, rejected_history=rejection, protocol=manifest["upstream_protocol"]
    )


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
async def test_openrouter_translated_first_turn_sdk_semantics(group, downstream, stream):
    """Ensures recorded content, tools, usage and explicit translation boundaries match SDK output."""
    await replay_translated_first_turn(group, downstream, stream)


@pytest.mark.parametrize("stream", [False, True])
async def test_openrouter_liquid_mandatory_reasoning_error_sdk_parity(stream):
    """Ensures both native modes preserve Liquid's recorded HTTP400 mandatory-reasoning error."""
    group = ROOT / f"openrouter-chat-text-{'stream' if stream else 'nonstream'}-none-liquid"
    manifest = json.loads((group / "manifest.json").read_text())
    recording = read_yaml(group / manifest["turns"][0]["files"][0])
    recorded = recording["request"]["body"]
    extras = manifest["extra_body"]
    body = {key: value for key, value in recorded.items() if key not in extras}
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        assert str(request.url) == manifest["upstream_url"]
        assert json_loads(request.content) == recorded
        assert request.headers["authorization"] == "Bearer replay-key"
        response = recording["response"]
        return httpx.Response(
            response["status_code"],
            content=response["body_bytes"],
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        sdk = AsyncOpenAI(
            api_key="replay-key",
            base_url="https://openrouter.ai/api/v1",
            http_client=transport,
            max_retries=0,
        )
        with pytest.raises(BadRequestError) as reference:
            await sdk.chat.completions.create(**body, extra_body=extras)
    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        with pytest.raises(BadRequestError) as actual:
            await molto.chat_completions(
                body, provider="openrouter", api_key="replay-key", extra_body=extras
            )
    assert calls == 2
    assert actual.value.status_code == reference.value.status_code == 400
    assert actual.value.body == reference.value.body == recording["response"]["body"]["error"]
    assert actual.value.message == reference.value.message
    assert (
        actual.value.response.content
        == reference.value.response.content
        == recording["response"]["body_bytes"]
    )
    assert actual.value.response.is_closed and reference.value.response.is_closed


async def test_openrouter_low_reasoning_native_events_preserve_sdk_accumulation_failure():
    """Ensures raw native events remain exact while corrupted SDK reasoning history is rejected."""
    group = ROOT / "openrouter-chat-tools-stream-low"
    manifest = json.loads((group / "manifest.json").read_text())
    recording = read_yaml(group / manifest["turns"][0]["files"][0])
    recorded = recording["request"]["body"]
    extras = manifest["extra_body"]
    body = {key: value for key, value in recorded.items() if key not in extras}
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        assert str(request.url) == manifest["upstream_url"]
        assert json_loads(request.content) == recorded
        return httpx.Response(
            200,
            text="".join(recording["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        sdk = AsyncOpenAI(
            api_key="replay-key",
            base_url="https://openrouter.ai/api/v1",
            http_client=transport,
            max_retries=0,
        )
        async with await sdk.chat.completions.create(**body, extra_body=extras) as result:
            reference = [event async for event in result]
        async with Molto(http_client=transport) as molto:
            async with await molto.chat_completions(
                body, provider="openrouter", api_key="replay-key", extra_body=extras
            ) as result:
                actual = [event.model_dump(exclude_unset=True) async for event in result]
    expected = [event.model_dump(exclude_unset=True) for event in reference]
    assert actual == expected
    assert calls == 2
    state = ChatCompletionStreamState()
    roles = {}
    for event in reference:
        state.handle_chunk(accumulation_chunk(event, roles))
    history = assistant_history(state.get_final_completion().choices[0].message, events=expected)
    with pytest.raises(ValueError, match="SDK accumulation changed reasoning_details history"):
        verify_accumulation(expected, history)
