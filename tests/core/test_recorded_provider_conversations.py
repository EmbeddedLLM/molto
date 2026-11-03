"""Replay original provider conversations without regenerating expected history."""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import ValidationError

from molto import Molto
from molto.utils.io import json_loads, read_yaml
from scripts.record_provider_conversation import (
    accumulation_chunk,
    assistant_history,
    verify_accumulation,
)

ROOT = Path(__file__).parents[1] / "cassettes/provider_recordings/2026-11-03"
GROUPS = [
    path.parent
    for path in sorted(ROOT.glob("*/manifest.json"))
    if json.loads(path.read_text())["status"] == "complete"
]


# These are recorded limitations, not accepted native continuation cases. Keep
# the full SDK-produced history unchanged so the local rejection stays visible.
REJECTED_HISTORY = {
    "together-qwen-text-nonstream": {"reasoning_content": "string_type"},
    "together-qwen-text-stream": {
        "reasoning_content": "string_type",
        "tool_calls": "list_type",
        "annotations": "extra_forbidden",
    },
    "together-qwen-tools-stream": {"annotations": "extra_forbidden"},
}


async def collect(result, stream):
    if not stream:
        return result, result.model_dump()
    state = ChatCompletionStreamState()
    events = []
    roles = {}
    async with result:
        async for event in result:
            events.append(event.model_dump())
            state.handle_chunk(accumulation_chunk(event, roles))
    completion = state.get_final_completion()
    verify_accumulation(events, assistant_history(completion.choices[0].message))
    return completion, events


async def replay_conversation(group, rejected_history=None):
    """Replay every SDK turn and assert Molto parity up to an explicit history boundary."""
    manifest = json.loads((group / "manifest.json").read_text())
    recordings = [read_yaml(group / turn["files"][0]) for turn in manifest["turns"]]
    stream = manifest["stream"]
    messages = deepcopy(recordings[0]["request"]["body"]["messages"])
    calls = 0
    rejected = False
    for turn, recording in enumerate(recordings):
        expected_body = recording["request"]["body"]
        assert messages == expected_body["messages"]
        body = {**deepcopy(expected_body), "messages": deepcopy(messages)}
        extra_body = manifest["extra_body"]
        for key in extra_body or {}:
            body.pop(key)

        def handler(request, expected_body=expected_body, recording=recording):
            nonlocal calls
            calls += 1
            assert str(request.url) == manifest["upstream_url"]
            assert json_loads(request.content) == expected_body
            assert request.headers["authorization"] == "Bearer replay-key"
            recorded = recording["response"]
            if "sse" in recorded:
                return httpx.Response(
                    recorded["status_code"],
                    content="".join(recorded["sse"]),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                recorded["status_code"],
                content=recorded["body_bytes"],
                headers={"content-type": "application/json"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            sdk = AsyncOpenAI(
                api_key="replay-key",
                base_url=manifest["upstream_url"].removesuffix("/chat/completions"),
                http_client=transport,
                max_retries=0,
            )
            reference, expected_output = await collect(
                await sdk.chat.completions.create(**body, extra_body=extra_body), stream
            )
        if rejected_history and turn == 1:
            before = calls
            async with Molto(
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            ) as molto:
                with pytest.raises(ValidationError) as error:
                    await molto.chat_completions(
                        body,
                        provider=manifest["provider"],
                        api_key="replay-key",
                        extra_body=extra_body,
                    )
            assert calls == before
            assistant_errors = {
                failure["loc"][-1]: failure["type"]
                for failure in error.value.errors()
                if failure["loc"][:3]
                == ("messages", 1, "ChatCompletionAssistantMessageWithReasoningParam")
            }
            assert assistant_errors == rejected_history
            for field in rejected_history:
                assert messages[1][field] is None
            rejected = True
        if not rejected:
            async with Molto(
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            ) as molto:
                actual, actual_output = await collect(
                    await molto.chat_completions(
                        body,
                        provider=manifest["provider"],
                        api_key="replay-key",
                        extra_body=extra_body,
                    ),
                    stream,
                )
            assert actual_output == expected_output
            assert actual.model_dump() == reference.model_dump()
            history = assistant_history(actual.choices[0].message)
        else:
            # Molto has rejected continuation. Only the independent SDK continues
            # replaying the remaining recorded turns, using its actual output.
            history = assistant_history(reference.choices[0].message)
        messages.append(history)
        if turn + 1 < len(recordings):
            next_messages = recordings[turn + 1]["request"]["body"]["messages"]
            # Compare actual output before appending only recorded new user/tool input.
            assert messages == next_messages[: len(messages)]
            suffix = next_messages[len(messages) :]
            assert suffix and all(message["role"] in ("user", "tool") for message in suffix)
            messages.extend(deepcopy(suffix))
    assert rejected == bool(rejected_history)
    assert calls == (len(recordings) + 1 if rejected else len(recordings) * 2)


@pytest.mark.parametrize(
    "group",
    [path for path in GROUPS if path.name not in REJECTED_HISTORY],
    ids=lambda path: path.name,
)
async def test_recorded_native_conversation_uses_actual_output_as_history(group):
    """Ensures supported complete runs preserve exact SDK results and actual next-turn history."""
    await replay_conversation(group)


@pytest.mark.parametrize(
    "group", [path for path in GROUPS if path.name in REJECTED_HISTORY], ids=lambda path: path.name
)
async def test_recorded_history_has_explicit_native_rejection(group):
    """Ensures recorded incompatible SDK history rejects locally while independent SDK replay succeeds."""
    await replay_conversation(group, rejected_history=REJECTED_HISTORY[group.name])


@pytest.mark.parametrize("downstream", ["responses", "chat_completions", "messages"])
async def test_together_alternative_endpoint_requires_explicit_destination(downstream):
    """Ensures a Together key cannot be sent to a canonical service by an unsupported default route."""
    body = (
        {"model": "example", "input": "Hi"}
        if downstream == "responses"
        else {
            "model": "example",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 20,
        }
    )
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Unconfigured Together endpoint reached HTTP")
            )
        )
    ) as molto:
        with pytest.raises(ValueError, match="explicit base_url"):
            await getattr(molto, downstream)(
                body, provider="togetherai", endpoint="responses", api_key="provider-key"
            )
