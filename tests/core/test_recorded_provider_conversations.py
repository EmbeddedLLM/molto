"""Replay original provider conversations without regenerating expected history."""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic
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
from tests.core.history_helpers import _collect_response

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
        "annotations": "extra_forbidden",
    },
    "together-qwen-tools-stream": {"annotations": "extra_forbidden"},
}


async def collect(result, stream, protocol="chat/completions"):
    if not stream:
        return result, result.model_dump()
    if protocol != "chat/completions":
        async with result:
            events = [event async for event in result]
        return _collect_response(events, protocol), [event.model_dump() for event in events]
    state = ChatCompletionStreamState()
    events = []
    roles = {}
    async with result:
        async for event in result:
            events.append(event.model_dump(exclude_unset=True))
            state.handle_chunk(accumulation_chunk(event, roles))
    completion = state.get_final_completion()
    verify_accumulation(events, assistant_history(completion.choices[0].message))
    return completion, events


def response_history(response, protocol, *, events=None):
    """Retain native output fields and remove only incremental SDK parser bookkeeping."""
    if protocol == "chat/completions":
        return [assistant_history(response.choices[0].message, events=events)]
    if protocol == "responses":
        return [item.model_dump(exclude_unset=True) for item in response.output]
    return [
        {
            "role": response.role,
            "content": [
                block.model_dump(exclude_unset=True, exclude={"__json_buf", "parsed_output"})
                for block in response.content
            ],
        }
    ]


async def replay_conversation(group, rejected_history=None, *, protocol="chat/completions"):
    """Replay every SDK turn and assert Molto parity up to an explicit history boundary."""
    manifest = json.loads((group / "manifest.json").read_text())
    recordings = [read_yaml(group / turn["files"][0]) for turn in manifest["turns"]]
    stream = manifest["stream"]
    history_key = "input" if protocol == "responses" else "messages"
    messages = deepcopy(recordings[0]["request"]["body"][history_key])
    calls = 0
    rejected = False
    for turn, recording in enumerate(recordings):
        expected_body = recording["request"]["body"]
        assert messages == expected_body[history_key]
        body = {**deepcopy(expected_body), history_key: deepcopy(messages)}
        extra_body = manifest.get("extra_body")
        for key in extra_body or {}:
            body.pop(key)

        def handler(request, expected_body=expected_body, recording=recording):
            nonlocal calls
            calls += 1
            assert str(request.url) == manifest["upstream_url"]
            assert json_loads(request.content) == expected_body
            if protocol == "messages":
                assert request.headers["x-api-key"] == "replay-key"
            else:
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
            sdk_type = AsyncAnthropic if protocol == "messages" else AsyncOpenAI
            suffix = "/v1/messages" if protocol == "messages" else f"/{protocol}"
            sdk = sdk_type(
                api_key="replay-key",
                base_url=manifest["upstream_url"].removesuffix(suffix),
                http_client=transport,
                max_retries=0,
            )
            create = (
                sdk.messages.create
                if protocol == "messages"
                else sdk.responses.create
                if protocol == "responses"
                else sdk.chat.completions.create
            )
            reference, expected_output = await collect(
                await create(**body, extra_body=extra_body), stream, protocol
            )
        if rejected_history and turn == 1:
            before = calls
            async with Molto(
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            ) as molto:
                with pytest.raises(ValidationError) as error:
                    await getattr(molto, protocol.replace("/", "_"))(
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
                    await getattr(molto, protocol.replace("/", "_"))(
                        body,
                        provider=manifest["provider"],
                        api_key="replay-key",
                        extra_body=extra_body,
                    ),
                    stream,
                    protocol,
                )
            if protocol == "responses" and stream:
                # Native Responses already documents omitted cache-write usage as zero.
                # Keep every other independently SDK-parsed field exact.
                reference = reference.model_copy(deep=True)
                for event in expected_output:
                    if event["type"] == "response.completed":
                        details = event["response"]["usage"]["input_tokens_details"]
                        if details["cache_write_tokens"] is None:
                            details["cache_write_tokens"] = 0
                            reference.usage.input_tokens_details.cache_write_tokens = 0
            assert actual_output == expected_output
            assert actual.model_dump() == reference.model_dump()
            history = response_history(
                actual,
                protocol,
                events=actual_output
                if stream and manifest.get("history_policy") == "omit-absent-sdk-null-bookkeeping"
                else None,
            )
        else:
            # Molto has rejected continuation. Only the independent SDK continues
            # replaying the remaining recorded turns, using its actual output.
            history = response_history(
                reference,
                protocol,
                events=expected_output
                if stream and manifest.get("history_policy") == "omit-absent-sdk-null-bookkeeping"
                else None,
            )
        messages.extend(history)
        if turn + 1 < len(recordings):
            next_messages = recordings[turn + 1]["request"]["body"][history_key]
            # Compare actual output before appending only recorded new user/tool input.
            assert messages == next_messages[: len(messages)]
            suffix = next_messages[len(messages) :]
            assert suffix and all(
                message.get("role") in ("user", "tool")
                or message.get("type") == "function_call_output"
                for message in suffix
            )
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
