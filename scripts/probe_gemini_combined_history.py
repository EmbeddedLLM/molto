"""Record bounded native Gemini reasoning/signature continuation, outside pytest."""

import asyncio
import copy
import json
import os
from pathlib import Path

import httpx
import yaml
from loguru import logger
from openai import AsyncOpenAI

from scripts import recorder_proxy as recorder

ROOT = Path("/tmp/molto-gemini-combined-history")
MODEL = "gemini-3.1-flash-lite"
RESULTS = []
ATTEMPTS = 0
TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}


async def count(request):
    global ATTEMPTS
    ATTEMPTS += 1
    assert ATTEMPTS <= 4, "Request budget exceeded"


def normalize_thought(message):
    # Construct the continuation shape from the returned marker and its opaque state.
    google = message.get("extra_content", {}).get("google", {})
    if google.pop("thought", None) is True and message.get("content"):
        message["reasoning_content"] = message.pop("content")
    if "extra_content" in message and not google:
        del message["extra_content"]


async def invoke(sdk, body):
    response = await sdk.chat.completions.create(**body)
    if not body["stream"]:
        message = response.choices[0].message.model_dump(exclude_none=True, exclude_unset=True)
        normalize_thought(message)
        return (
            message,
            response.model_dump(),
            response.choices[0].finish_reason,
        )
    message = {"role": "assistant"}
    calls = {}
    events = []
    finish = None
    async with response:
        async for chunk in response:
            events.append(chunk.model_dump())
            for choice in chunk.choices:
                finish = choice.finish_reason or finish
                delta = choice.delta.model_dump(exclude_none=True, exclude_unset=True)
                normalize_thought(delta)
                for name in ("content", "reasoning_content"):
                    if delta.get(name):
                        message[name] = message.get(name, "") + delta[name]
                if delta.get("extra_content"):
                    message["extra_content"] = delta["extra_content"]
                for call in delta.get("tool_calls", []):
                    identity = call.get("id") or next(iter(calls), None)
                    assert identity, "Missing initial call identity"
                    target = calls.setdefault(
                        identity,
                        {
                            "id": identity,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    function = call.get("function", {})
                    for name in ("name", "arguments"):
                        if function.get(name):
                            target["function"][name] += function[name]
                    if call.get("extra_content"):
                        target["extra_content"] = call["extra_content"]
    if calls:
        message["tool_calls"] = list(calls.values())
    return message, events, finish


async def capture(sdk, body, label, key):
    before = set(ROOT.glob("*.yaml"))
    message, events, finish = await invoke(sdk, body)
    files = set(ROOT.glob("*.yaml")) - before
    assert len(files) == 1
    path = files.pop()
    target = ROOT / (label + ".yaml")
    path.rename(target)
    assert key.encode() not in target.read_bytes(), "Secret masking failed"
    recorded = yaml.safe_load(target.read_text())
    assert recorded["response"]["status_code"] == 200
    if body["stream"]:
        assert recorded["response"]["stream_status"] == "complete"

    def replay(request):
        assert json.loads(request.content) == recorded["request"]["body"]
        saved = recorded["response"]
        content = "".join(saved["sse"]).encode() if "sse" in saved else saved["body_bytes"]
        return httpx.Response(
            200, headers={"content-type": saved["headers"]["content-type"]}, content=content
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(replay)) as transport:
        independent = AsyncOpenAI(
            api_key="replay-placeholder",
            base_url="https://replay.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        replayed, actual, replay_finish = await invoke(independent, body)
    assert events == actual and message == replayed and finish == replay_finish
    row = {
        "file": target.name,
        "finish": finish,
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "text_chars": len(message.get("content") or ""),
        "message_signature": bool(
            message.get("extra_content", {}).get("google", {}).get("thought_signature")
        ),
        "calls": len(message.get("tool_calls") or []),
        "sdk_replay": "passed",
    }
    RESULTS.append(row)
    print(json.dumps(row), flush=True)
    assert finish in ("stop", "tool_calls"), "Incomplete generation"
    return message


async def main():
    key = os.environ["GEMINI_API_KEY"]
    ROOT.mkdir(parents=True, exist_ok=True)
    recorder.OUTPUT_DIR = ROOT
    recorder.TARGET_HOST = "https://generativelanguage.googleapis.com"
    recorder.TIMEOUT = 45
    logger.remove()
    await recorder.HTTP_CLIENT.aclose()
    recorder.HTTP_CLIENT = httpx.AsyncClient(timeout=45, event_hooks={"request": [count]})
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=recorder.app), timeout=50
        ) as transport:
            sdk = AsyncOpenAI(
                api_key=key,
                base_url="http://recording.local/v1beta/openai",
                http_client=transport,
                max_retries=0,
                timeout=45,
            )
            # Each source is a genuine assistant message returned for the same prompt.
            # Keep the visible answer and its original signature together in both arms.
            seeds = Path("/tmp/molto-gemini-history-seeds")

            def prior_request(mode):
                path = seeds / f"reasoning_text-gemini-chat_completions-{mode}-01.yaml"
                return yaml.safe_load(path.read_text())["request"]["body"]

            visible = prior_request("stream")
            marked = prior_request("non_stream")
            assert visible["messages"][0] == marked["messages"][0]
            signed = visible["messages"][1]
            assert signed["content"] and signed["extra_content"]["google"]["thought_signature"]
            reasoning = marked["messages"][1]["reasoning_content"]
            assert reasoning and "reasoning_content" not in signed
            for stream in (False, True):
                for combined in (False, True):
                    wire = copy.deepcopy(visible)
                    wire["stream"] = stream
                    if combined:
                        wire["messages"][1]["reasoning_content"] = reasoning
                    # The recorded wire has a top-level extra_body key used by Google's API.
                    wire["extra_body"] = {"extra_body": wire["extra_body"]}
                    mode = "stream" if stream else "non_stream"
                    scenario = "combined" if combined else "control"
                    reply = await capture(
                        sdk, wire, f"{scenario}-gemini-chat_completions-{mode}-01", key
                    )
                    assert "36" in (reply.get("content") or reply.get("reasoning_content", "")), (
                        "Unexpected continuation"
                    )
                    RESULTS.append({"scenario": scenario, "mode": mode, "continuation": "passed"})
    finally:
        await recorder.HTTP_CLIENT.aclose()
        report = {
            "model": MODEL,
            "thinking_level": "medium",
            "include_thoughts": True,
            "requests_sent": ATTEMPTS,
            "results": RESULTS,
        }
        (ROOT / "results.json").write_text(json.dumps(report, indent=2))
        assert all(key.encode() not in p.read_bytes() for p in ROOT.iterdir() if p.is_file())
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=480))
