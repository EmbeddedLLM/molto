import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState

from molto.utils.io import read_yaml
from scripts import recorder_proxy

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "gemini-recordings"


async def main():
    key = os.environ["GEMINI_API_KEY"]
    assert key
    logger.remove()
    recorder_proxy.TARGET_HOST = "https://generativelanguage.googleapis.com"
    try:
        for scenario, turns in [("text_chat", 2), ("tool_use", 3)]:
            for stream in [True, False]:
                mode = "stream" if stream else "non_stream"
                directory = STAGING / scenario / mode
                directory.mkdir(parents=True, exist_ok=False)
                recorder_proxy.OUTPUT_DIR = directory
                template = read_yaml(
                    ROOT
                    / "tests/cassettes/upstream_ir"
                    / scenario
                    / "openai-chat_completions-stream-00.yaml"
                )["request"]["body"]
                body = deepcopy(template)
                body.update(
                    model="gemini-3.1-flash-lite",
                    stream=stream,
                    reasoning_effort="minimal",
                    max_completion_tokens=1024,
                )
                if not stream:
                    body.pop("stream_options", None)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=recorder_proxy.app)
                ) as transport:
                    async with AsyncOpenAI(
                        api_key=key,
                        base_url="http://recorder.local/v1beta/openai",
                        http_client=transport,
                        max_retries=0,
                        timeout=90,
                    ) as sdk:
                        for turn in range(turns):
                            before = set(directory.glob("*.yaml"))
                            async with asyncio.timeout(90):
                                result = await sdk.chat.completions.create(**body)
                                if stream:
                                    state = ChatCompletionStreamState()
                                    async with result:
                                        async for chunk in result:
                                            # Single-call prompts make a missing stream index unambiguous.
                                            # Only the SDK accumulator sees this copy; recorded wire stays intact.
                                            chunk = chunk.model_copy(deep=True)
                                            for delta_choice in chunk.choices:
                                                tools = delta_choice.delta.tool_calls or []
                                                assert len(tools) <= 1
                                                for tool in tools:
                                                    if tool.index is None:
                                                        tool.index = 0
                                            state.handle_chunk(chunk)
                                    result = state.get_final_completion()
                            choice = result.choices[0]
                            assert choice.finish_reason in {"stop", "tool_calls"}
                            message = choice.message
                            calls = message.tool_calls or []
                            if scenario == "tool_use" and turn < 2:
                                assert len(calls) == 1
                                assert calls[0].function.name == "read_file"
                                assert json.loads(calls[0].function.arguments) == {
                                    "path": "index.txt" if turn == 0 else "garden.txt"
                                }
                            else:
                                assert not calls
                                assert message.content and message.content.strip()
                            created = set(directory.glob("*.yaml")) - before
                            assert len(created) == 1
                            filename = directory / f"gemini-chat_completions-{mode}-{turn:02}.yaml"
                            created.pop().rename(filename)
                            capture = read_yaml(filename)
                            assert capture["response"]["status_code"] == 200
                            if stream:
                                assert capture["response"]["stream_status"] == "complete"
                                assert capture["response"]["terminal_event"] == "[DONE]"
                            assert key not in filename.read_text()
                            print(f"{scenario} {mode} turn {turn}: completed", flush=True)
                            if turn + 1 < turns:
                                history = message.model_dump(exclude_unset=True)
                                history.pop("parsed", None)
                                assert history.pop("annotations", None) in (None, [])
                                for tool in history.get("tool_calls") or []:
                                    tool.pop("index", None)
                                    tool["function"].pop("parsed_arguments", None)
                                body["messages"].append(history)
                                if scenario == "text_chat":
                                    body["messages"].append(
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "Explain why that name fits.",
                                                }
                                            ],
                                        }
                                    )
                                else:
                                    body["messages"].append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": calls[0].id,
                                            "content": "garden.txt"
                                            if turn == 0
                                            else "Water the tomatoes every morning and harvest ripe fruit.",
                                        }
                                    )
    finally:
        await recorder_proxy.HTTP_CLIENT.aclose()


try:
    asyncio.run(main())
except Exception as error:
    print(
        f"Recording failed: {type(error).__name__}; HTTP {getattr(error, 'status_code', None)}",
        flush=True,
    )
    details = getattr(error, "body", None)
    if details:
        sanitized = json.dumps(details).replace(os.environ["GEMINI_API_KEY"], "***")
        print(sanitized[:2000], flush=True)
    raise SystemExit(1) from None
