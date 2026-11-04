#!/usr/bin/env python3
"""Record controlled native Chat conversations independently of Molto translation.

Run explicitly outside pytest. Each invocation creates a fresh staging directory;
review its original recordings before promotion. SDK accumulation failures are
evidence, and deliberately stop the conversation instead of repairing history.
"""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import traceback
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState

from scripts import recorder_proxy


def assistant_history(message) -> dict:
    """Remove only SDK bookkeeping and empty output-only annotations."""
    history = message.model_dump(exclude_unset=True)
    history.pop("parsed", None)
    if history.get("annotations") == []:
        history.pop("annotations")
    for call in history.get("tool_calls") or []:
        call.pop("index", None)
        if "function" in call:
            call["function"].pop("parsed_arguments", None)
    return history


def accumulation_chunk(event, roles: dict[int, str]):
    """Remove repeated role snapshots from a copy before SDK string accumulation."""
    chunk = event.model_copy(deep=True)
    for choice in chunk.choices:
        role = choice.delta.role
        if role is None:
            continue
        previous = roles.get(choice.index)
        if previous is not None and role != previous:
            raise ValueError("Provider changed the role during a streamed choice.")
        if previous is not None:
            # Together repeats the complete role, whereas the SDK concatenates strings.
            choice.delta = type(choice.delta).model_validate(
                choice.delta.model_dump(exclude_unset=True, exclude={"role"})
            )
        roles[choice.index] = role
    return chunk


def verify_accumulation(events: list[dict], history: dict) -> None:
    """Fail if SDK accumulation changes reasoning bytes or opaque attachment values."""
    reasoning = {"reasoning": [], "reasoning_content": []}
    for event in events:
        for choice in event["choices"]:
            if choice["index"] != 0:
                raise ValueError("The recording scenario requires exactly one choice.")
            delta = choice["delta"]
            if delta.get("role") is not None and delta["role"] != history.get("role"):
                raise ValueError("Accumulated role differs from the original provider metadata.")
            for field, fragments in reasoning.items():
                if delta.get(field) is not None:
                    if not isinstance(delta[field], str):
                        raise ValueError(
                            "Cannot independently accumulate non-string reasoning deltas."
                        )
                    fragments.append(delta[field])
            owners = [(delta, history)]
            for call in delta.get("tool_calls") or []:
                owners.append((call, history["tool_calls"][call["index"]]))
            for source, collected in owners:
                if "extra_content" in source and source["extra_content"] != collected.get(
                    "extra_content"
                ):
                    raise ValueError(
                        "SDK accumulation changed an opaque extra_content attachment."
                    )
    for field, fragments in reasoning.items():
        if fragments and history.get(field) != "".join(fragments):
            raise ValueError("SDK accumulation changed reasoning history.")


def validate_turn(history: dict, finish_reason: str, scenario: str, turn: int) -> None:
    """Require the complete requested conversation before marking provider evidence complete."""
    if history.get("role") != "assistant":
        raise ValueError("SDK accumulation did not retain the assistant role.")
    calls = history.get("tool_calls") or []
    expected = []
    if scenario == "tools" and turn < 2:
        expected = [[2, 3]] if turn == 0 else [[4, 5]]
    elif scenario == "parallel" and turn == 0:
        expected = [[2, 3], [4, 7]]
    if len(calls) != len(expected):
        raise ValueError("Provider did not return the required number of tool calls.")
    if finish_reason != ("tool_calls" if expected else "stop"):
        raise ValueError("Provider did not complete the requested turn.")
    if not calls:
        if not isinstance(history.get("content"), str) or not history["content"].strip():
            raise ValueError("Provider did not return a final text answer.")
        return
    identities = [call.get("id") for call in calls]
    if any(not isinstance(identity, str) or not identity for identity in identities) or len(
        set(identities)
    ) != len(identities):
        raise ValueError("Provider tool identities are missing or duplicated.")
    arguments = []
    for call in calls:
        function = call["function"]
        if function["name"] != "add":
            raise ValueError("Provider returned an unexpected tool.")
        values = json.loads(function["arguments"])
        if set(values) != {"a", "b"} or any(type(value) is not int for value in values.values()):
            raise ValueError("Provider returned invalid add arguments.")
        arguments.append(sorted(values.values()))
    if sorted(arguments) != expected:
        raise ValueError("Provider did not execute the requested tool scenario.")


async def record(args: argparse.Namespace) -> int:
    load_dotenv(override=False)
    key = os.environ.get(args.key_env)
    if not key:
        raise ValueError(f"Configure {args.key_env} before recording.")
    url = urlsplit(args.base_url)
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Use a base URL without embedded credentials, query, or fragment.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    recorder_proxy.OUTPUT_DIR = args.output_dir
    recorder_proxy.TARGET_HOST = f"{url.scheme}://{url.netloc}"
    manifest = {
        "provenance": "recorded",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "diff", "--name-only"])),
        "python": platform.python_version(),
        "source_hashes": {
            "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "recorder_sha256": sha256(Path(recorder_proxy.__file__).read_bytes()).hexdigest(),
        },
        "versions": {name: version(name) for name in ("openai", "anthropic", "httpx")},
        "provider": args.provider,
        "upstream_protocol": "chat/completions",
        "upstream_url": args.base_url.rstrip("/") + "/chat/completions",
        "requested_model": args.model,
        "authentication": f"Bearer credential from {args.key_env}; value excluded",
        "scenario": args.scenario,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "extra_body": args.extra_body,
        "sdk_retries": 0,
        "proxy_retries": 0,
        "accumulation_policy": (
            "OpenAI SDK accumulation with repeated identical role snapshots omitted on copies; "
            "original SSE and role metadata retained and checked independently"
        ),
        "turns": [],
        "status": "incomplete",
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Remember the exact phrase: blue  lantern."},
                {"type": "text", "text": "Reply briefly."},
            ],
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    if args.scenario != "text":
        messages = [
            {
                "role": "user",
                "content": (
                    "Use add to add 2 and 3. After the result, use add to add that result and 4. "
                    "Then give the final answer."
                    if args.scenario == "tools"
                    else "Call add twice in parallel: first 2 and 3, then 7 and 4. Then give both results."
                ),
            }
        ]
    exit_code = 0
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=recorder_proxy.app)
        ) as transport:
            async with AsyncOpenAI(
                api_key=key,
                base_url="http://recorder.local" + url.path,
                http_client=transport,
                max_retries=0,
                timeout=120,
            ) as client:
                for turn in range(3 if args.scenario == "tools" else 2):
                    before = set(args.output_dir.glob("*.yaml"))
                    turn_info = {"turn": turn, "status": "incomplete"}
                    manifest["turns"].append(turn_info)
                    body = {
                        "model": args.model,
                        "messages": messages,
                        "stream": args.stream,
                        "max_tokens": args.max_tokens,
                    }
                    if args.stream:
                        body["stream_options"] = {"include_usage": True}
                    if args.scenario != "text":
                        body["tools"] = tools
                        if args.scenario == "parallel":
                            body["parallel_tool_calls"] = True
                            body["tool_choice"] = "required" if turn == 0 else "none"
                        else:
                            body["tool_choice"] = (
                                "none"
                                if turn == 2
                                else {"type": "function", "function": {"name": "add"}}
                            )
                    try:
                        result = await client.chat.completions.create(
                            **body, extra_body=args.extra_body
                        )
                        if args.stream:
                            state = ChatCompletionStreamState()
                            events = []
                            roles = {}
                            async with result:
                                async for event in result:
                                    events.append(event.model_dump(exclude_unset=True))
                                    state.handle_chunk(accumulation_chunk(event, roles))
                            result = state.get_final_completion()
                        turn_info["returned_model"] = result.model
                        if len(result.choices) != 1:
                            raise ValueError("The recording scenario requires exactly one choice.")
                        choice = result.choices[0]
                        turn_info["finish_reason"] = choice.finish_reason
                        history = assistant_history(choice.message)
                        if args.stream:
                            verify_accumulation(events, history)
                        validate_turn(history, choice.finish_reason, args.scenario, turn)
                        messages.append(history)
                        calls = history.get("tool_calls") or []
                        for call in calls:
                            function = call["function"]
                            arguments = json.loads(function["arguments"])
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "content": str(arguments["a"] + arguments["b"]),
                                }
                            )
                        if args.scenario == "text" and turn == 0:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Repeat the phrase"},
                                        {
                                            "type": "text",
                                            "text": " exactly, preserving its spaces.",
                                        },
                                    ],
                                }
                            )
                        turn_info["status"] = "complete"
                    finally:
                        turn_info["files"] = [
                            p.name for p in sorted(set(args.output_dir.glob("*.yaml")) - before)
                        ]
        manifest["status"] = "complete"
    except Exception as error:
        # Exception strings may contain echoed headers or credentials; retain the
        # class and stack locations while the proxy retains the reviewed response.
        manifest["failure"] = {
            "type": type(error).__name__,
            "traceback": traceback.format_tb(error.__traceback__),
        }
        exit_code = 1
    finally:
        await recorder_proxy.HTTP_CLIENT.aclose()
        (args.output_dir / "manifest.json").write_text(
            json.dumps(recorder_proxy.mask_secrets(manifest, [key]), indent=2) + "\n"
        )
    print(f"{args.provider}/{args.scenario}: {manifest['status']}; {args.output_dir}")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=("text", "tools", "parallel"), default="text")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--extra-body",
        type=json.loads,
        default=None,
        help="Nonsecret provider settings only; persisted verbatim in the manifest.",
    )
    return asyncio.run(record(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
