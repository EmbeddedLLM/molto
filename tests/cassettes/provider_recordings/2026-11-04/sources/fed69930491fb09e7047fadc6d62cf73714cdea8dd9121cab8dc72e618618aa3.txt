#!/usr/bin/env python3
"""Record native Responses or Messages conversations independently of Molto.

Use the Chat driver for Chat Completions. This companion retains actual Responses
output or SDK-collected Messages blocks as history and stops on incomplete turns.
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
from anthropic import AsyncAnthropic
from anthropic import omit as anthropic_omit
from dotenv import load_dotenv
from openai import AsyncOpenAI, omit

from scripts import record_provider_conversation, recorder_proxy
from scripts.record_provider_conversation import validate_turn


def initial_history(protocol: str, scenario: str) -> list[dict]:
    if scenario == "text":
        part_type = "input_text" if protocol == "responses" else "text"
        return [
            {
                "role": "user",
                "content": [
                    {"type": part_type, "text": "Remember the exact phrase: blue  lantern."},
                    {"type": part_type, "text": "Reply briefly."},
                ],
            }
        ]
    prompt = (
        "Use add to add 2 and 3. After the result, use add to add that result and 4. Then give the final answer."
        if scenario == "tools"
        else "Call add twice in parallel: first 2 and 3, then 7 and 4. Then give both results."
    )
    return [{"role": "user", "content": prompt}]


def request_body(args, history: list[dict], turn: int) -> dict:
    body = {"model": args.model}
    if args.extra_body is not None:
        body["extra_body"] = args.extra_body
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    final = turn == (2 if args.scenario == "tools" else 1)
    if args.protocol == "responses":
        body.update(
            input=history,
            store=False,
            max_output_tokens=args.max_tokens,
            reasoning={"effort": args.reasoning},
        )
        if args.scenario != "text":
            body["tools"] = [
                {
                    "type": "function",
                    "name": "add",
                    "description": "Add two integers.",
                    "parameters": schema,
                }
            ]
            body["tool_choice"] = (
                "none"
                if final
                else "required"
                if args.scenario == "parallel"
                else {"type": "function", "name": "add"}
            )
            body["parallel_tool_calls"] = args.scenario == "parallel"
    else:
        body.update(messages=history, max_tokens=args.max_tokens)
        body["thinking"] = (
            {"type": "disabled"}
            if args.reasoning == "none"
            else {"type": "enabled", "budget_tokens": 1024}
        )
        if args.scenario != "text":
            body["tools"] = [
                {"name": "add", "description": "Add two integers.", "input_schema": schema}
            ]
            body["tool_choice"] = (
                {"type": "none"}
                if final
                else {"type": "auto"}
                if args.reasoning != "none"
                else {"type": "any", "disable_parallel_tool_use": False}
                if args.scenario == "parallel"
                else {"type": "tool", "name": "add", "disable_parallel_tool_use": True}
            )
    return body


async def result_and_history(client, args, body: dict):
    if args.protocol == "responses":
        result = await client.responses.create(**body, stream=args.stream)
        if args.stream:
            terminal = None
            async with result:
                async for event in result:
                    if event.type in {
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    }:
                        if terminal is not None:
                            raise ValueError("Multiple Responses terminal events.")
                        terminal = event.response
            if terminal is None:
                raise ValueError("Responses stream has no terminal result.")
            result = terminal
        if result.status != "completed":
            raise ValueError("Responses result did not complete.")
        content = [item.model_dump(exclude_unset=True) for item in result.output]
        calls = [
            {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
            for item in content
            if item["type"] == "function_call"
        ]
        text = "".join(
            part["text"]
            for item in content
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
        reason = "tool_calls" if calls else "stop"
        history = content
    else:
        if args.stream:
            async with client.messages.stream(**body) as stream:
                stops = 0
                async for event in stream:
                    if event.type == "message_stop":
                        stops += 1
                if stops != 1:
                    raise ValueError("Messages stream must contain exactly one message_stop.")
                result = await stream.get_final_message()
        else:
            result = await client.messages.create(**body, stream=False)
        content = [
            part.model_dump(exclude_unset=True, exclude={"__json_buf", "parsed_output"})
            for part in result.content
        ]
        calls = [
            {
                "id": part["id"],
                "type": "function",
                "function": {"name": part["name"], "arguments": json.dumps(part["input"])},
            }
            for part in content
            if part["type"] == "tool_use"
        ]
        text = "".join(part["text"] for part in content if part["type"] == "text")
        reason = {"tool_use": "tool_calls", "end_turn": "stop"}.get(
            result.stop_reason, result.stop_reason
        )
        history = [{"role": "assistant", "content": content}]
    return result, history, {"role": "assistant", "content": text, "tool_calls": calls}, reason


async def record(args) -> int:
    load_dotenv(override=False)
    key = os.environ.get(args.key_env)
    if not key:
        raise ValueError(f"Configure {args.key_env} before recording.")
    if args.protocol == "messages" and any(
        line.partition(":")[0].strip().lower() in {"authorization", "x-api-key"}
        for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines()
    ):
        raise ValueError(
            "Remove authentication entries from ANTHROPIC_CUSTOM_HEADERS before recording."
        )
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
        "versions": {name: version(name) for name in ("openai", "anthropic", "httpx")},
        "source_hashes": {
            name: sha256(Path(path).read_bytes()).hexdigest()
            for name, path in {
                "driver_sha256": __file__,
                "recorder_sha256": recorder_proxy.__file__,
                "validation_sha256": record_provider_conversation.__file__,
            }.items()
        },
        "provider": args.provider,
        "upstream_protocol": args.protocol,
        "upstream_url": args.base_url.rstrip("/")
        + ("/responses" if args.protocol == "responses" else "/v1/messages"),
        "requested_model": args.model,
        "scenario": args.scenario,
        "stream": args.stream,
        "reasoning_setting": args.reasoning,
        "max_tokens": args.max_tokens,
        "extra_body": args.extra_body,
        "authentication": f"{'Bearer' if args.protocol == 'responses' or args.auth == 'bearer' else 'x-api-key'} credential from {args.key_env}; value excluded",
        "sdk_retries": 0,
        "proxy_retries": 0,
        "accumulation_policy": "Responses terminal output or Anthropic SDK final content; model_dump(exclude_unset=True); exclude only __json_buf/parsed_output SDK bookkeeping; manual actual-output history",
        "turns": [],
        "status": "incomplete",
    }
    history = initial_history(args.protocol, args.scenario)
    exit_code = 0
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=recorder_proxy.app)
        ) as transport:
            client_type = AsyncOpenAI if args.protocol == "responses" else AsyncAnthropic
            auth = {"api_key": key}
            if args.protocol == "messages":
                auth = {
                    "api_key": None if args.auth == "bearer" else key,
                    "auth_token": key if args.auth == "bearer" else None,
                }
            async with client_type(
                **auth,
                base_url="http://recorder.local" + url.path,
                http_client=transport,
                default_headers=(
                    {
                        "Authorization": f"Bearer {key}",
                        "OpenAI-Organization": omit,
                        "OpenAI-Project": omit,
                    }
                    if args.protocol == "responses"
                    else {
                        "Authorization": f"Bearer {key}"
                        if args.auth == "bearer"
                        else anthropic_omit,
                        "X-Api-Key": anthropic_omit if args.auth == "bearer" else key,
                    }
                ),
                max_retries=0,
                timeout=120,
            ) as client:
                for turn in range(3 if args.scenario == "tools" else 2):
                    before = set(args.output_dir.glob("*.yaml"))
                    info = {"turn": turn, "status": "incomplete"}
                    manifest["turns"].append(info)
                    try:
                        result, output, validation, reason = await result_and_history(
                            client, args, request_body(args, history, turn)
                        )
                        info["returned_model"] = result.model
                        info["finish_reason"] = reason
                        validate_turn(validation, reason, args.scenario, turn)
                        history.extend(output)
                        tool_results = []
                        for call in validation["tool_calls"]:
                            values = json.loads(call["function"]["arguments"])
                            answer = str(values["a"] + values["b"])
                            if args.protocol == "responses":
                                history.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": call["id"],
                                        "output": answer,
                                    }
                                )
                            else:
                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": call["id"],
                                        "content": answer,
                                    }
                                )
                        if tool_results:
                            history.append({"role": "user", "content": tool_results})
                        if args.scenario == "text" and turn == 0:
                            part_type = "input_text" if args.protocol == "responses" else "text"
                            history.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": part_type, "text": "Repeat the phrase"},
                                        {
                                            "type": part_type,
                                            "text": " exactly, preserving its spaces.",
                                        },
                                    ],
                                }
                            )
                        info["status"] = "complete"
                    finally:
                        info["files"] = [
                            p.name for p in sorted(set(args.output_dir.glob("*.yaml")) - before)
                        ]
        manifest["status"] = "complete"
    except Exception as error:
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
    print(
        f"{args.provider}/{args.protocol}/{args.scenario}: {manifest['status']}; {args.output_dir}"
    )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--protocol", choices=("responses", "messages"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--auth", choices=("api-key", "bearer"), default="api-key")
    parser.add_argument(
        "--extra-body",
        type=json.loads,
        default=None,
        help="Nonsecret provider settings only; persisted verbatim in the manifest.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=("text", "tools", "parallel"), default="text")
    parser.add_argument("--reasoning", choices=("none", "low"), default="none")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2048)
    return asyncio.run(record(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
