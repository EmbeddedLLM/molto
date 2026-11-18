"""Opt-in live checks for one explicit provider/model, without recording fixtures.

Run with ``python -m scripts.smoke_provider --help`` from the repository root.
Passing the required arguments authorizes live requests, which may incur provider charges.
"""

import argparse
import asyncio
import json
from contextlib import aclosing
from copy import deepcopy
from typing import get_args

import httpx

from molto import Molto
from molto.providers import Endpoint, Provider
from molto.utils.test import collect_sdk_response

PROTOCOLS = get_args(Endpoint)
FUNCTION = {
    "name": "add",
    "description": "Add two integers.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    },
    "strict": False,
}


def request_body(protocol, model, stream, max_tokens, *, tools=False):
    prompt = "Call add with a=2 and b=3, then report the result." if tools else "Reply with OK."
    body = {"model": model, "stream": stream}
    if protocol == "responses":
        body.update(input=[{"role": "user", "content": prompt}], max_output_tokens=max_tokens)
        if tools:
            body.update(
                tools=[{"type": "function", **deepcopy(FUNCTION)}],
                tool_choice={"type": "function", "name": "add"},
            )
    else:
        body["messages"] = [{"role": "user", "content": prompt}]
        body["max_tokens" if protocol == "messages" else "max_completion_tokens"] = max_tokens
        if tools:
            if protocol == "messages":
                body.update(
                    tools=[
                        {
                            "name": "add",
                            "description": FUNCTION["description"],
                            "input_schema": deepcopy(FUNCTION["parameters"]),
                        }
                    ],
                    tool_choice={"type": "tool", "name": "add"},
                )
            else:
                body.update(
                    tools=[{"type": "function", "function": deepcopy(FUNCTION)}],
                    tool_choice={"type": "function", "function": {"name": "add"}},
                )
    return body


def response_parts(protocol, response):
    """Read observable content and portable calls from each SDK response."""
    if protocol == "responses":
        assert response.status == "completed", "Response did not complete"
        return response.output_text, [
            (item.call_id, item.name, json.loads(item.arguments))
            for item in response.output
            if item.type == "function_call"
        ]
    if protocol == "messages":
        calls = [
            (item.id, item.name, item.input)
            for item in response.content
            if item.type == "tool_use"
        ]
        assert response.stop_reason == ("tool_use" if calls else "end_turn"), (
            "Unexpected Messages stop reason"
        )
        return "".join(item.text for item in response.content if item.type == "text"), calls
    choice = response.choices[0]
    assert choice.finish_reason in {"stop", "tool_calls"}, "Unexpected Chat finish reason"
    return choice.message.content or "", [
        (item.id, item.function.name, json.loads(item.function.arguments))
        for item in choice.message.tool_calls or []
    ]


def append_tool_result(protocol, body, response, call_id):
    """Carry actual output forward, removing only SDK output bookkeeping."""
    if protocol == "responses":
        body["input"].extend(
            item.model_dump(exclude_unset=True, exclude={"status"}) for item in response.output
        )
        body["input"].append({"type": "function_call_output", "call_id": call_id, "output": "5"})
    elif protocol == "messages":
        blocks = [block.model_dump(exclude_unset=True) for block in response.content]
        for block in blocks:
            if block.get("caller") == {"type": "direct"}:
                block.pop("caller")
            if block.get("type") == "thinking" and not block.get("signature"):
                block.pop("signature", None)
        body["messages"].extend(
            [
                {"role": "assistant", "content": blocks},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": call_id, "content": "5"}],
                },
            ]
        )
    else:
        history = response.choices[0].message.model_dump(exclude_unset=True)
        assert history.pop("annotations", None) in (None, []), "Annotated history needs a policy"
        history.pop("parsed", None)
        for field in ("reasoning_content", "tool_calls"):
            if history.get(field) is None:
                history.pop(field, None)
        # Some compatible providers repeat role in each chunk; the SDK concatenates it.
        assert history["role"] and not history["role"].replace("assistant", "")
        history["role"] = "assistant"
        for call in history.get("tool_calls", []):
            call.pop("index", None)
            call["function"].pop("parsed_arguments", None)
        body["messages"].extend(
            [history, {"role": "tool", "tool_call_id": call_id, "content": "5"}]
        )
    body["tool_choice"] = "none" if protocol != "messages" else {"type": "none"}


async def complete(client, protocol, body, options):
    coordinator = getattr(client, protocol.replace("/", "_"))
    response = await coordinator(body=body, **options)
    if body["stream"]:
        async with aclosing(response):
            return collect_sdk_response(protocol, [event async for event in response])
    return response


async def run(args):
    """Run native and selected translated checks with finite HTTP and wall-clock budgets."""
    request_count = 0
    failures = 0

    async def count_request(request):
        nonlocal request_count
        if request_count >= args.max_requests:
            raise RuntimeError("Smoke HTTP request budget exhausted")
        request_count += 1

    options = {"provider": args.provider, "endpoint": args.upstream, "base_url": args.base_url}
    if args.extra_body:
        options["extra_body"] = args.extra_body
    protocols = list(dict.fromkeys([args.upstream, *(args.downstream or PROTOCOLS)]))
    modes = [False, True] if args.mode == "both" else [args.mode == "stream"]
    try:
        async with (
            asyncio.timeout(args.deadline),
            Molto(
                http_client=httpx.AsyncClient(
                    timeout=args.timeout, event_hooks={"request": [count_request]}
                )
            ) as client,
        ):
            for protocol in protocols:
                for stream in modes:
                    for tools in (False, True):
                        label = f"{protocol} {'stream' if stream else 'non-stream'} {'tool' if tools else 'text'}"
                        try:
                            body = request_body(
                                protocol, args.model, stream, args.max_tokens, tools=tools
                            )
                            response = await complete(client, protocol, body, options)
                            text, calls = response_parts(protocol, response)
                            if tools:
                                assert len(calls) == 1 and calls[0][1:] == (
                                    "add",
                                    {"a": 2, "b": 3},
                                ), "Unexpected tool call"
                                append_tool_result(protocol, body, response, calls[0][0])
                                text, calls = response_parts(
                                    protocol, await complete(client, protocol, body, options)
                                )
                                assert "5" in text, "Tool result absent from final answer"
                            assert text.strip() and not calls, (
                                "Final answer is missing or calls a tool"
                            )
                            print(f"PASS {label}", flush=True)
                        except Exception as error:
                            failures += 1
                            status = getattr(error, "status_code", None)
                            print(
                                f"FAIL {label}: {type(error).__name__} status={status}", flush=True
                            )
                            if request_count >= args.max_requests:
                                return 1
    except TimeoutError:
        failures += 1
        print("FAIL run deadline exceeded", flush=True)
    finally:
        print(
            f"HTTP attempts: {request_count}/{args.max_requests}; failed cases: {failures}",
            flush=True,
        )
    return int(bool(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=get_args(Provider))
    parser.add_argument("--model", required=True)
    parser.add_argument("--upstream", required=True, choices=PROTOCOLS)
    parser.add_argument("--downstream", action="append", choices=PROTOCOLS)
    parser.add_argument("--base-url")
    parser.add_argument(
        "--extra-body", help="Provider options as JSON; never include credentials."
    )
    parser.add_argument("--mode", choices=("both", "stream", "non-stream"), default="both")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--deadline", type=float, default=180)
    args = parser.parse_args(argv)
    if min(args.max_tokens, args.max_requests, args.timeout, args.deadline) <= 0:
        parser.error("Token, request, and time limits must be positive")
    if args.extra_body:
        try:
            args.extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError:
            parser.error("--extra-body must be a valid JSON object")
        if not isinstance(args.extra_body, dict):
            parser.error("--extra-body must be a JSON object")
        if args.extra_body.keys() & {
            "model",
            "input",
            "messages",
            "tools",
            "tool_choice",
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
        }:
            parser.error(
                "--extra-body cannot override smoke inputs, tools, modes, or token limits"
            )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
