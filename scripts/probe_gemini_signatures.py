"""Bounded live signature-placement experiment; never records pytest cassettes."""

import asyncio
import copy
import json
import os
from pathlib import Path

from openai import APIStatusError, AsyncOpenAI

MODEL = "gemini-3.1-flash-lite"
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
MESSAGES = [
    {
        "role": "user",
        "content": "Say 'I will add these numbers' as ordinary assistant text, then call add with a=2 and b=3. After receiving the tool result, report it.",
    }
]
RESULTS = []
ATTEMPTS = 0


def emit(data):
    RESULTS.append(data)
    print(json.dumps(data), flush=True)
    Path("/tmp/molto-gemini-signature-probe-results.json").write_text(
        json.dumps(RESULTS, indent=2)
    )


def signature(message):
    return message.get("extra_content", {}).get("google", {}).get("thought_signature")


async def request(client, label, messages, choice):
    global ATTEMPTS
    ATTEMPTS += 1
    assert ATTEMPTS <= 8
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[TOOL],
            tool_choice=choice,
            reasoning_effort="minimal",
            max_completion_tokens=256,
            stream=False,
        )
    except APIStatusError as exc:
        # Emit only a fixed classification, never provider payloads or credentials.
        body = str(exc.body).lower()
        flags = [
            s
            for s in (
                "location",
                "permission",
                "quota",
                "thought_signature",
                "missing",
                "invalid",
                "not found",
            )
            if s in body
        ]
        emit(
            {
                "case": label,
                "status": exc.status_code,
                "error_class": type(exc).__name__,
                "error_flags": flags,
            }
        )
        return None
    except Exception as exc:
        emit({"case": label, "error_class": type(exc).__name__})
        return None
    msg = response.choices[0].message.model_dump(exclude_none=True, exclude_unset=True)
    msg.pop("annotations", None)
    msg.pop("parsed", None)
    calls = msg.get("tool_calls") or []
    emit(
        {
            "case": label,
            "status": 200,
            "finish_reason": response.choices[0].finish_reason,
            "has_text": bool(msg.get("content")),
            "tool_count": len(calls),
            "message_signature": bool(signature(msg)),
            "tool_signatures": [bool(signature(c)) for c in calls],
            "reports_five": "5" in (msg.get("content") or "")
            or "five" in (msg.get("content") or "").lower(),
        }
    )
    return msg


async def run():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        emit({"error": "Gemini credential unavailable"})
        raise SystemExit(2)
    async with asyncio.timeout(240):
        async with AsyncOpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            max_retries=0,
            timeout=35,
        ) as client:
            original = await request(client, "generate_native_text_and_tool", MESSAGES, "auto")
            if original is None:
                raise SystemExit(2)
            if not original.get("tool_calls"):
                original = await request(
                    client,
                    "generate_native_forced_tool",
                    MESSAGES,
                    {"type": "function", "function": {"name": "add"}},
                )
            if original is None or len(original.get("tool_calls") or []) != 1:
                emit({"error": "No single native tool call available"})
                raise SystemExit(2)
            call = original["tool_calls"][0]
            args = json.loads(call["function"]["arguments"])
            result = {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": str(args["a"] + args["b"]),
            }
            control = await request(
                client, "native_continuation_control", MESSAGES + [original, result], "none"
            )
            if control is None:
                raise SystemExit(2)
            source = original
            synthetic = not bool(signature(source))
            if synthetic:
                source = await request(
                    client, "generate_native_message_signature", MESSAGES, "none"
                )
            if source is None or not signature(source):
                emit({"error": "No native message-level signature available"})
                raise SystemExit(2)
            message = copy.deepcopy(original)
            message["content"] = source.get("content") or "I will add these numbers."
            message["extra_content"] = copy.deepcopy(source["extra_content"])
            message["tool_calls"][0].pop("extra_content", None)
            emit(
                {
                    "comparison": "message_to_tool",
                    "synthetic_combination": synthetic,
                    "original_tool_signature_removed": bool(signature(call)),
                    "model": MODEL,
                    "reasoning_effort": "minimal",
                }
            )
            for with_text in (True, False):
                before = copy.deepcopy(message)
                if not with_text:
                    before.pop("content", None)
                after = copy.deepcopy(before)
                after["tool_calls"][0]["extra_content"] = after.pop("extra_content")
                suffix = "with_text" if with_text else "without_text"
                await request(
                    client, "message_attachment_" + suffix, MESSAGES + [before, result], "none"
                )
                await request(
                    client, "moved_to_tool_" + suffix, MESSAGES + [after, result], "none"
                )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as exc:
        emit({"experiment_error_class": type(exc).__name__})
        raise SystemExit(2) from None
