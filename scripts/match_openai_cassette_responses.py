#!/usr/bin/env python3
"""Prepare a Chat Completions cassette for non-streaming response parity.

The Responses cassette is canonical. This script rewrites the representable
request and JSON response fields in a Chat Completions cassette while preserving
its native wire format and transport metadata. Unsigned raw Responses reasoning
maps to Chat Completions `reasoning_content`; reasoning summaries and encrypted
continuation state are not representable in Chat Completions.

Hosted-search fixtures cover Chat downstream only. Hosted calls and structured
URL citations are omitted while actual answer text and inline links are retained.
Signed or encrypted reasoning is rejected, including on hosted-search turns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from molto.utils.io import dump_yaml, read_yaml

if __package__:
    from .match_openai_cassette_events import _match_search_request
else:
    from match_openai_cassette_events import _match_search_request


def _request_body(cassette: dict[str, Any]) -> dict[str, Any]:
    try:
        request_body = cassette["request"]["body"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cassette must contain request.body.") from exc
    if not isinstance(request_body, dict):
        raise ValueError("Cassette request.body must be an object.")
    return request_body


def _response_message_content(
    content: Any, *, preserve_parts: bool = False
) -> str | list[dict[str, str]]:
    if isinstance(content, str):
        if preserve_parts:
            return [{"text": content, "type": "text"}]
        return content
    if not isinstance(content, list):
        raise ValueError("Responses message content must be text or a list of text parts.")

    text: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {
            "input_text",
            "output_text",
            "text",
        }:
            raise ValueError("Only Responses text message content can map to Chat Completions.")
        part_text = part.get("text")
        if not isinstance(part_text, str):
            raise ValueError("Responses text content must contain a string text field.")
        text.append(part_text)
    if preserve_parts:
        return [{"text": part, "type": "text"} for part in text]
    return "".join(text)


def _raw_reasoning_content(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        raise ValueError("Responses reasoning item must contain a content list.")
    if item.get("summary") not in (None, []):
        raise ValueError("Responses reasoning summaries cannot map to Chat Completions.")
    if item.get("encrypted_content") is not None:
        raise ValueError("Encrypted Responses reasoning cannot map to Chat Completions.")

    reasoning: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "reasoning_text":
            raise ValueError("Only raw Responses reasoning can map to Chat Completions.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Responses reasoning text must be a string.")
        reasoning.append(text)
    return "".join(reasoning)


def _chat_messages_from_response_input(response_input: Any) -> list[dict[str, Any]]:
    if isinstance(response_input, str):
        return [{"content": response_input, "role": "user"}]
    if not isinstance(response_input, list):
        raise ValueError("Responses input must be a string or an input-item list.")

    messages: list[dict[str, Any]] = []
    tool_call_ids: set[str] = set()
    returned_tool_call_ids: set[str] = set()
    for item in response_input:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects.")

        item_type = item.get("type", "message")
        if item_type == "web_search_call":
            continue
        if item_type == "reasoning":
            messages.append(
                {
                    "reasoning_content": _raw_reasoning_content(item),
                    "role": "assistant",
                }
            )
            continue

        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not all(isinstance(value, str) for value in (call_id, name, arguments)):
                raise ValueError(
                    "Responses function calls require string call_id, name, and arguments."
                )
            if call_id in tool_call_ids:
                raise ValueError(f"Duplicate Responses function call ID: {call_id!r}.")
            if not messages or messages[-1].get("role") != "assistant":
                messages.append({"role": "assistant"})
            messages[-1].setdefault("tool_calls", []).append(
                {
                    "function": {"arguments": arguments, "name": name},
                    "id": call_id,
                    "type": "function",
                }
            )
            tool_call_ids.add(call_id)
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            if not isinstance(call_id, str):
                raise ValueError("Responses function call outputs require a string call_id.")
            if call_id not in tool_call_ids:
                raise ValueError(f"Responses function call output has unknown ID: {call_id!r}.")
            if call_id in returned_tool_call_ids:
                raise ValueError(f"Duplicate Responses function call output ID: {call_id!r}.")
            messages.append(
                {
                    "content": _response_message_content(output, preserve_parts=True),
                    "role": "tool",
                    "tool_call_id": call_id,
                }
            )
            returned_tool_call_ids.add(call_id)
            continue

        if item_type != "message":
            raise ValueError(f"Responses input type {item_type!r} cannot map to Chat Completions.")
        role = item.get("role")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"Responses input role {role!r} cannot map to Chat Completions.")
        message = {
            "content": _response_message_content(
                item.get("content"), preserve_parts=role == "user"
            ),
            "role": role,
        }
        if (
            role == "assistant"
            and messages
            and messages[-1].get("role") == "assistant"
            and ("content" not in messages[-1])
        ):
            messages[-1].update(message)
        else:
            messages.append(message)
    return messages


def _match_request_reasoning_effort(
    chat_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    chat_request = _request_body(chat_cassette)
    responses_request = _request_body(responses_cassette)
    reasoning = responses_request.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    response_input = responses_request.get("input")
    has_reasoning_history = isinstance(response_input, list) and any(
        isinstance(item, dict) and item.get("type") == "reasoning" for item in response_input
    )
    if effort is not None or has_reasoning_history:
        chat_request["reasoning_effort"] = effort or "high"
    else:
        chat_request.pop("reasoning_effort", None)


def _map_usage(response_usage: dict[str, Any], chat_usage: dict[str, Any]) -> None:
    chat_usage["prompt_tokens"] = response_usage["input_tokens"]
    chat_usage["completion_tokens"] = response_usage["output_tokens"]
    chat_usage["total_tokens"] = response_usage["total_tokens"]

    input_details = response_usage.get("input_tokens_details") or {}
    prompt_details = {
        field: input_details[field]
        for field in ("cache_write_tokens", "cached_tokens")
        if field in input_details
    }
    if prompt_details:
        chat_usage["prompt_tokens_details"] = (
            chat_usage.get("prompt_tokens_details") or {}
        ) | prompt_details

    output_details = response_usage.get("output_tokens_details") or {}
    if "reasoning_tokens" in output_details:
        chat_usage["completion_tokens_details"] = (
            chat_usage.get("completion_tokens_details") or {}
        ) | {"reasoning_tokens": output_details["reasoning_tokens"]}


def _response_body(cassette: dict[str, Any]) -> dict[str, Any]:
    try:
        response_body = cassette["response"]["body"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cassette must contain response.body.") from exc
    if not isinstance(response_body, dict):
        raise ValueError("Cassette response.body must be an object.")
    return response_body


def _require_non_streaming(cassette: dict[str, Any]) -> None:
    stream = _request_body(cassette).get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise ValueError("Cassette request stream must be a boolean when provided.")
    if stream:
        raise ValueError("Non-streaming response matching does not accept stream=true.")


def _response_output(
    response: dict[str, Any],
    *,
    has_search: bool = False,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses response output must be a list.")

    text: list[str] = []
    has_text = False
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("Responses output items must be objects.")
        item_type = item.get("type")
        if item_type == "web_search_call":
            continue
        if item_type == "reasoning":
            if reasoning_content is not None:
                raise ValueError(
                    "Multiple Responses reasoning items cannot map to Chat Completions."
                )
            reasoning_content = _raw_reasoning_content(item)
            continue
        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not all(isinstance(value, str) for value in (call_id, name, arguments)):
                raise ValueError(
                    "Responses function calls require string call_id, name, and arguments."
                )
            tool_calls.append(
                {
                    "function": {"arguments": arguments, "name": name},
                    "id": call_id,
                    "type": "function",
                }
            )
            continue
        if item_type != "message" or item.get("role") != "assistant":
            raise ValueError(
                f"Responses output type {item_type!r} cannot map to Chat Completions."
            )

        content = item.get("content")
        if not isinstance(content, list):
            raise ValueError("Responses output message content must be a list.")
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "output_text":
                raise ValueError("Only Responses output text can map to Chat Completions.")
            part_text = part.get("text")
            if not isinstance(part_text, str):
                raise ValueError("Responses output text must contain a string text field.")
            if any(
                not has_search or annotation.get("type") != "url_citation"
                for annotation in part.get("annotations") or []
            ):
                raise ValueError("Annotated Responses output cannot map to Chat Completions.")
            if part.get("logprobs") not in (None, []):
                raise ValueError("Responses output logprobs cannot map to Chat Completions.")
            text.append(part_text)
            has_text = True
    return reasoning_content, "".join(text) if has_text else None, tool_calls


def _finish_reason(response: dict[str, Any], *, has_tool_calls: bool) -> str:
    status = response.get("status")
    if status == "completed":
        return "tool_calls" if has_tool_calls else "stop"
    if status != "incomplete":
        raise ValueError("Responses response status must be completed or incomplete.")

    incomplete_details = response.get("incomplete_details")
    if incomplete_details is None:
        return "stop"
    if not isinstance(incomplete_details, dict):
        raise ValueError("Incomplete Responses incomplete_details must be an object or null.")
    return {
        "content_filter": "content_filter",
        "max_output_tokens": "length",
    }.get(incomplete_details.get("reason"), "stop")


def _validated_usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Responses response must contain usage for response matching.")
    token_counts = [
        usage.get(field) for field in ("input_tokens", "output_tokens", "total_tokens")
    ]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in token_counts):
        raise ValueError("Responses usage token counts must be integers.")
    if token_counts[0] + token_counts[1] != token_counts[2]:
        raise ValueError("Responses total tokens do not equal input plus output tokens.")
    return usage


def match_chat_completions_response(
    chat_cassette: dict[str, Any], responses_cassette: dict[str, Any]
) -> None:
    """Rewrite one Chat Completions JSON response from a canonical Responses body."""
    _require_non_streaming(chat_cassette)
    _require_non_streaming(responses_cassette)

    chat_request = _request_body(chat_cassette)
    responses_request = _request_body(responses_cassette)
    chat_request["messages"] = _chat_messages_from_response_input(responses_request.get("input"))
    has_search = _match_search_request(chat_cassette, responses_cassette)
    _match_request_reasoning_effort(chat_cassette, responses_cassette)

    chat_response = _response_body(chat_cassette)
    response = _response_body(responses_cassette)
    has_search = has_search or any(
        isinstance(item, dict) and item.get("type") == "web_search_call"
        for item in response.get("output", [])
    )
    choices = chat_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Chat Completions response must contain exactly one choice.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Chat Completions choice must contain a message object.")
    chat_usage = chat_response.get("usage")
    if not isinstance(chat_usage, dict):
        raise ValueError("Chat Completions response must contain usage.")

    response_id = response.get("id")
    created_at = response.get("created_at")
    model = response.get("model")
    if not isinstance(response_id, str) or not isinstance(model, str):
        raise ValueError("Responses response must contain string id and model fields.")
    if not isinstance(created_at, int) or isinstance(created_at, bool):
        raise ValueError("Responses response created_at must be an integer.")

    reasoning_content, content, tool_calls = _response_output(response, has_search=has_search)
    message["role"] = "assistant"
    message["content"] = content
    message.pop("reasoning_content", None)
    message.pop("tool_calls", None)
    if has_search:
        message.pop("annotations", None)
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls

    choice = choices[0]
    choice["finish_reason"] = _finish_reason(response, has_tool_calls=bool(tool_calls))
    chat_response["id"] = response_id
    chat_response["created"] = created_at
    chat_response["model"] = model
    chat_response["object"] = "chat.completion"
    chat_response["service_tier"] = "default"
    _map_usage(_validated_usage(response), chat_usage)
    # Matched output intentionally replaces the original recorded response bytes.
    chat_cassette["response"].pop("body_bytes", None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Chat Completions cassette for non-streaming protocol parity "
            "against canonical Responses. The Chat Completions cassette is "
            "updated in place unless --output is provided."
        )
    )
    parser.add_argument("chat_completions_cassette", type=Path)
    parser.add_argument("responses_cassette", type=Path)
    parser.add_argument("--output", type=Path, help="Write to a different YAML file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    chat_cassette = read_yaml(str(args.chat_completions_cassette))
    responses_cassette = read_yaml(str(args.responses_cassette))
    if not isinstance(chat_cassette, dict) or not isinstance(responses_cassette, dict):
        raise ValueError("Cassette roots must be YAML mappings.")

    match_chat_completions_response(chat_cassette, responses_cassette)
    output_path = args.output or args.chat_completions_cassette
    dump_yaml(chat_cassette, str(output_path), sort_keys=True)
    print(f"Matched Chat Completions response and wrote {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
