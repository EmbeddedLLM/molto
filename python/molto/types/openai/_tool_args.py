from __future__ import annotations

import json
from typing import Any


def _safe_tool_args(arguments: str | None) -> tuple[dict[str, Any], str | None]:
    """Convert OpenAI tool argument strings into IR-safe fields.

    Chat Completions and Responses both represent function-call arguments as a
    string that is expected to contain a JSON object. Molto's IR keeps that
    object in `AgentToolCall.args`, but it also has `AgentToolCall.raw_args` for
    compatibility with existing history that contains malformed JSON or valid
    JSON with a non-object top-level value.

    Return a parsed dict only when the string is a JSON object. For invalid JSON
    and non-object JSON, return an empty `args` dict and preserve the original
    string in `raw_args` so it can be round-tripped without crashing conversion
    or silently changing the caller's data. Empty strings and `None` are treated
    as "no arguments" and do not set `raw_args`.
    """
    if not arguments:
        return {}, None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}, arguments
    if isinstance(parsed, dict):
        return parsed, None
    return {}, arguments


def _serialize_tool_args(args: dict[str, Any], raw_args: str | None) -> str:
    """Return the OpenAI-compatible argument string for an IR tool call.

    `raw_args` takes precedence because it carries the exact string received from
    an OpenAI-compatible API when the value could not be represented as an IR
    object. Otherwise, serialize the structured dict compactly for the wire.
    """
    if raw_args is not None:
        return raw_args
    return json.dumps(args, ensure_ascii=False, separators=(",", ":"))
