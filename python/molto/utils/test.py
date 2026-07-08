from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.utils.io import json_loads, read_yaml

STREAM_PARAMS = dict(argvalues=[True, False], ids=["stream", "non-stream"])

CASSETTE_DIR = Path("tests/cassettes")


def read_cassette(path: Path, stream: bool) -> dict[str, Any] | list[str]:
    """Read a YAML cassette file and return the response payload.

    For stream cassettes the file stem is suffixed with "-stream" (e.g.
    "foo.yaml" → "foo-stream.yaml") and the "sse" list of SSE lines is
    returned.  For non-stream cassettes the "body" dict is returned.
    """
    if stream:
        path = path.with_stem(f"{path.stem}-stream")
    data = read_yaml(path)["response"]
    return data["sse"] if stream else data["body"]


def read_cassette_request(path: Path, stream: bool = False) -> dict[str, Any]:
    """Read the "request.body" section from a cassette YAML file."""
    if stream:
        path = path.with_stem(f"{path.stem}-stream")
    return read_yaml(path)["request"]["body"]


def extract_cassette_response(
    path: Path,
    stream: bool,
    terminal_event_type: str = "response.completed",
) -> dict[str, Any]:
    """Extract the full response dict from a cassette.

    For non-stream cassettes this is simply the response body.  For stream
    cassettes the SSE lines are scanned for the *single* frame whose "type"
    matches "terminal_event_type" (default "response.completed") and its
    nested "response" object is returned.
    """
    data = read_cassette(path, stream=stream)
    if stream:
        frames = [
            line
            for line in data
            if line.startswith("data: ") and f'"type":"{terminal_event_type}"' in line
        ]
        assert len(frames) == 1, (
            f"Expected exactly one {terminal_event_type} frame, found {len(frames)}"
        )
        return json_loads(frames[0][len("data: ") :])["response"]
    return data


def extract_output_ids(cassette_resp: dict[str, Any], *types: str) -> dict[str, str]:
    """Return the first "id" for each requested output "type" in a cassette response.

    Given a cassette response dict (containing an "output" list) and one or
    more output item types (e.g. "message", "function_call"), returns a
    mapping of {type: id} using the first occurrence of each type.

    Example:

        ids = extract_output_ids(resp, "message", "function_call")
        # {"message": "msg_abc", "function_call": "fc_xyz"}
    """
    result: dict[str, str] = {}
    for output_item in cassette_resp["output"]:
        item_type = output_item["type"]
        if item_type in types and item_type not in result:
            result[item_type] = output_item["id"]
    return result


def assert_usage(
    usage: Any, *, output_tokens_exact: int | None = None, expect_reasoning: bool = False
) -> None:
    """Assert that a usage object has positive token counts and consistent totals.

    Args:
        usage: Object with "input_tokens", "output_tokens", and
            "total_tokens" attributes (e.g. an OpenAI CompletionUsage).
        output_tokens_exact: If provided, asserts the output token count
            equals this value exactly.
        expect_reasoning: If True, also asserts that
            "output_tokens_details.reasoning_tokens" is present and positive.
    """
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    if output_tokens_exact is not None:
        assert usage.output_tokens == output_tokens_exact
    if expect_reasoning:
        assert usage.output_tokens_details is not None
        assert usage.output_tokens_details.reasoning_tokens > 0


def assert_stream_text_events(events: list[Any]) -> None:
    """Assert the structure of streamed text events.

    For each unique assistant message ID, validates that there is exactly one
    "output_item.added" and one "output_item.done", at least one
    "output_text.delta", and at least one "output_text.done" referencing that
    item ID.

    Args:
        events: Collected stream event objects from a Molto.chat() call.
    """
    from molto.types.openai.responses import (
        OpenAIResponsesStreamOutput,
        OpenAIResponsesStreamText,
    )

    msg_item_added = [
        e
        for e in events
        if isinstance(e, OpenAIResponsesStreamOutput)
        and e.type == "response.output_item.added"
        and hasattr(e.item, "role")
        and e.item.role == "assistant"
    ]
    msg_item_done = [
        e
        for e in events
        if isinstance(e, OpenAIResponsesStreamOutput)
        and e.type == "response.output_item.done"
        and hasattr(e.item, "role")
        and e.item.role == "assistant"
    ]
    assert len(msg_item_added) >= 1
    assert len(msg_item_done) >= 1
    added_ids = [e.item.id for e in msg_item_added]
    done_ids = [e.item.id for e in msg_item_done]
    for aid in added_ids:
        assert aid is not None
        assert aid.startswith("msg_")
    assert added_ids == done_ids

    for msg_id in added_ids:
        deltas = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamText)
            and e.type == "response.output_text.delta"
            and e.item_id == msg_id
        ]
        assert len(deltas) >= 1

        done = [
            e
            for e in events
            if isinstance(e, OpenAIResponsesStreamText)
            and e.type == "response.output_text.done"
            and e.item_id == msg_id
        ]
        assert len(done) == 1, f"expected 1 text_done for item {msg_id!r}, got {len(done)}"

    assert_events_before_terminal(events, "response.output_text.delta")


def assert_events_before_terminal(events: list[Any], event_type: str) -> None:
    """Assert all events of "event_type" appear before the terminal stream event."""
    indices = [i for i, event in enumerate(events) if event.type == event_type]
    assert all(i < len(events) - 1 for i in indices)


# TODO: Add a code interpreter stream parity test once a cassette is available.
_SDK_TYPE_MAP: dict[str, str] = {
    "response.reasoning_summary_text.delta": "response.reasoning.delta",
    "response.reasoning_summary_text.done": "response.reasoning.done",
}

# field -> event types that carry it (used for 1:1 payload comparison)
_FIELD_CHECKS: dict[str, frozenset[str]] = {
    "item_id": frozenset(
        {
            "response.output_text.delta",
            "response.output_text.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.reasoning.delta",
            "response.reasoning.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call.completed",
            "response.code_interpreter_call_code.delta",
            "response.code_interpreter_call_code.done",
        }
    ),
    "delta": frozenset(
        {
            "response.output_text.delta",
            "response.function_call_arguments.delta",
            "response.reasoning.delta",
        }
    ),
    "text": frozenset({"response.output_text.done", "response.reasoning.done"}),
    "arguments": frozenset({"response.function_call_arguments.done"}),
}


def _normalize_sdk_event_type(event_type: str) -> str:
    """Return the Molto/OpenResponses event type matching an SDK event type."""
    return _SDK_TYPE_MAP.get(event_type, event_type)


def _normalized_event_types(events: list[Any]) -> list[str]:
    """Return event types after applying SDK-to-Molto naming compatibility."""
    return [_normalize_sdk_event_type(e.type) for e in events]


def _assert_event_type_sequence_matches(sdk_events: list[Any], molto_events: list[Any]) -> None:
    sdk_types = _normalized_event_types(sdk_events)
    molto_types = [e.type for e in molto_events]
    assert sdk_types == molto_types, (
        f"Event type sequence mismatch\nSDK:   {sdk_types}\nMolto: {molto_types}"
    )


def _assert_matching_scalar_fields(event_type: str, sdk_event: Any, molto_event: Any) -> None:
    for field, event_types in _FIELD_CHECKS.items():
        if event_type in event_types:
            assert getattr(sdk_event, field, None) == getattr(molto_event, field, None), (
                f"{field} mismatch at {event_type}"
            )


def _assert_matching_output_item(event_type: str, sdk_event: Any, molto_event: Any) -> None:
    if event_type not in {"response.output_item.added", "response.output_item.done"}:
        return

    sdk_item = getattr(sdk_event, "item", None)
    molto_item = getattr(molto_event, "item", None)
    if sdk_item is None or molto_item is None:
        return

    assert sdk_item.id == molto_item.id, f"item.id mismatch at {event_type}"
    assert sdk_item.type == molto_item.type, f"item.type mismatch at {event_type}"


def _assert_matching_terminal_response_id(
    event_type: str, sdk_event: Any, molto_event: Any
) -> None:
    if event_type not in {"response.completed", "response.incomplete"}:
        return

    # Only compare response.id for terminal events: Molto uses a provisional UUID
    # for response.created/in_progress that will not match the cassette ID.
    sdk_response_id = getattr(getattr(sdk_event, "response", None), "id", None)
    if sdk_response_id is not None:
        molto_response_id = getattr(getattr(molto_event, "response", None), "id", None)
        assert sdk_response_id == molto_response_id, f"response.id mismatch at {event_type}"


def assert_output_item_order(events: list[Any]) -> None:
    """Assert every output_item.added event precedes its corresponding output_item.done."""
    added_indices = [i for i, e in enumerate(events) if e.type == "response.output_item.added"]
    done_indices = [i for i, e in enumerate(events) if e.type == "response.output_item.done"]
    assert len(added_indices) == len(done_indices)
    assert all(added_indices[i] < done_indices[i] for i in range(len(added_indices)))


def assert_stream_events_match(sdk_events: list[Any], molto_events: list[Any]) -> None:
    """Assert SDK and Molto stream events have the same order and key payloads."""
    _assert_event_type_sequence_matches(sdk_events, molto_events)

    for sdk_event, molto_event in zip(sdk_events, molto_events, strict=True):
        event_type = _normalize_sdk_event_type(sdk_event.type)
        _assert_matching_scalar_fields(event_type, sdk_event, molto_event)
        _assert_matching_output_item(event_type, sdk_event, molto_event)
        _assert_matching_terminal_response_id(event_type, sdk_event, molto_event)


def make_handler(
    cassette_path: Path,
    req_body: dict[str, Any] | None = None,
    assert_attrs: list[str] | None = None,
) -> partial[httpx.Response]:
    """Create an httpx.MockTransport handler that replays a cassette.

    The returned handler validates the incoming request method (POST), checks
    that "stream" matches the cassette variant being loaded, and optionally
    asserts specific request body attributes.  Error cassettes (status >= 400)
    are returned directly without request validation.

    Args:
        cassette_path: Path to the YAML cassette file.
        req_body: Expected request body values for attribute assertions.
        assert_attrs: Keys in *req_body* to assert against the incoming
            request body.
    """
    cassette = read_yaml(cassette_path)
    resp = cassette.get("response")
    is_error = resp is not None and isinstance(resp, dict) and resp.get("status_code", 200) >= 400

    if is_error:
        status_code: int = resp["status_code"]
        error_body: Any = resp["body"]
        raw_headers = resp.get("headers") or {}
        error_headers = {k: v for k, v in raw_headers.items() if k.lower() != "content-length"}

    def handler(request: httpx.Request, load_stream_cassette: bool) -> httpx.Response:
        if is_error:
            if isinstance(error_body, str):
                return httpx.Response(
                    status_code, content=error_body.encode(), headers=error_headers
                )
            return httpx.Response(status_code, json=error_body, headers=error_headers)

        try:
            assert request.method == "POST"
            body = json_loads(request.content)
            if req_body and assert_attrs:
                for attr in assert_attrs:
                    assert body.get(attr) == req_body[attr], (
                        f"Expected {attr} to be {req_body[attr]!r}, got {body.get(attr)!r}"
                    )
            assert body.get("stream", False) is load_stream_cassette
            data = read_cassette(cassette_path, stream=load_stream_cassette)
            if load_stream_cassette:
                return httpx.Response(
                    200,
                    content="".join(data).encode(),
                    headers={"content-type": "text/event-stream; charset=utf-8"},
                )
            return httpx.Response(200, json=data)
        except AssertionError as e:
            return httpx.Response(
                400,
                json={"error": f"Assertions failed: {e!r}"},
            )
        except Exception as e:
            return httpx.Response(
                400,
                json={"error": f"Invalid request: {e!r}"},
            )

    return handler


def make_openai_client(
    handler: Any,
    load_stream_cassette: bool,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client backed by a mock transport.

    Args:
        handler: An httpx handler (as returned by make_handler()).
        load_stream_cassette: Whether the handler should load the stream
            variant of the cassette.
    """
    return AsyncOpenAI(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                partial(handler, load_stream_cassette=load_stream_cassette)
            ),
            base_url="http://testserver",
        ),
        api_key=ENV_CONFIG.openai_api_key_plain,
        max_retries=1,
    )


def make_molto_client(
    handler: Any = None,
    *,
    cassette_path: Path | None = None,
) -> Molto:
    """Create a Molto instance backed by a mock transport.

    Either pass *handler* directly or provide *cassette_path* to have one
    created via make_handler(). The client is always configured for streaming
    (load_stream_cassette=True).
    """
    if cassette_path is not None:
        handler = make_handler(cassette_path)
    return Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(partial(handler, load_stream_cassette=True)),
            base_url="http://testserver",
        ),
    )


def prepend_model_prefix(body: dict[str, Any], prefix: str = "openai/") -> dict[str, Any]:
    """Return a shallow copy of "body" with the model field prefixed.

    Example::

        prepend_model_prefix({"model": "gpt-4o", ...})
        # {"model": "openai/gpt-4o", ...}
    """
    copy_body = deepcopy(body)
    copy_body["model"] = prefix + copy_body["model"]
    return copy_body
