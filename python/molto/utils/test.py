from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from anthropic.lib.streaming._messages import accumulate_event
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import BaseModel

from molto.providers import Endpoint
from molto.types.anthropic_messages import AnthropicMessagesStreamEventAdapter
from molto.types.openai_chat_completions import (
    OpenAIChatCompletionsCreateRequest,
    OpenAIChatCompletionsStreamChunk,
)
from molto.types.openai_responses import (
    OpenAIResponsesCreateRequest,
    OpenAIResponsesStreamEventAdapter,
)
from molto.utils.io import dumps_sse, load_sse

_RECONSTRUCTIBLE_RESPONSE_FIELDS = {
    "id",
    "incomplete_details",
    "model",
    "object",
    "output",
    "service_tier",
    "status",
    "usage",
}

_RECONSTRUCTIBLE_CHAT_COMPLETIONS_FIELDS = {
    "choices",
    "created",
    "id",
    "model",
    "object",
    "service_tier",
    "usage",
}

_ANTHROPIC_EVENT_ADAPTER = AnthropicMessagesStreamEventAdapter

# Request and response headers retained in cassette recordings.
RECORDED_HEADERS = {
    "content-type",
    "content-length",
    "authorization",
    "user-agent",
    "accept",
    "accept-encoding",
    "cache-control",
    "connection",
    # Helpful for grouping multi-request tool loops into a single "run"
    "x-run-id",
}


def mask_authorization(value: str) -> str:
    """
    Mask Authorization header values in recordings.

    Cassettes are intended to be shareable "golden" artifacts; never write any
    API key material to disk (even partially).
    """
    if not value:
        return value
    lower = value.lower()
    if lower.startswith("bearer "):
        return "Bearer ***"
    return "***"


def filter_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Filter headers to only include whitelisted ones."""
    return {
        k: v if k.lower() != "authorization" else mask_authorization(v)
        for k, v in headers.items()
        if k.lower() in RECORDED_HEADERS
    }


def collect_sdk_response(protocol: Endpoint, events: list[BaseModel]) -> BaseModel:
    """Collect parsed SDK events without invoking Molto's adapters or collectors."""
    if protocol == "responses":
        assert events[-1].type in {"response.completed", "response.incomplete"}
        return events[-1].response
    if protocol == "chat/completions":
        state = ChatCompletionStreamState()
        for event in events:
            state.handle_chunk(event.model_copy(deep=True))
        return state.current_completion_snapshot
    snapshot = None
    for event in events:
        snapshot = accumulate_event(event=event.model_copy(deep=True), current_snapshot=snapshot)
        # The SDK accumulator does not copy this compatible-provider extension,
        # although its parsed message_delta preserves the observed value.
        if (
            event.type == "message_delta"
            and getattr(event.usage, "output_tokens_details", None) is not None
        ):
            snapshot.usage = snapshot.usage.model_copy(
                update={"output_tokens_details": event.usage.output_tokens_details}
            )
    assert snapshot is not None
    return snapshot


def assert_stream_lifecycle(
    protocol: Endpoint, events: list[BaseModel], *, native: bool = False
) -> None:
    """Assert balanced protocol lifecycles, retaining native Responses snapshot differences."""
    if protocol == "responses":
        assert_responses_stream_invariants([dumps_sse(event) for event in events], native=native)
        assert events[0].type == "response.created"
        assert events[-1].type in {"response.completed", "response.incomplete"}
        added = [
            event.output_index for event in events if event.type == "response.output_item.added"
        ]
        done = [
            event.output_index for event in events if event.type == "response.output_item.done"
        ]
        assert added == done == list(range(len(added)))
    elif protocol == "messages":
        assert events[0].type == "message_start"
        assert events[-1].type == "message_stop"
        opened = set()
        starts = []
        for event in events:
            if event.type == "content_block_start":
                assert event.index == len(starts)
                starts.append(event.index)
                opened.add(event.index)
            elif event.type == "content_block_delta":
                assert event.index in opened
            elif event.type == "content_block_stop":
                assert event.index in opened
                opened.remove(event.index)
        assert not opened
        assert sum(event.type == "message_delta" for event in events) == 1
    else:
        assert (
            sum(choice.finish_reason is not None for event in events for choice in event.choices)
            == 1
        )


def assert_responses_stream_invariants(sse_events: list[str], *, native: bool = False) -> None:
    """Assert Responses sequence numbers, item identities, and completed output snapshots."""
    events = [load_sse(sse) for sse in sse_events]
    sequence_numbers = [event["sequence_number"] for event in events]
    assert all(current == previous + 1 for previous, current in pairwise(sequence_numbers)), (
        f"Non-contiguous sequence numbers: {sequence_numbers}"
    )

    started: dict[int, dict[str, Any]] = {}
    completed: dict[int, dict[str, Any]] = {}
    item_ids: set[str] = set()
    for event in events:
        event_type = event["type"]
        if "output_index" in event or "item_id" in event:
            index = event["output_index"]
            if event_type == "response.output_item.added":
                assert index not in started, f"Output item {index} started twice"
                item = event["item"]
                assert item["id"] and item["id"] not in item_ids, "Duplicate or empty item ID"
                item_ids.add(item["id"])
                started[index] = item
            else:
                assert index in started, f"Update before output item {index} started"
                assert index not in completed, f"Update after output item {index} completed"
                if event_type == "response.output_item.done":
                    item = event["item"]
                    assert item["id"] == started[index]["id"], "Completed item ID changed"
                    assert item["type"] == started[index]["type"], "Completed item type changed"
                    if item["type"] == "function_call":
                        assert item["call_id"] == started[index]["call_id"]
                    completed[index] = item
                else:
                    assert event["item_id"] == started[index]["id"], "Update targets another item"
        elif event_type in {"response.completed", "response.incomplete"}:
            assert event is events[-1], "Events after terminal response"
            output = event["response"]["output"]
            assert started.keys() == completed.keys() == set(range(len(output))), (
                "Unbalanced output item starts, completions, or terminal output"
            )
            for index, item in enumerate(output):
                done = completed[index]
                if (
                    native
                    and item["type"] == "reasoning"
                    and done.get("encrypted_content")
                    and item.get("encrypted_content")
                ):
                    # Native Responses may expose a third opaque snapshot. A
                    # translated terminal item must reuse its done snapshot.
                    done = {**done, "encrypted_content": item["encrypted_content"]}
                if (
                    native
                    and item["type"] == "web_search_call"
                    and item["action"]["type"] == "search"
                    and "sources" in item["action"]
                ):
                    # Native search may report more sources in its terminal
                    # snapshot. Translated output retains the done snapshot.
                    done = {
                        **done,
                        "action": {**done["action"], "sources": item["action"]["sources"]},
                    }
                assert item == done, f"Terminal output differs from completed item {index}"
    assert events[-1]["type"] in {"response.completed", "response.incomplete"}, (
        "Missing terminal response"
    )


def assert_contains_expected(actual: Any, expected: Any, path: str = "$") -> None:
    """Assert that actual recursively contains the expected value."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected an object at {path}, got {type(actual)}"
        missing_keys = expected.keys() - actual.keys()
        assert not missing_keys, f"Missing keys at {path}: {sorted(missing_keys)}"
        for key, expected_value in expected.items():
            assert_contains_expected(actual[key], expected_value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"Expected a list at {path}, got {type(actual)}"
        assert len(actual) == len(expected), (
            f"Expected {len(expected)} items at {path}, got {len(actual)}"
        )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            assert_contains_expected(actual_item, expected_item, f"{path}[{index}]")
    else:
        assert actual == expected, f"Value mismatch at {path}: {actual!r} != {expected!r}"


def reconstructable_chat_completions_chunks(
    sse: str,
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate and normalize one Chat Completions SSE chunk for comparison."""
    OpenAIChatCompletionsCreateRequest.model_validate(body)
    chunk = OpenAIChatCompletionsStreamChunk.model_validate(load_sse(sse)).model_dump(
        mode="json",
        exclude_unset=True,
    )
    chunk = {
        key: value
        for key, value in chunk.items()
        if key in _RECONSTRUCTIBLE_CHAT_COMPLETIONS_FIELDS
    }

    choices = chunk["choices"]
    for choice in choices:
        # TODO: Remove this as soon as we support logprobs
        if choice.get("logprobs") in (None, ""):
            choice.pop("logprobs", None)
        if choice["finish_reason"] is not None:
            delta = choice["delta"]
            for field_name in ("content", "reasoning_content"):
                if delta.get(field_name) in (None, ""):
                    delta.pop(field_name, None)

    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return [chunk]

    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    chunk["usage"] = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "prompt_tokens_details": {
            "cached_tokens": prompt_details.get("cached_tokens") or 0,
            "cache_write_tokens": prompt_details.get("cache_write_tokens") or 0,
        },
        "completion_tokens_details": {
            "reasoning_tokens": completion_details.get("reasoning_tokens") or 0,
        },
    }
    if not choices:
        return [chunk]

    # Providers may combine terminal usage with the finish chunk or send it next.
    usage_chunk = {**chunk, "choices": []}
    chunk["usage"] = None
    return [chunk, usage_chunk]


def reconstructable_responses_event(
    sse: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Validate a Responses SSE event and retain fields reconstructible from Chat."""
    request = OpenAIResponsesCreateRequest.model_validate(body)
    event = OpenAIResponsesStreamEventAdapter.validate_python(load_sse(sse)).model_dump(
        mode="json",
        exclude_unset=True,
    )
    event.pop("item_id", None)

    item = event.get("item")
    if isinstance(item, dict):
        # Chat results expose neither Responses item IDs nor assistant phases.
        item.pop("id", None)
        item.pop("phase", None)

    response = event.get("response")
    if isinstance(response, dict):
        response = {
            key: value
            for key, value in response.items()
            if key in request.model_fields_set or key in _RECONSTRUCTIBLE_RESPONSE_FIELDS
        }
        # SDK responses may omit this optional field when there is no incomplete reason.
        response.setdefault("incomplete_details", None)
        reasoning = response.get("reasoning")
        if isinstance(reasoning, dict):
            # Chat exposes reasoning effort, but not Responses context, mode, or summary.
            response["reasoning"] = {
                key: value for key, value in reasoning.items() if key == "effort"
            }
        if (
            event["type"] in {"response.created", "response.in_progress"}
            and response.get("service_tier") == "auto"
        ):
            # Native Responses may report an unresolved tier before completion;
            # translated responses use the documented target default throughout.
            response["service_tier"] = "default"
        for output_item in response.get("output") or []:
            output_item.pop("id", None)
            output_item.pop("phase", None)
        event["response"] = response

    return event


def reconstructable_anthropic_event(sse: str) -> dict[str, Any]:
    """Validate an Anthropic SSE event and retain fields available across protocols."""
    event = _ANTHROPIC_EVENT_ADAPTER.validate_python(load_sse(sse)).model_dump(
        mode="json",
        exclude_unset=True,
    )
    if event["type"] == "message_start":
        message = event["message"]
        message.pop("stop_details", None)
        # Other protocols report usage only when the stream ends. Matching the
        # Messages start counts would require buffering its otherwise-live stream.
        usage = message.pop("usage", None)
        if isinstance(usage, dict) and "service_tier" in usage:
            message["usage"] = {"service_tier": usage["service_tier"]}
    elif event["type"] == "message_delta":
        event["delta"].pop("stop_details", None)
    return event
