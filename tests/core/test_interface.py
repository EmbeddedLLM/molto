from copy import deepcopy
from functools import lru_cache
from itertools import pairwise, product
from pathlib import Path
from typing import Any, Literal

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.types.openai_responses import RECONSTRUCTIBLE_RESPONSE_FIELDS
from molto.utils.io import dumps_sse, json_loads, load_sse, read_yaml

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes"
Protocol = Literal["responses", "chat/completions"]
PROTOCOLS: tuple[Protocol, ...] = ("responses", "chat/completions")
ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(product(PROTOCOLS, PROTOCOLS))


def _cassette_path(
    protocol: Protocol,
    cassette_name: str,
    *,
    matched: bool = False,
) -> Path:
    protocol_dir = "chat_completions" if protocol == "chat/completions" else protocol
    if not matched:
        return CASSETTE_DIR / "openai" / protocol_dir / "test_interface" / cassette_name

    scenario, separator, variant = cassette_name.partition("-")
    if not separator:
        raise ValueError(f"Matched cassette name must include a scenario: {cassette_name}")
    return CASSETTE_DIR / "matched" / scenario / f"openai_{protocol_dir}-{variant}"


@lru_cache(maxsize=5)
def _read_cassette(path: Path) -> dict[str, Any]:
    """Read a cassette YAML file."""
    return read_yaml(path)


def _get_http_client(
    cassette_names: list[str],
    upstream: Protocol = "responses",
    *,
    matched: bool = False,
) -> AsyncClient:
    cassette_names = cassette_names.copy()

    def cassette_handler(request: Request) -> Response:
        cassette = _read_cassette(_cassette_path(upstream, cassette_names.pop(0), matched=matched))
        request_body = json_loads(request.content)

        assert request.method == cassette["request"]["method"]
        assert request.url.path == cassette["request"]["path"]
        assert request_body == cassette["request"]["body"]

        return Response(
            cassette["response"]["status_code"],
            content="".join(cassette["response"]["sse"]).encode(),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )

    return AsyncClient(transport=MockTransport(cassette_handler))


def _assert_contains_expected(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected an object at {path}, got {type(actual)}"
        missing_keys = expected.keys() - actual.keys()
        assert not missing_keys, f"Missing keys at {path}: {sorted(missing_keys)}"
        for key, expected_value in expected.items():
            _assert_contains_expected(actual[key], expected_value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"Expected a list at {path}, got {type(actual)}"
        assert len(actual) == len(expected), (
            f"Expected {len(expected)} items at {path}, got {len(actual)}"
        )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_contains_expected(actual_item, expected_item, f"{path}[{index}]")
    else:
        assert actual == expected, f"Value mismatch at {path}: {actual!r} != {expected!r}"


def _assert_sse_events_contain(
    actual_sse_events: list[str], expected_sse_events: list[str]
) -> None:
    assert len(actual_sse_events) == len(expected_sse_events)
    for index, (actual_sse, expected_sse) in enumerate(
        zip(actual_sse_events, expected_sse_events, strict=True)
    ):
        _assert_contains_expected(
            load_sse(actual_sse),
            load_sse(expected_sse),
            path=f"event[{index}]",
        )


def test_event_comparison_allows_extra_actual_fields():
    expected = {
        "type": "response.completed",
        "response": {"output": [{"type": "message", "status": "completed"}]},
    }
    actual = {
        "type": "response.completed",
        "sequence_number": 10,
        "response": {
            "id": "response_123",
            "output": [
                {
                    "id": "message_123",
                    "type": "message",
                    "status": "completed",
                }
            ],
        },
    }

    _assert_contains_expected(actual, expected)
    with pytest.raises(AssertionError, match="Missing keys"):
        _assert_contains_expected(expected, actual)


def _sse_records(cassette: dict[str, Any]) -> list[str]:
    records = []
    for event in "".join(cassette["response"]["sse"]).split("\n\n"):
        event = event.strip()
        if event and event != "data: [DONE]":
            records.append(f"{event}\n\n")
    return records


def _chat_stream_semantics(sse_events: list[str]) -> dict[str, Any]:
    events = [load_sse(sse) for sse in sse_events]
    first = events[0]
    reasoning_deltas: list[str] = []
    text_deltas: list[str] = []
    finish_reasons: list[str] = []
    usage: dict[str, Any] | None = None

    for event in events:
        for choice in event["choices"]:
            delta = choice["delta"]
            if choice["finish_reason"] is None:
                if delta.get("reasoning_content") is not None:
                    reasoning_deltas.append(delta["reasoning_content"])
                if delta.get("content") is not None:
                    text_deltas.append(delta["content"])
            else:
                finish_reasons.append(choice["finish_reason"])

        raw_usage = event.get("usage")
        if raw_usage is not None:
            prompt_details = raw_usage.get("prompt_tokens_details") or {}
            completion_details = raw_usage.get("completion_tokens_details") or {}
            usage = {
                "prompt_tokens": raw_usage["prompt_tokens"],
                "completion_tokens": raw_usage["completion_tokens"],
                "total_tokens": raw_usage["total_tokens"],
                "cached_tokens": prompt_details.get("cached_tokens", 0),
                "cache_write_tokens": prompt_details.get("cache_write_tokens", 0),
                "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
            }

    return {
        "id": first["id"],
        "created": first["created"],
        "model": first["model"],
        "reasoning_deltas": reasoning_deltas,
        "text_deltas": text_deltas,
        "finish_reasons": finish_reasons,
        "usage": usage,
    }


def _assert_responses_stream_invariants(sse_events: list[str]) -> None:
    events = [load_sse(sse) for sse in sse_events]
    sequence_numbers = [event["sequence_number"] for event in events]
    assert all(current == previous + 1 for previous, current in pairwise(sequence_numbers)), (
        f"Non-contiguous sequence numbers: {sequence_numbers}"
    )

    output_item_ids = [
        event["item"]["id"]
        for event in events
        if event.get("type") == "response.output_item.added"
    ]
    assert len(output_item_ids) == len(set(output_item_ids)), (
        f"Duplicate output item IDs: {output_item_ids}"
    )


def _reconstructable_responses_event(sse: str, body: dict[str, Any]) -> dict[str, Any]:
    event = load_sse(sse)
    event.pop("item_id", None)

    item = event.get("item")
    if isinstance(item, dict):
        item.pop("id", None)

    response = event.get("response")
    if isinstance(response, dict):
        response = {
            k: v for k, v in response.items() if k in body or k in RECONSTRUCTIBLE_RESPONSE_FIELDS
        }
        usage = response.get("usage")
        if usage is not None:
            assert isinstance(usage, dict)
            input_details = usage.get("input_tokens_details")
            if isinstance(input_details, dict):
                input_details.setdefault("cache_write_tokens", 0)
        for output_item in response.get("output") or []:
            if isinstance(output_item, dict):
                output_item.pop("id", None)
        event["response"] = response

    return event


def test_chat_completions_event_comparison_preserves_reconstructable_fields():
    expected_events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_expected"},
        },
        {
            "type": "response.completed",
            "response": {
                "background": False,
                "completed_at": 123,
                "model": "expected-model",
                "output": [{"id": "msg_expected"}],
            },
        },
    ]
    actual_events = deepcopy(expected_events)
    actual_events[0]["item"]["id"] = "msg_generated"
    actual_events[1]["response"].pop("background")
    actual_events[1]["response"]["completed_at"] = 456
    actual_events[1]["response"]["output"][0]["id"] = "msg_generated"

    def reconstructable(
        events: list[dict[str, Any]], body: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [_reconstructable_responses_event(dumps_sse(e), body) for e in events]

    assert reconstructable(actual_events, {}) == reconstructable(expected_events, {})
    assert reconstructable(actual_events, {"background": False}) != reconstructable(
        expected_events, {"background": False}
    )

    actual_events[1]["response"]["model"] = "different-model"
    assert reconstructable(actual_events, {}) != reconstructable(expected_events, {})


async def test_extra_inputs():
    cassette_names = [
        "simple_text-stream-00.yaml",
    ]

    async with _get_http_client(cassette_names, matched=True) as http_client:
        client = Molto(http_client=http_client)

        # --- Extra inputs --- #
        cassette = _read_cassette(_cassette_path("responses", cassette_names[0], matched=True))
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            await client.responses(
                body=dict(
                    messages=[
                        {"role": "user", "content": "Write one short sentence about the ocean."}
                    ],
                    **cassette["request"]["body"],
                )
            )


async def _assert_stream_events(
    cassette_names: list[str],
    downstream: Protocol,
    upstream: Protocol,
    *,
    matched: bool = False,
) -> None:
    num_turns = len(cassette_names)
    async with _get_http_client(cassette_names, upstream, matched=matched) as http_client:
        client = Molto(http_client=http_client)
        for _ in range(num_turns):
            cassette = _read_cassette(
                _cassette_path(downstream, cassette_names.pop(0), matched=matched)
            )
            coordinator = (
                client.responses if downstream == "responses" else client.chat_completions
            )
            response = await coordinator(
                body=deepcopy(cassette["request"]["body"]),
                endpoint=upstream,
            )
            events = [event async for event in response]
            expected_sse_events = _sse_records(cassette)
            actual_sse_events = []
            for event in events:
                assert isinstance(event, BaseModel)
                actual_sse_events.append(dumps_sse(event))
            if downstream == "responses":
                assert len(events) == len(expected_sse_events)
                _assert_responses_stream_invariants(actual_sse_events)
                if upstream == "responses":
                    _assert_sse_events_contain(actual_sse_events, expected_sse_events)
                elif upstream == "chat/completions":
                    # Chat does not expose Responses output-item IDs or the exact terminal
                    # timestamp. Every other translated field must still match exactly.
                    actual_events = [
                        _reconstructable_responses_event(
                            sse, body=deepcopy(cassette["request"]["body"])
                        )
                        for sse in actual_sse_events
                    ]
                    expected_events = [
                        _reconstructable_responses_event(
                            sse, body=deepcopy(cassette["request"]["body"])
                        )
                        for sse in expected_sse_events
                    ]
                    assert actual_events == expected_events
                else:
                    raise ValueError(f"Unsupported upstream protocol: {upstream}")
            elif downstream == "chat/completions":
                if upstream == "chat/completions":
                    assert len(events) == len(expected_sse_events)
                    _assert_sse_events_contain(actual_sse_events, expected_sse_events)
                elif upstream == "responses":
                    assert _chat_stream_semantics(actual_sse_events) == _chat_stream_semantics(
                        expected_sse_events
                    )
                else:
                    raise ValueError(f"Unsupported upstream protocol: {upstream}")
            else:
                raise ValueError(f"Unsupported downstream protocol: {downstream}")


async def _assert_responses_events(
    cassette_names: list[str],
    upstream: Protocol = "responses",
    *,
    matched: bool = False,
) -> None:
    await _assert_stream_events(
        cassette_names,
        downstream="responses",
        upstream=upstream,
        matched=matched,
    )


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_simple_text_stream(downstream: Protocol, upstream: Protocol):
    cassette_names = [
        "simple_text-stream-00.yaml",
        "simple_text-stream-01.yaml",
    ]
    await _assert_stream_events(cassette_names, downstream, upstream, matched=True)


@pytest.mark.parametrize(("downstream", "upstream"), ROUTES)
async def test_reasoning_stream_raw_content(downstream: Protocol, upstream: Protocol):
    cassette_names = [
        "reasoning_raw-stream-00.yaml",
        "reasoning_raw-stream-01.yaml",
    ]
    await _assert_stream_events(cassette_names, downstream, upstream, matched=True)


async def test_reasoning_stream_encrypted_content():
    cassette_names = [
        "reasoning-stream-00.yaml",
        "reasoning-stream-01.yaml",
    ]
    await _assert_responses_events(cassette_names)


async def test_instructions_stream():
    cassette_names = [
        "instructions-stream-00.yaml",
        "instructions-stream-01.yaml",
    ]
    await _assert_responses_events(cassette_names)
