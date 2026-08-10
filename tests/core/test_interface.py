import json
from base64 import b64encode
from collections.abc import AsyncIterator
from copy import deepcopy
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.types.openai_responses import RECONSTRUCTIBLE_RESPONSE_FIELDS
from molto.utils.io import read_yaml

XLSX_IMAGE_URL = "data:image/jpeg;base64," + b64encode(
    (Path(__file__).parents[1] / "files" / "xlsx.jpg").read_bytes()
).decode("ascii")

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes"
Endpoint = Literal["responses", "chat/completions"]
ENDPOINTS: tuple[Endpoint, ...] = ("responses", "chat/completions")


def _cassette_path(
    endpoint: Endpoint,
    cassette_name: str,
    *,
    matched: bool = False,
) -> Path:
    endpoint_dir = "chat_completions" if endpoint == "chat/completions" else endpoint
    if not matched:
        return CASSETTE_DIR / "openai" / endpoint_dir / "test_interface" / cassette_name

    scenario, separator, variant = cassette_name.partition("-")
    if not separator:
        raise ValueError(f"Matched cassette name must include a scenario: {cassette_name}")
    return CASSETTE_DIR / "matched" / scenario / f"openai_{endpoint_dir}-{variant}"


@lru_cache(maxsize=5)
def _read_cassette(path: Path) -> dict[str, Any]:
    """Read a cassette YAML file."""
    return read_yaml(path)


def _get_http_client(
    cassette_names: list[str],
    endpoint: Endpoint = "responses",
    *,
    matched: bool = False,
) -> AsyncClient:
    cassette_names = cassette_names.copy()

    def cassette_handler(request: Request) -> Response:
        cassette = _read_cassette(_cassette_path(endpoint, cassette_names.pop(0), matched=matched))
        request_body = json.loads(request.content)

        assert request.method == cassette["request"]["method"]
        assert request.url.path == cassette["request"]["path"]
        assert request_body == cassette["request"]["body"]

        return Response(
            cassette["response"]["status_code"],
            content="".join(cassette["response"]["sse"]).encode(),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )

    return AsyncClient(transport=MockTransport(cassette_handler))


class _MockResponsesEvent(BaseModel, extra="forbid"):
    sse: str

    def as_sse(self) -> str:
        return self.sse


def _mock_responses_factory(
    cassette_names: list[str],
    *,
    matched: bool = False,
):
    cassette_names = cassette_names.copy()

    async def _mock_responses(
        self: Molto,
        *,
        body: dict[str, Any],
        endpoint: Endpoint = "responses",
        provider: Literal["openai", "vllm"] = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[BaseModel]:
        # Mock validation
        if "messages" in body:
            raise ValidationError.from_exception_data(
                "ResponsesRequest",
                [{"type": "extra_forbidden", "loc": ("messages",), "input": body["messages"]}],
            )
        assert self._http_client is not None

        cassette_name = cassette_names.pop(0)
        cassette = _read_cassette(_cassette_path(endpoint, cassette_name, matched=matched))
        response = await self._http_client.post(
            f"https://api.openai.com/v1/{endpoint}",
            json=body if endpoint == "responses" else cassette["request"]["body"],
        )
        response.raise_for_status()

        async def iter_events() -> AsyncIterator[BaseModel]:
            if endpoint == "responses":
                for event in response.text.split("\n\n"):
                    if event:
                        yield _MockResponsesEvent(sse=f"{event}\n\n")
            else:
                cassette = _read_cassette(
                    _cassette_path("responses", cassette_name, matched=matched)
                )
                for event in "".join(cassette["response"]["sse"]).split("\n\n"):
                    if event:
                        yield _MockResponsesEvent(sse=f"{event}\n\n")

        return iter_events()

    return _mock_responses


def _load_sse(sse: str) -> dict[str, Any]:
    data = "\n".join(
        line.removeprefix("data:").lstrip()
        for line in sse.splitlines()
        if line.startswith("data:")
    )
    event = json.loads(data)
    assert isinstance(event, dict)
    return event


def _assert_responses_stream_invariants(sse_events: list[str]) -> None:
    events = [_load_sse(sse) for sse in sse_events]
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
    event = _load_sse(sse)
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
        if usage is None:
            response.pop("usage", None)
        else:
            assert isinstance(usage, dict)
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

    def as_sse(events: list[dict[str, Any]]) -> list[str]:
        return [f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events]

    def reconstructable(
        events: list[dict[str, Any]], body: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [_reconstructable_responses_event(sse, body) for sse in as_sse(events)]

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


async def _assert_responses_events(
    cassette_names: list[str],
    endpoint: Endpoint = "responses",
    *,
    matched: bool = False,
) -> None:
    num_turns = len(cassette_names)
    async with _get_http_client(cassette_names, endpoint, matched=matched) as http_client:
        client = Molto(http_client=http_client)
        # --- Output matches upstream format (Responses <-> Responses) --- #
        for _ in range(num_turns):
            cassette = _read_cassette(
                _cassette_path("responses", cassette_names.pop(0), matched=matched)
            )
            response = await client.responses(
                body=deepcopy(cassette["request"]["body"]),
                endpoint=endpoint,
            )
            events = [event async for event in response]
            expected_sse_events = [
                f"{event}\n\n"
                for event in "".join(cassette["response"]["sse"]).split("\n\n")
                if event
            ]
            assert len(events) == len(expected_sse_events)
            actual_sse_events = []
            for event in events:
                assert isinstance(event, BaseModel)
                actual_sse_events.append(event.as_sse())
            _assert_responses_stream_invariants(actual_sse_events)

            if endpoint == "responses":
                assert _load_sse(actual_sse_events) == _load_sse(expected_sse_events)
            else:
                # Chat does not expose Responses output-item IDs or the exact terminal
                # timestamp. Every other translated field must still match exactly.
                actual_events = [
                    _reconstructable_responses_event(
                        sse, body=deepcopy(cassette["request"]["body"])
                    )
                    for sse in actual_sse_events
                ]
                for e in actual_events:
                    print(e)
                expected_events = [
                    _reconstructable_responses_event(
                        sse, body=deepcopy(cassette["request"]["body"])
                    )
                    for sse in expected_sse_events
                ]
                assert actual_events == expected_events


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_simple_text_stream(endpoint: str):
    cassette_names = [
        "simple_text-stream-00.yaml",
        "simple_text-stream-01.yaml",
    ]
    await _assert_responses_events(cassette_names, endpoint, matched=True)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_reasoning_stream_raw_content(endpoint: str):
    cassette_names = [
        "reasoning_raw-stream-00.yaml",
        "reasoning_raw-stream-01.yaml",
    ]
    await _assert_responses_events(cassette_names, endpoint, matched=True)


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
