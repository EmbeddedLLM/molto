import json
from base64 import b64encode
from collections.abc import AsyncIterator
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.utils.io import read_yaml

XLSX_IMAGE_URL = "data:image/jpeg;base64," + b64encode(
    (Path(__file__).parents[1] / "files" / "xlsx.jpg").read_bytes()
).decode("ascii")

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes"
RESPONSES_CASSETTE_DIR = CASSETTE_DIR / "openai" / "responses" / "test_interface"
CC_CASSETTE_DIR = CASSETTE_DIR / "openai" / "chat_completions" / "test_interface"
UPSTREAM_CASSETTE_DIRS = {
    "responses": RESPONSES_CASSETTE_DIR,
    "chat/completions": CC_CASSETTE_DIR,
}
Endpoint = Literal["responses", "chat/completions"]
ENDPOINTS: tuple[Endpoint, ...] = ("responses", "chat/completions")


@lru_cache(maxsize=5)
def _read_cassette(path: Path) -> dict[str, Any]:
    """Read a cassette YAML file."""
    return read_yaml(path)


def _get_http_client(
    cassette_names: list[str],
    endpoint: Endpoint = "responses",
) -> AsyncClient:
    cassette_names = cassette_names.copy()
    cassette_dir = UPSTREAM_CASSETTE_DIRS[endpoint]

    def cassette_handler(request: Request) -> Response:
        cassette = _read_cassette(cassette_dir / cassette_names.pop(0))
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
        cassette = _read_cassette(UPSTREAM_CASSETTE_DIRS[endpoint] / cassette_name)
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
                cassette = _read_cassette(UPSTREAM_CASSETTE_DIRS["responses"] / cassette_name)
                for event in "".join(cassette["response"]["sse"]).split("\n\n"):
                    if event:
                        yield _MockResponsesEvent(sse=f"{event}\n\n")

        return iter_events()

    return _mock_responses


async def test_extra_inputs(monkeypatch: pytest.MonkeyPatch):
    cassette_names = [
        "simple_text-stream-00.yaml",
    ]
    # TODO: Remove monkeypatching
    monkeypatch.setattr(Molto, "responses", _mock_responses_factory(cassette_names), raising=False)

    async with _get_http_client(cassette_names) as http_client:
        client = Molto(http_client=http_client)

        # --- Extra inputs --- #
        cassette = _read_cassette(RESPONSES_CASSETTE_DIR / cassette_names[0])
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
) -> None:
    num_turns = len(cassette_names)
    async with _get_http_client(cassette_names, endpoint) as http_client:
        client = Molto(http_client=http_client)
        # --- Output matches upstream format (Responses <-> Responses) --- #
        for _ in range(num_turns):
            cassette = _read_cassette(RESPONSES_CASSETTE_DIR / cassette_names.pop(0))
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
            for event, expected_sse in zip(events, expected_sse_events, strict=True):
                assert isinstance(event, BaseModel)
                assert event.as_sse() == expected_sse


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_simple_text_stream(endpoint: str, monkeypatch: pytest.MonkeyPatch):
    cassette_names = [
        "simple_text-stream-00.yaml",
        "simple_text-stream-01.yaml",
    ]
    # TODO: Remove monkeypatching
    monkeypatch.setattr(Molto, "responses", _mock_responses_factory(cassette_names), raising=False)
    await _assert_responses_events(cassette_names, endpoint)


async def test_reasoning_stream(monkeypatch: pytest.MonkeyPatch):
    cassette_names = [
        "reasoning-stream-00.yaml",
        "reasoning-stream-01.yaml",
    ]
    # TODO: Remove monkeypatching
    monkeypatch.setattr(Molto, "responses", _mock_responses_factory(cassette_names), raising=False)
    await _assert_responses_events(cassette_names)


async def test_instructions_stream(monkeypatch: pytest.MonkeyPatch):
    cassette_names = [
        "instructions-stream-00.yaml",
        "instructions-stream-01.yaml",
    ]
    # TODO: Remove monkeypatching
    monkeypatch.setattr(Molto, "responses", _mock_responses_factory(cassette_names), raising=False)
    await _assert_responses_events(cassette_names)
