import json
from base64 import b64encode
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from pydantic import BaseModel, ValidationError

from molto import Molto
from molto.utils.io import read_yaml

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes" / "openai" / "responses" / "test_interface"
XLSX_IMAGE_URL = "data:image/jpeg;base64," + b64encode(
    (Path(__file__).parents[1] / "files" / "xlsx.jpg").read_bytes()
).decode("ascii")


def _read_cassette(path: Path) -> dict[str, Any]:
    """Read a cassette YAML file."""
    return read_yaml(path)


def _get_http_client(cassette_names: list[str]) -> AsyncClient:
    assert all((CASSETTE_DIR / name).is_file() for name in cassette_names), (
        f"Missing cassettes in {CASSETTE_DIR!r}."
    )

    def cassette_handler(request: Request) -> Response:
        name = cassette_names.pop(0)
        cassette = _read_cassette(CASSETTE_DIR / name)
        assert request.method == cassette["request"]["method"]
        request_body = json.loads(request.content) if request.content else {}
        # Check model
        request_model: str = request_body.pop("model", "")
        cassette_model: str = cassette["request"]["body"].pop("model", "")
        assert request_model == cassette_model, (
            f"Request model {request_model!r} does not match cassette model {cassette_model!r} in {name!r}."
        )
        # The rest of the request body should match exactly
        assert request_body == cassette["request"]["body"]
        return Response(
            200,
            content="".join(cassette["response"]["sse"]).encode(),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )

    http_client = AsyncClient(transport=MockTransport(cassette_handler))
    return http_client


class _MockResponsesEvent(BaseModel, extra="forbid"):
    sse: str

    def as_sse(self) -> str:
        return self.sse


async def _mock_responses(
    self: Molto,
    *,
    body: dict[str, Any],
    endpoint: Literal["responses", "chat/completions"] = "responses",
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
    response = await self._http_client.post(
        f"https://api.openai.com/v1/{endpoint}",
        json=body,
    )
    response.raise_for_status()

    async def iter_events() -> AsyncIterator[BaseModel]:
        for event in response.text.split("\n\n"):
            if event:
                yield _MockResponsesEvent(sse=f"{event}\n\n")

    return iter_events()


async def test_extra_inputs(monkeypatch: pytest.MonkeyPatch):
    # TODO: Remove monkeypatching
    monkeypatch.setattr(Molto, "responses", _mock_responses, raising=False)
    cassette_names = [
        "standard_text-stream-00.yaml",
    ]

    async with _get_http_client(cassette_names) as http_client:
        client = Molto(http_client=http_client)

        # --- Extra inputs --- #
        cassette = _read_cassette(CASSETTE_DIR / cassette_names[0])
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            await client.responses(
                body=dict(
                    messages=[
                        {"role": "user", "content": "Write one short sentence about the ocean."}
                    ],
                    **cassette["request"]["body"],
                )
            )


async def test_standard_text_stream(monkeypatch: pytest.MonkeyPatch):
    # TODO: Remove monkeypatching
    monkeypatch.setattr(Molto, "responses", _mock_responses, raising=False)
    cassette_names = [
        "instructions-stream-00.yaml",
        "instructions-stream-01.yaml",
    ]

    async with _get_http_client(cassette_names) as http_client:
        client = Molto(http_client=http_client)

        # --- Multi turn --- #
        for _ in range(2):
            cassette = _read_cassette(CASSETTE_DIR / cassette_names[0])
            response = await client.responses(
                body=cassette["request"]["body"],
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


async def test_reasoning_stream(monkeypatch: pytest.MonkeyPatch):
    # TODO: Remove monkeypatching
    monkeypatch.setattr(Molto, "responses", _mock_responses, raising=False)
    cassette_names = [
        "reasoning-stream-00.yaml",
        "reasoning-stream-01.yaml",
    ]

    async with _get_http_client(cassette_names) as http_client:
        client = Molto(http_client=http_client)

        # --- Multi turn --- #
        for _ in range(2):
            cassette = _read_cassette(CASSETTE_DIR / cassette_names[0])
            response = await client.responses(
                body=cassette["request"]["body"],
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
