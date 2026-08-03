import json
from base64 import b64encode
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from pydantic import BaseModel

from molto import Molto
from molto.utils.io import read_yaml

CASSETTE_DIR = (
    Path(__file__).parents[1] / "cassettes" / "openai" / "chat_completions" / "test_interface"
)
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
        # Model sent in request might contain provider as prefix
        request_model: str = request_body.pop("model", "")
        cassette_model: str = cassette["request"]["body"].pop("model", "")
        assert request_model.split("/")[-1] == cassette_model, (
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


class _ChatCompletionEvent(BaseModel):
    sse: str

    def as_sse(self) -> str:
        return self.sse


async def _mock_chat_completions(
    self: Molto,
    *,
    body: dict[str, Any],
    **kwargs: Any,
) -> AsyncIterator[BaseModel]:
    assert self._http_client is not None
    response = await self._http_client.post(
        "https://api.openai.com/v1/responses",
        json=body,
        **kwargs,
    )
    response.raise_for_status()

    async def iter_events() -> AsyncIterator[BaseModel]:
        for event in response.text.split("\n\n"):
            if event:
                yield _ChatCompletionEvent(sse=f"{event}\n\n")

    return iter_events()


async def test_openai_chat_completions(monkeypatch: pytest.MonkeyPatch):
    cassette_names = [
        "00-short_sentence-stream.yaml",
    ]

    async with _get_http_client(cassette_names) as http_client:
        client = Molto(http_client=http_client)

        # --- Short sentence --- #
        cassette = _read_cassette(CASSETTE_DIR / cassette_names[0])
        expected_sse_events = [
            f"{event}\n\n" for event in "".join(cassette["response"]["sse"]).split("\n\n") if event
        ]
        # TODO: Remove monkeypatching
        monkeypatch.setattr(Molto, "chat_completions", _mock_chat_completions, raising=False)
        response = await client.chat_completions(
            body=dict(
                model="gpt-5.4-nano",
                instructions="You are a helpful assistant.",
                input="Write one short sentence about the ocean.",
                stream=True,
            )
        )
        events = [event async for event in response]
        assert len(events) == len(expected_sse_events)
        for event, expected_sse in zip(events, expected_sse_events, strict=True):
            assert isinstance(event, BaseModel)
            assert event.as_sse() == expected_sse
