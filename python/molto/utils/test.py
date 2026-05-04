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
    if stream:
        path = path.with_stem(f"{path.stem}-stream")
    data = read_yaml(path)["response"]
    return data["sse"] if stream else data["body"]


def read_cassette_request(path: Path, stream: bool = False) -> dict[str, Any]:
    if stream:
        path = path.with_stem(f"{path.stem}-stream")
    return read_yaml(path)["request"]["body"]


def make_handler(
    cassette_path: Path,
    req_body: dict[str, Any],
    assert_attrs: list[str],
) -> partial[httpx.Response]:
    def handler(request: httpx.Request, load_stream_cassette: bool) -> httpx.Response:
        try:
            assert request.method == "POST"
            body = json_loads(request.content)
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
    handler: Any,
) -> Molto:
    return Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(partial(handler, load_stream_cassette=True)),
            base_url="http://testserver",
        ),
    )


def prepend_model_prefix(body: dict[str, Any], prefix: str = "openai/") -> dict[str, Any]:
    copy_body = deepcopy(body)
    copy_body["model"] = prefix + copy_body["model"]
    return copy_body
