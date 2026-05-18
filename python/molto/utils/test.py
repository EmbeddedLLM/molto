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


def extract_cassette_response(
    path: Path,
    stream: bool,
    terminal_event_type: str = "response.completed",
) -> dict[str, Any]:
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


def extract_stream_cassette_response(
    path: Path,
    terminal_event_type: str = "response.completed",
) -> dict[str, Any]:
    return extract_cassette_response(path, stream=True, terminal_event_type=terminal_event_type)


def extract_output_ids(cassette_resp: dict[str, Any], *types: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for output_item in cassette_resp["output"]:
        item_type = output_item["type"]
        if item_type in types and item_type not in result:
            result[item_type] = output_item["id"]
    return result


def assert_usage(
    usage: Any, *, output_tokens_exact: int | None = None, expect_reasoning: bool = False
) -> None:
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    if output_tokens_exact is not None:
        assert usage.output_tokens == output_tokens_exact
    if expect_reasoning:
        assert usage.output_tokens_details is not None
        assert usage.output_tokens_details.reasoning_tokens > 0


def assert_stream_text_events(events: list[Any], expected_msg_id: str | None = None) -> str:
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
    assert len(msg_item_added) == 1
    assert len(msg_item_done) == 1
    added_id = msg_item_added[0].item.id
    done_id = msg_item_done[0].item.id
    assert added_id is not None
    assert added_id.startswith("msg_")
    assert added_id == done_id
    if expected_msg_id is not None:
        assert added_id == expected_msg_id

    text_deltas = [
        e
        for e in events
        if isinstance(e, OpenAIResponsesStreamText) and e.type == "response.output_text.delta"
    ]
    assert len(text_deltas) >= 1
    for delta in text_deltas:
        assert delta.item_id == added_id

    text_done = [
        e
        for e in events
        if isinstance(e, OpenAIResponsesStreamText) and e.type == "response.output_text.done"
    ]
    assert len(text_done) == 1
    assert text_done[0].item_id == added_id

    return added_id


def make_handler(
    cassette_path: Path,
    req_body: dict[str, Any] | None = None,
    assert_attrs: list[str] | None = None,
) -> partial[httpx.Response]:
    cassette = read_yaml(cassette_path)
    resp = cassette.get("response")
    is_error = (
        resp is not None and isinstance(resp, dict) and resp.get("status_code", 200) >= 400
    )

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
    if cassette_path is not None:
        handler = make_handler(cassette_path)
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
