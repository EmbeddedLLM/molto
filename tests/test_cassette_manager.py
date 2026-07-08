from pathlib import Path

import httpx
import pytest

from molto.utils.test import (
    OpenAICassetteManager,
    OpenAICassetteTransport,
    sanitize_request_body,
    sanitize_test_name,
)


def _manager(tmp_path: Path, node_name: str = "test_demo[stream]") -> OpenAICassetteManager:
    """Create a cassette manager rooted in pytest's temporary directory.

    Example:
        manager = _manager(tmp_path, "test_reasoning[stream]")
        manager.node_slug
        # "test_reasoning-stream"
    """
    return OpenAICassetteManager(
        node_name=node_name,
        node_id=f"tests/test_demo.py::{node_name}",
        module_name="test_demo",
        record=False,
        root=tmp_path,
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("test_reasoning[stream]", "test_reasoning-stream"),
        (
            "test_reasoning[stream-reasoning_no_summary]",
            "test_reasoning-stream-reasoning_no_summary",
        ),
    ],
)
def test_cassette_slugging(name: str, expected: str) -> None:
    """Preserve pytest parameter IDs while making safe filenames.

    Example:
        sanitize_test_name("test_reasoning[stream]")
        # "test_reasoning-stream"
    """
    assert sanitize_test_name(name) == expected


def test_manager_node_slug_uses_sanitized_node_name(tmp_path: Path) -> None:
    """Expose the sanitized pytest node name without a test-prefixed manager method.

    Example:
        _manager(tmp_path, "test_reasoning[stream]").node_slug
        # "test_reasoning-stream"
    """
    manager = _manager(tmp_path, "test_reasoning[stream]")

    assert manager.node_slug == "test_reasoning-stream"


def test_cassette_path_uses_optional_cassette_suffix(tmp_path: Path) -> None:
    """Build the planned cassette path layout from manager metadata.

    Example:
        manager.cassette_path("responses")
        # tmp_path/openai/responses/test_demo/test_demo-stream.yaml
    """
    manager = _manager(tmp_path)

    assert manager.cassette_path("responses") == (
        tmp_path / "openai" / "responses" / "test_demo" / "test_demo-stream.yaml"
    )
    assert manager.cassette_path("responses", "sse") == (
        tmp_path / "openai" / "responses" / "test_demo" / "test_demo-stream-sse.yaml"
    )


def test_missing_cassette_error_includes_record_command(tmp_path: Path) -> None:
    """Report missing playback cassettes with path, node id, and record command.

    Example:
        manager.read(Path("missing.yaml"))
        # raises AssertionError containing "pytest --record-cassettes"
    """
    manager = _manager(tmp_path)
    path = manager.cassette_path("responses")

    with pytest.raises(AssertionError) as exc_info:
        manager.read(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert manager.node_id in message
    assert f"pytest --record-cassettes {manager.node_id}" in message


def test_secret_scrubbing_recurses_through_request_body() -> None:
    """Scrub secret-looking request fields without changing normal fields.

    Example:
        sanitize_request_body({"api_key": "sk", "model": "gpt-4.1"})
        # {"api_key": "***", "model": "gpt-4.1"}
    """
    body = {
        "model": "gpt-4.1",
        "api_key": "sk-test",
        "nested": {"refresh_token": "token", "keep": "value"},
        "items": [{"secret": "hide", "content": "show"}],
    }

    assert sanitize_request_body(body) == {
        "model": "gpt-4.1",
        "api_key": "***",
        "nested": {"refresh_token": "***", "keep": "value"},
        "items": [{"secret": "***", "content": "show"}],
    }


async def test_non_stream_playback_returns_json_body(tmp_path: Path) -> None:
    """Replay a non-stream cassette as a JSON httpx response.

    Example:
        await client.post("/v1/responses", json={"stream": False})
        # response.json() == {"id": "resp_test", "status": "completed"}
    """
    manager = _manager(tmp_path, "test_json[non-stream]")
    manager.write(
        manager.cassette_path("responses"),
        {
            "request": {},
            "response": {
                "status_code": 200,
                "headers": {"content-type": "application/json"},
                "body": {"id": "resp_test", "status": "completed"},
            },
        },
    )
    client = httpx.AsyncClient(
        transport=OpenAICassetteTransport(manager, cassette_suffix=None),
        base_url="https://api.openai.com",
    )

    response = await client.post("/v1/responses", json={"stream": False})

    assert response.status_code == 200
    assert response.json() == {"id": "resp_test", "status": "completed"}


async def test_streaming_sse_playback_returns_event_stream(tmp_path: Path) -> None:
    """Replay an SSE cassette as text/event-stream content.

    Example:
        await client.post("/v1/responses", json={"stream": True})
        # response.text has one data frame followed by a blank line.
    """
    manager = _manager(tmp_path, "test_json[stream]")
    sse_lines = ['data: {"type":"response.created"}\n', "\n"]
    manager.write(
        manager.cassette_path("responses"),
        {
            "request": {},
            "response": {
                "status_code": 200,
                "headers": {"content-type": "text/event-stream; charset=utf-8"},
                "sse": sse_lines,
            },
        },
    )
    client = httpx.AsyncClient(
        transport=OpenAICassetteTransport(manager, cassette_suffix=None),
        base_url="https://api.openai.com",
    )

    response = await client.post("/v1/responses", json={"stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.text == "".join(sse_lines)
