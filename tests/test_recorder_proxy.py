"""Offline regressions for the standalone provider evidence recorder."""

import json
from pathlib import Path

import httpx
import pytest
from starlette.requests import Request
from yaml import safe_load

from molto.utils.test import OpenAICassetteManager, OpenAICassetteTransport
from scripts import recorder_proxy


@pytest.fixture
def recording_dir(monkeypatch, tmp_path):
    """Ensure recorder tests write temporary artifacts, never committed cassettes."""
    monkeypatch.setattr(recorder_proxy, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(recorder_proxy, "TARGET_HOST", "https://provider.test/prefix")
    return tmp_path


def read_recording(path: Path) -> dict:
    files = list(path.glob("*.yaml"))
    assert len(files) == 1
    return safe_load(files[0].read_text())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("status", "content_type", "body"),
    [
        (400, "application/json", b""),
        (502, "text/html", b"<html>Bad gateway</html>\n"),
        (429, "application/json; charset=utf-8", b'{ "error": "slow down" }\n'),
        (400, "application/problem+json", b'{"detail":"bad request"}'),
        (503, "application/json", b"not json"),
        (400, "application/octet-stream", b"\xff\x00"),
        (200, "application/json", b'{ "id": "result" }\n'),
    ],
)
async def test_response_bytes_and_errors_survive_proxy(
    monkeypatch, recording_dir, stream, status, content_type, body
):
    """Ensures all result modes retain status, exact body bytes, headers, and one attempt."""
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://provider.test/prefix/v1/responses"
        return httpx.Response(
            status,
            content=body,
            headers={
                "content-type": content_type,
                "x-request-id": "request-1",
                "retry-after": "1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=recorder_proxy.app), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/responses", json={"model": "m", "stream": stream})

    assert len(requests) == 1
    assert response.status_code == status
    assert response.content == body
    assert response.headers["content-type"] == content_type
    assert response.headers["x-request-id"] == "request-1"
    captured = read_recording(recording_dir)["response"]
    assert captured["body_bytes"] == body
    assert captured["status_code"] == status
    assert captured["headers"]["retry-after"] == "1"
    if "json" in content_type and body.startswith(b"{"):
        assert captured["body"] == json.loads(body)


class ProviderStream(httpx.AsyncByteStream):
    def __init__(self, chunks, fail=False):
        self.chunks = chunks
        self.fail = fail
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise httpx.ReadError("URL may contain credentials")

    async def aclose(self):
        self.closed = True


def proxy_request():
    async def receive():
        return {"type": "http.request", "body": b'{"stream":true}', "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        },
        receive,
    )


@pytest.mark.parametrize(
    ("chunks", "terminal", "error"),
    [
        ([b'data: {"choices":[]}\n\ndata: [DONE]\n\n'], "[DONE]", False),
        (
            [b'event: response.incomplete\ndata: {"type":"response.incomplete"}\n\n'],
            "response.incomplete",
            False,
        ),
        ([b'event: message_stop\ndata: {"type":"message_stop"}\n\n'], "message_stop", False),
        ([b'data: {"error":{"message":"invalid request"}}\n\n'], None, True),
        ([b'data: {"choices":[]}\n\n'], None, False),
    ],
)
async def test_stream_capture_reports_eof_and_terminal_evidence(
    monkeypatch, recording_dir, chunks, terminal, error
):
    """Ensures EOF, protocol termination, and HTTP 200 stream errors remain distinct evidence."""
    source = ProviderStream(chunks)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, stream=source, headers={"content-type": "text/event-stream; charset=utf-8"}
            )
        )
    ) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        response = await recorder_proxy.proxy_request(proxy_request(), "v1/responses")
        messages = []

        async def send(message):
            messages.append(message)

        await response.stream_response(send)

    captured = read_recording(recording_dir)["response"]
    assert captured["status_code"] == 200
    assert captured["stream_status"] == "complete"
    assert captured["terminal_event"] == terminal
    assert captured["error_event"] == error
    assert "".join(captured["sse"]).encode() == b"".join(chunks)
    assert b"".join(item.get("body", b"") for item in messages) == b"".join(chunks)
    assert source.closed


@pytest.mark.parametrize("disconnect_at", ["http.response.start", "http.response.body"])
async def test_disconnected_stream_is_saved_and_closed(monkeypatch, recording_dir, disconnect_at):
    """Ensures disconnects during headers or body save partial evidence and close transport."""
    source = ProviderStream([b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, stream=source, headers={"content-type": "text/event-stream"}
            )
        )
    ) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        response = await recorder_proxy.proxy_request(proxy_request(), "v1/responses")

        async def send(message):
            if message["type"] == disconnect_at:
                raise OSError("client disconnected")

        with pytest.raises(OSError, match="client disconnected"):
            await response.stream_response(send)

    captured = read_recording(recording_dir)["response"]
    assert captured["stream_status"] == "disconnected"
    assert captured["terminal_event"] is None
    assert not any("[DONE]" in line for line in captured["sse"])
    assert source.closed


async def test_failed_stream_propagates_and_retains_partial_evidence(monkeypatch, recording_dir):
    """Ensures an upstream read failure is not swallowed or reported as successful completion."""
    source = ProviderStream([b'data: {"choices":[]}\n\n'], fail=True)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, stream=source, headers={"content-type": "text/event-stream"}
            )
        )
    ) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        response = await recorder_proxy.proxy_request(proxy_request(), "v1/responses")

        async def send(message):
            pass

        with pytest.raises(httpx.ReadError):
            await response.stream_response(send)

    captured = read_recording(recording_dir)["response"]
    assert captured["stream_status"] == "transport_error"
    assert captured["transport_error"] == "ReadError"
    assert captured["terminal_event"] is None
    assert captured["sse"] == ['data: {"choices":[]}\n', "\n"]
    assert source.closed


async def test_credentials_are_masked_in_recordings_only(monkeypatch, recording_dir):
    """Ensures echoed credentials are masked while continuation state and actual traffic survive."""
    payload = {
        "extra_content": {"google": {"thought_signature": "opaque-state"}},
        "key": "semantic-key",
    }

    def handler(request):
        assert request.headers["authorization"] == "Bearer header-secret"
        assert request.headers["x-api-key"] == "api-secret"
        assert request.url.params["key"] == "query-secret"
        assert json.loads(request.content) == payload
        return httpx.Response(400, text="header-secret api-secret query-secret opaque-state")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=recorder_proxy.app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/v1/responses?key=query-secret",
                json=payload,
                headers={"authorization": "Bearer header-secret", "x-api-key": "api-secret"},
            )

    assert response.text == "header-secret api-secret query-secret opaque-state"
    captured = read_recording(recording_dir)
    assert captured["request"]["body"] == payload
    assert captured["request"]["query_params"]["key"] == "***"
    assert captured["request"]["headers"]["authorization"] == "Bearer ***"
    assert captured["response"]["body_bytes"] == b"*** *** *** opaque-state"
    assert "secret" not in next(recording_dir.glob("*.yaml")).read_text()


@pytest.mark.parametrize("body", [b"", b'{ "error": "bad request" }\n', b"\xff\x00"])
def test_replay_uses_authoritative_response_bytes(monkeypatch, tmp_path, body):
    """Ensures replay does not reserialize JSON or corrupt empty and binary error bodies."""
    manager = OpenAICassetteManager("recorder", "recorder", "recorder", False, root=tmp_path)
    cassette = {
        "request": {
            "method": "POST",
            "path": "/v1/responses",
            "body": {"model": "m"},
            "query_params": {},
            "headers": {"content-type": "application/json"},
        },
        "response": {
            "status_code": 400,
            "headers": {"content-type": "application/json", "x-request-id": "request-1"},
            "body": {"error": "parsed value is not authoritative"},
            "body_bytes": body,
        },
    }
    monkeypatch.setattr(manager, "read", lambda _: cassette)
    transport = OpenAICassetteTransport(manager, playback_only=True)
    request = httpx.Request("POST", "https://provider.test/v1/responses", json={"model": "m"})
    response = transport._response_from_cassette(
        request, {"model": "m"}, tmp_path / "capture.yaml"
    )
    assert response.content == body
    assert response.status_code == 400
    assert response.headers["x-request-id"] == "request-1"


@pytest.mark.parametrize("content_type", ["application/json", "text/html"])
async def test_partial_non_streaming_error_retains_received_bytes(
    monkeypatch, recording_dir, content_type
):
    """Ensures truncated error bodies retain observed bytes and failure status without fabrication."""
    body = b'{"error":"truncated' if content_type == "application/json" else b"<html>truncated"
    source = ProviderStream([body], fail=True)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(502, stream=source, headers={"content-type": content_type})
        )
    ) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        with pytest.raises(httpx.ReadError):
            await recorder_proxy.proxy_request(proxy_request(), "v1/responses")
    captured = read_recording(recording_dir)["response"]
    assert captured["status_code"] == 502
    assert captured["body_bytes"] == body
    assert captured["body"] == body.decode()
    assert captured["transport_error"] == "ReadError"
    assert source.closed


def test_standalone_proxy_disables_credential_bearing_access_logs(monkeypatch, recording_dir):
    """Ensures standalone startup cannot expose query credentials through Uvicorn access logs."""
    captured = {}
    monkeypatch.setattr("sys.argv", ["recorder_proxy.py"])
    monkeypatch.setattr(
        recorder_proxy.uvicorn, "run", lambda app, **options: captured.update(options)
    )
    recorder_proxy.main()
    assert captured["access_log"] is False
