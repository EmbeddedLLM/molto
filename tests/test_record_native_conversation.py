"""Offline checks of the standalone native Responses/Messages recording driver."""

import argparse
import json
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic
from yaml import safe_load

from scripts import record_native_conversation as driver
from scripts import recorder_proxy


@pytest.fixture
def recording_args(monkeypatch, tmp_path):
    """Ensure driver checks use a temporary output directory and controlled credentials."""
    monkeypatch.setattr(driver, "load_dotenv", lambda **_: None)
    monkeypatch.setenv("TEST_NATIVE_KEY", "native-test-key")
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    return argparse.Namespace(
        provider="fireworks",
        protocol="responses",
        key_env="TEST_NATIVE_KEY",
        model="accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
        base_url="https://api.fireworks.ai/inference/v1",
        output_dir=tmp_path / "run",
        scenario="text",
        reasoning="none",
        stream=False,
        max_tokens=2048,
        auth="api-key",
        extra_body=None,
    )


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scenario", ["text", "tools"])
@pytest.mark.parametrize("bearer", [False, True])
async def test_native_driver_replays_exact_recorded_conversation(
    monkeypatch, recording_args, protocol, stream, scenario, bearer
):
    """Ensures actual native SDK output yields each exact recorded next request and linked tool result."""
    args = recording_args
    args.protocol, args.stream, args.scenario = protocol, stream, scenario
    if bearer:
        args.auth = "bearer"
        args.extra_body = {"provider": {"max_price": {"prompt": 0, "completion": 0}}}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-token")
    args.base_url = "https://api.fireworks.ai/inference" + (
        "/v1" if protocol == "responses" else ""
    )
    mode = "stream" if stream else "nonstream"
    group = (
        Path(__file__).parent
        / "cassettes/provider_recordings/2026-11-04"
        / f"fireworks-{protocol}-{scenario}-{mode}-none"
    )
    manifest = json.loads((group / "manifest.json").read_text())
    recordings = [safe_load((group / turn["files"][0]).read_text()) for turn in manifest["turns"]]
    requests = []

    def handler(request):
        captured = recordings[len(requests)]
        requests.append(json.loads(request.content))
        assert requests[-1] == captured["request"]["body"] | (args.extra_body or {})
        assert str(request.url) == manifest["upstream_url"]
        if protocol == "responses":
            assert request.headers["authorization"] == "Bearer native-test-key"
            assert requests[-1]["store"] is False
        else:
            if bearer:
                assert request.headers["authorization"] == "Bearer native-test-key"
                assert "x-api-key" not in request.headers
            else:
                assert request.headers["x-api-key"] == "native-test-key"
                assert "authorization" not in request.headers
        result = captured["response"]
        if "sse" in result:
            return httpx.Response(
                result["status_code"],
                content="".join(result["sse"]),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            result["status_code"],
            content=result["body_bytes"],
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        assert await driver.record(args) == 0
        assert upstream.is_closed
    actual = json.loads((args.output_dir / "manifest.json").read_text())
    assert actual["status"] == "complete"
    assert actual["extra_body"] == args.extra_body
    assert len(requests) == len(recordings)
    assert all(
        turn["status"] == "complete" and len(turn["files"]) == 1 for turn in actual["turns"]
    )


async def test_messages_driver_rejects_eof_without_message_stop(recording_args):
    """Ensures end_turn without message_stop cannot certify a truncated native Messages stream."""
    args = recording_args
    args.protocol, args.stream = "messages", True
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Partial answer"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
    ]
    data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
        )
    ) as transport:
        async with AsyncAnthropic(api_key="test", http_client=transport, max_retries=0) as client:
            with pytest.raises(ValueError, match="exactly one message_stop"):
                await driver.result_and_history(
                    client,
                    args,
                    {
                        "model": "m",
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )


@pytest.mark.parametrize("stream", [False, True])
async def test_native_driver_retains_incomplete_failure_manifest(
    monkeypatch, recording_args, stream
):
    """Ensures incomplete Responses output retains its HTTP evidence and failed turn instead of successful history."""
    args = recording_args
    args.stream = stream
    payload = {"id": "r", "object": "response", "status": "incomplete", "model": "m", "output": []}

    def handler(request):
        if stream:
            data = {"type": "response.incomplete", "sequence_number": 0, "response": payload}
            return httpx.Response(
                200,
                text="event: response.incomplete\ndata: " + json.dumps(data) + "\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        assert await driver.record(args) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == manifest["turns"][0]["status"] == "incomplete"
    assert manifest["failure"]["type"] == "ValueError"
    assert len(manifest["turns"][0]["files"]) == 1
    assert (args.output_dir / manifest["turns"][0]["files"][0]).exists()


@pytest.mark.parametrize("header", ["Authorization", "x-api-key", "X-API-KEY"])
async def test_native_driver_rejects_conflicting_auth_before_recording(
    monkeypatch, recording_args, header
):
    """Ensures Anthropic custom headers cannot override the manifest's selected provider credential."""
    recording_args.protocol = "messages"
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", header + ": unrelated-key")
    with pytest.raises(ValueError, match="ANTHROPIC_CUSTOM_HEADERS"):
        await driver.record(recording_args)
    assert not recording_args.output_dir.exists()
