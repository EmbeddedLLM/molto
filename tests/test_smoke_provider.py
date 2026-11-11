"""Offline checks for the explicitly invoked live smoke runner."""

from types import SimpleNamespace

import httpx
import pytest

from molto.utils.io import dumps_sse, json_loads
from scripts import smoke_provider


def options(**overrides):
    return SimpleNamespace(
        **{
            "provider": "vllm",
            "upstream": "chat/completions",
            "downstream": None,
            "model": "test-model",
            "mode": "both",
            "base_url": "https://provider.test/v1",
            "extra_body": None,
            "max_tokens": 128,
            "max_requests": 18,
            "timeout": 2,
            "deadline": 10,
            **overrides,
        }
    )


def chat_response(request, *, tool_finish="tool_calls"):
    body = json_loads(request.content)
    tool_turn = isinstance(body.get("tool_choice"), dict)
    message = {"role": "assistant", "content": None if tool_turn else "5"}
    if tool_turn:
        message["tool_calls"] = [
            {
                "id": "call-add",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
            }
        ]
    finish = tool_finish if tool_turn else "stop"
    common = {"id": "chat-smoke", "model": "test-model", "created": 1}
    usage = {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
    if body.get("stream"):
        if tool_turn:
            message["tool_calls"][0]["index"] = 0
        chunk = {**common, "object": "chat.completion.chunk"}
        events = [
            {**chunk, "choices": [{"index": 0, "delta": message, "finish_reason": None}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {**chunk, "choices": [], "usage": usage},
        ]
        return httpx.Response(
            200,
            text="".join(dumps_sse(event) for event in events) + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        json={
            **common,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        },
    )


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient

    class MockClient(original):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(smoke_provider.httpx, "AsyncClient", MockClient)


async def test_smoke_runs_native_and_translated_tool_continuations(monkeypatch, capsys):
    """Ensures the runner exercises all protocols/modes and reuses actual calls in tool history."""
    requests = []

    def handle(request):
        body = json_loads(request.content)
        requests.append(body)
        if body.get("tool_choice") == "none":
            result = body["messages"][-1]
            assert result["role"] == "tool" and result["tool_call_id"] == "call-add"
            assert result["content"] in ("5", [{"type": "text", "text": "5"}])
            assert body["messages"][-2]["tool_calls"][0]["id"] == "call-add"
        return chat_response(request)

    mock_http(monkeypatch, handle)
    assert await smoke_provider.run(options()) == 0
    assert len(requests) == 18
    assert sum(body.get("tool_choice") == "none" for body in requests) == 6
    assert capsys.readouterr().out.count("PASS ") == 12


async def test_smoke_bounds_requests_and_does_not_print_provider_errors(monkeypatch, capsys):
    """Ensures bounded failed runs cannot disclose a provider's credential-bearing error text."""
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "LEAK-SENTINEL", "type": "auth"}})

    mock_http(monkeypatch, handle)
    assert await smoke_provider.run(options(max_requests=1)) == 1
    assert len(requests) == 1
    output = capsys.readouterr().out
    assert "LEAK-SENTINEL" not in output
    assert "status=401" in output
    assert "HTTP attempts: 1/1" in output


async def test_smoke_detects_incorrect_messages_tool_termination(monkeypatch, capsys):
    """Ensures an end_turn response containing a tool call fails the translated smoke check."""
    mock_http(monkeypatch, lambda request: chat_response(request, tool_finish="stop"))
    assert await smoke_provider.run(options(downstream=["messages"], mode="non-stream")) == 1
    assert "FAIL messages non-stream tool: AssertionError" in capsys.readouterr().out


@pytest.mark.parametrize("extra_body", ["{", "[]", '{"max_tokens":100000}'])
def test_smoke_rejects_invalid_extra_body_before_live_execution(extra_body):
    """Ensures malformed provider options fail at argument parsing without starting live work."""
    with pytest.raises(SystemExit) as error:
        smoke_provider.main(
            [
                "--provider",
                "vllm",
                "--model",
                "test",
                "--upstream",
                "responses",
                "--extra-body",
                extra_body,
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("limit", ["--max-tokens", "--max-requests", "--timeout", "--deadline"])
def test_smoke_rejects_nonpositive_limits(limit):
    """Ensures invalid bounds fail before any provider traffic starts."""
    with pytest.raises(SystemExit) as error:
        smoke_provider.main(
            ["--provider", "vllm", "--model", "test", "--upstream", "responses", limit, "0"]
        )
    assert error.value.code == 2
