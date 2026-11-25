"""Raw reasoning profile checks; synthetic history does not establish provider support."""

from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from molto import Molto
from molto.utils.io import json_loads, read_yaml
from molto.utils.test import collect_sdk_response


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        ("openrouter", "responses"),
        ("fireworks", "responses"),
        ("fireworks", "chat/completions"),
        ("togetherai", "responses"),
        ("togetherai", "chat/completions"),
        ("vllm", "responses"),
        ("vllm", "chat/completions"),
        ("bedrock", "chat/completions"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", ["", "Prior thought"])
async def test_raw_history_preserves_signatures_on_the_wire(provider, endpoint, stream, thinking):
    """Ensures signed raw history survives every fixed raw profile on both wires."""
    body = {
        "model": "synthetic",
        "max_tokens": 32,
        "stream": stream,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": thinking, "signature": "synthetic-state"}
                ],
            }
        ],
    }
    before = deepcopy(body)
    cassette_protocol = endpoint.replace("/", "_")
    recording = read_yaml(
        Path(__file__).parents[1]
        / f"cassettes/matched/simple_text/openai-{cassette_protocol}-stream-00.yaml"
    )
    requests = []

    def handle(request):
        requests.append(json_loads(request.content))
        return httpx.Response(
            200,
            content="".join(recording["response"]["sse"]),
            headers={"content-type": "text/event-stream"},
        )

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        result = await client.messages(
            body,
            provider=provider,
            endpoint=endpoint,
            chat_signature_namespace="issuer",
            api_key="test-key",
            base_url="https://example.test",
        )
        if stream:
            events = [event async for event in result]
            result = collect_sdk_response("messages", events)
    assert result.content
    if endpoint == "responses":
        (item,) = requests[0]["input"]
        assert item["encrypted_content"] == "synthetic-state"
        assert [part["text"] for part in item.get("content", [])] == (
            [thinking] if thinking else []
        )
        assert item["summary"] == []
    else:
        (item,) = requests[0]["messages"]
        assert item["extra_content"] == {"issuer": {"thought_signature": "synthetic-state"}}
        assert (item.get("reasoning_content") or item.get("reasoning") or "") == thinking
    assert body == before
