"""OpenRouter Responses interprets explicit wire reasoning without changing other protocols."""

import httpx
import pytest

from molto.providers import PROVIDER_CAPABILITIES, ProviderCapabilities
from molto.types.normalized import (
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    Thinking,
    ThinkingDelta,
)
from molto.upstream.openrouter import OpenRouterResponsesClient
from tests.upstream.test_openai import _response_event, _unexpected_http


def test_openrouter_uses_raw_history_profile_with_explicit_response_labels():
    """Ensures ambiguous history is raw while Responses keeps explicit wire labels."""
    profile = ProviderCapabilities(thinking="summary")
    resolved = OpenRouterResponsesClient.resolve_capabilities(profile)
    assert resolved == ProviderCapabilities(thinking="raw")
    assert profile.thinking == "summary"


async def normalized_events(monkeypatch, items):
    async def native_events(request):
        for item in items:
            yield _response_event(item)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_http)) as transport:
        client = OpenRouterResponsesClient(
            transport, capabilities=PROVIDER_CAPABILITIES["openrouter"]
        )
        monkeypatch.setattr(client, "native_events", native_events)
        return [event async for event in client.events(NormalizedRequest(model="m"))]


def reasoning_item(**fields):
    return {"id": "rs_1", "type": "reasoning", "summary": [], **fields}


def item_event(item, *, done=False):
    return {
        "type": "response.output_item.done" if done else "response.output_item.added",
        "output_index": 0,
        "sequence_number": 0,
        "item": item,
    }


@pytest.mark.parametrize(("raw", "signature"), [(True, None), (False, None), (False, "opaque")])
async def test_wire_reasoning_preserves_raw_or_summary(monkeypatch, raw, signature):
    """Ensures explicit raw text and summaries preserve their representation and observed state."""
    item = reasoning_item(
        **(
            {"content": [{"type": "reasoning_text", "text": "Thought"}], "format": "unknown"}
            if raw
            else {
                "summary": [{"type": "summary_text", "text": "Thought"}],
                "encrypted_content": signature,
            }
        )
    )
    events = await normalized_events(monkeypatch, [item_event(item), item_event(item, done=True)])
    assert events[-1] == PartEndEvent(
        index=0,
        part=Thinking(
            id="rs_1",
            content="Thought" if raw else "",
            summary="" if raw else "Thought",
            signature=signature,
        ),
    )


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        (
            {
                "summary": [{"type": "summary_text", "text": "Summary"}],
                "encrypted_content": "opaque",
                "format": "unknown",
            },
            "format cannot be translated",
        ),
        (
            {"content": [{"type": "reasoning_text", "text": "Raw"}], "format": "named-v1"},
            "format cannot be translated",
        ),
    ],
)
async def test_wire_reasoning_rejects_unsupported_formats(monkeypatch, fields, error):
    """Ensures wire interpretation rejects unsupported opaque formats."""
    with pytest.raises(ValueError, match=error):
        await normalized_events(monkeypatch, [item_event(reasoning_item(**fields), done=True)])


@pytest.mark.parametrize("first_raw", [False, True])
async def test_wire_reasoning_preserves_both_delta_labels(monkeypatch, first_raw):
    """Ensures both delta labels survive in a single item in either arrival order."""
    deltas = [
        {
            "type": "response.reasoning_text.delta"
            if raw
            else "response.reasoning_summary_text.delta",
            "content_index" if raw else "summary_index": 0,
            "item_id": "rs_1",
            "output_index": 0,
            "sequence_number": index,
            "delta": "Thought",
        }
        for index, raw in enumerate((first_raw, not first_raw))
    ]
    item = reasoning_item(
        content=[{"type": "reasoning_text", "text": "Thought"}],
        summary=[{"type": "summary_text", "text": "Thought"}],
        encrypted_content="opaque",
    )
    events = await normalized_events(
        monkeypatch, [item_event(reasoning_item()), *deltas, item_event(item, done=True)]
    )
    expected = [ThinkingDelta(content_delta="Thought"), ThinkingDelta(summary_delta="Thought")]
    assert [event.delta for event in events if isinstance(event, PartDeltaEvent)] == [
        *(expected if first_raw else reversed(expected)),
        ThinkingDelta(signature_delta="opaque"),
    ]
    assert events[-1] == PartEndEvent(
        index=0, part=Thinking(id="rs_1", content="Thought", summary="Thought", signature="opaque")
    )
