"""OpenRouter Responses interprets explicit wire reasoning without changing other protocols."""

import httpx
import pytest

from molto.providers import PROVIDER_CAPABILITIES
from molto.types.normalized import NormalizedRequest, PartEndEvent, Thinking
from molto.upstream.openrouter import OpenRouterResponsesClient
from tests.upstream.test_openai import _response_event, _unexpected_http


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


@pytest.mark.parametrize("raw", [False, True])
async def test_wire_reasoning_preserves_raw_or_signed_summary(monkeypatch, raw):
    """Ensures explicit raw text and signed summaries retain their distinct IR representations."""
    item = reasoning_item(
        **(
            {"content": [{"type": "reasoning_text", "text": "Thought"}], "format": "unknown"}
            if raw
            else {
                "summary": [{"type": "summary_text", "text": "Thought"}],
                "encrypted_content": "opaque",
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
            signature=None if raw else "opaque",
        ),
    )


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        (
            {"summary": [{"type": "summary_text", "text": "Summary"}]},
            "summary requires a signature",
        ),
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
        (
            {
                "content": [{"type": "reasoning_text", "text": "Raw"}],
                "summary": [{"type": "summary_text", "text": "Summary"}],
                "encrypted_content": "opaque",
            },
            "both raw reasoning and a summary",
        ),
        (
            {
                "content": [{"type": "reasoning_text", "text": "Raw"}],
                "encrypted_content": "opaque",
            },
            "signature cannot accompany raw reasoning",
        ),
    ],
)
async def test_wire_reasoning_keeps_format_and_signature_boundaries(monkeypatch, fields, error):
    """Ensures wire interpretation does not accept unsigned summaries, opaque formats, or mixed parts."""
    with pytest.raises(ValueError, match=error):
        await normalized_events(monkeypatch, [item_event(reasoning_item(**fields), done=True)])


@pytest.mark.parametrize("first_raw", [False, True])
async def test_wire_reasoning_cannot_change_representation_mid_item(monkeypatch, first_raw):
    """Ensures individually valid deltas cannot mix raw and summarized reasoning in one item."""
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
    with pytest.raises(ValueError, match="cannot mix raw text"):
        await normalized_events(monkeypatch, [item_event(reasoning_item()), *deltas])
