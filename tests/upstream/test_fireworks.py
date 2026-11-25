"""Fireworks reasoning semantics retain native wire labels for continuation."""

import httpx
import pytest

from molto.providers import ProviderCapabilities
from molto.types.normalized import (
    NormalizedRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    Thinking,
    ThinkingDelta,
)
from molto.types.openai_responses import OpenAIResponsesStreamEventAdapter
from molto.upstream.fireworks import FireworksChatCompletionsClient, FireworksResponsesClient


@pytest.mark.parametrize("client_type", [FireworksChatCompletionsClient, FireworksResponsesClient])
def test_fireworks_resolves_raw_profile_without_mutating_input(client_type):
    """Ensures both Fireworks clients retain raw policy even with a supplied summary profile."""
    profile = ProviderCapabilities(thinking="summary")
    assert client_type.resolve_capabilities(profile).thinking == "raw"
    assert profile.thinking == "summary"


async def test_summary_labeled_lifecycle_retains_wire_labels(monkeypatch):
    """Ensures Fireworks summary labels survive every part of the Responses lifecycle."""
    items = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "A"}],
            },
        },
        {
            "type": "response.reasoning_summary_part.added",
            "item_id": "rs_1",
            "summary_index": 0,
            "part": {"type": "summary_text", "text": "B"},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "summary_index": 0,
            "delta": "C",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "ABC"}],
            },
        },
    ]

    async def native_events(request):
        for index, item in enumerate(items):
            yield OpenAIResponsesStreamEventAdapter.validate_python(
                {"output_index": 0, "sequence_number": index, **item}
            )

    async with httpx.AsyncClient() as transport:
        client = FireworksResponsesClient(
            transport, capabilities=ProviderCapabilities(thinking="raw")
        )
        monkeypatch.setattr(client, "native_events", native_events)
        events = [event async for event in client.events(NormalizedRequest(model="m"))]
    assert events == [
        PartStartEvent(index=0, part=Thinking(id="rs_1", content="", summary="A")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="B")),
        PartDeltaEvent(index=0, delta=ThinkingDelta(summary_delta="C")),
        PartEndEvent(index=0, part=Thinking(id="rs_1", content="", summary="ABC")),
    ]
