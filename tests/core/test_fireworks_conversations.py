"""Replay independently recorded Fireworks protocol conversations and translated output."""

import json
from pathlib import Path

import pytest

from tests.core.test_recorded_provider_conversations import (
    replay_conversation,
    replay_translated_first_turn,
)

ROOT = Path(__file__).parents[1] / "cassettes/provider_recordings/2026-11-04"
GROUPS = [
    path.parent
    for path in sorted(ROOT.glob("fireworks-*/manifest.json"))
    if json.loads(path.read_text())["status"] == "complete"
]


@pytest.mark.parametrize("group", GROUPS, ids=lambda path: path.name)
async def test_fireworks_native_conversation_sdk_parity(group):
    """Ensures native Fireworks output and actual subsequent history match independent SDK replay."""
    manifest = json.loads((group / "manifest.json").read_text())
    await replay_conversation(group, protocol=manifest["upstream_protocol"])


@pytest.mark.parametrize(
    ("group", "downstream"),
    [
        pytest.param(group, downstream, id=f"{group.name}-{downstream}")
        for group in GROUPS
        if json.loads((group / "manifest.json").read_text())["stream"]
        for downstream in ("chat/completions", "responses", "messages")
        if downstream != json.loads((group / "manifest.json").read_text())["upstream_protocol"]
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_fireworks_translated_first_turn_sdk_semantics(group, downstream, stream):
    """Ensures recorded first-turn content, calls, usage and lifecycle survive public translation."""
    await replay_translated_first_turn(group, downstream, stream)
