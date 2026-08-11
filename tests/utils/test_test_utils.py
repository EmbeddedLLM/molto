from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from molto.utils.io import dumps_sse
from molto.utils.test import reconstructable_responses_event


def _responses_events() -> list[dict[str, Any]]:
    output_item = {
        "id": "msg_expected",
        "type": "message",
        "status": "completed",
        "content": [],
        "role": "assistant",
    }
    return [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**output_item, "status": "in_progress"},
            "sequence_number": 0,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_expected",
                "object": "response",
                "created_at": 123,
                "status": "completed",
                "background": False,
                "completed_at": 123,
                "model": "expected-model",
                "output": [output_item],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": None,
            },
            "sequence_number": 1,
        },
    ]


def _reconstructable(events: list[dict[str, Any]], body: dict[str, Any]) -> list[dict[str, Any]]:
    return [reconstructable_responses_event(dumps_sse(event), body) for event in events]


def test_reconstructable_responses_event_preserves_reconstructable_fields():
    expected_events = _responses_events()
    actual_events = deepcopy(expected_events)
    actual_events[0]["item"]["id"] = "msg_generated"
    actual_events[1]["response"].pop("background")
    actual_events[1]["response"]["completed_at"] = 456
    actual_events[1]["response"]["output"][0]["id"] = "msg_generated"

    assert _reconstructable(actual_events, {}) == _reconstructable(expected_events, {})
    assert _reconstructable(actual_events, {"background": False}) != _reconstructable(
        expected_events, {"background": False}
    )

    actual_events[1]["response"]["model"] = "different-model"
    assert _reconstructable(actual_events, {}) != _reconstructable(expected_events, {})


def test_reconstructable_responses_event_validates_event_and_request():
    event = _responses_events()[0]

    with pytest.raises(ValidationError):
        reconstructable_responses_event(dumps_sse({"type": "invalid"}), {})
    with pytest.raises(ValidationError):
        reconstructable_responses_event(dumps_sse(event), {"invalid": True})
