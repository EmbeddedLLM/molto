from typing import Any

import pytest
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTextAnnotationAddedEvent,
)
from openai.types.responses.response_output_text import AnnotationURLCitation

from molto.types.openai.responses import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesStreamAnnotation,
)
from tests.support import (
    STREAM_PARAMS,
    assert_events_before_terminal,
    run_responses_parity,
)

REQUEST_BODY = {
    "model": "gpt-5-nano",
    "input": [
        {
            "role": "user",
            "content": (
                "Search openai.com for when OpenAI introduced the Responses API. "
                "Answer in one sentence and include an inline citation to the source."
            ),
        }
    ],
    "tools": [
        {
            "type": "web_search",
            "search_context_size": "high",
            "filters": {"allowed_domains": ["openai.com"]},
        }
    ],
    "tool_choice": "required",
}


def _annotation_payload(annotation: Any) -> Any:
    if hasattr(annotation, "model_dump"):
        return annotation.model_dump(mode="python", exclude_unset=True)
    return annotation


def _annotation_event_payload(event: Any) -> dict[str, Any]:
    return {
        "annotation": _annotation_payload(event.annotation),
        "annotation_index": event.annotation_index,
        "content_index": event.content_index,
        "item_id": event.item_id,
        "output_index": event.output_index,
        "sequence_number": event.sequence_number,
    }


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_url_citations_round_trip_through_sdk(stream: bool, cassette_manager) -> None:
    """Preserve populated URL citations from parsed SDK output through Molto."""
    req_body = {**REQUEST_BODY, "stream": stream}

    parity = await run_responses_parity(cassette_manager, req_body)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    if stream:
        sdk_annotation_events = [
            event
            for event in parity.sdk_events
            if isinstance(event, ResponseOutputTextAnnotationAddedEvent)
        ]
        assert sdk_annotation_events
        assert_events_before_terminal(parity.sdk_events, "response.output_text.annotation.added")

    sdk_message = next(
        item for item in sdk_response.output if isinstance(item, ResponseOutputMessage)
    )
    sdk_text = next(part for part in sdk_message.content if isinstance(part, ResponseOutputText))
    sdk_citations = [
        annotation
        for annotation in sdk_text.annotations
        if isinstance(annotation, AnnotationURLCitation)
    ]
    assert sdk_citations
    for citation in sdk_citations:
        assert citation.type == "url_citation"
        assert citation.url.startswith(("https://", "http://"))
        assert citation.title
        assert 0 <= citation.start_index < citation.end_index <= len(sdk_text.text)
    sdk_citation_payloads = [_annotation_payload(citation) for citation in sdk_citations]

    if stream:
        molto_annotation_events = [
            event
            for event in parity.molto_events
            if isinstance(event, OpenAIResponsesStreamAnnotation)
        ]
        assert len(molto_annotation_events) == len(sdk_annotation_events)
        assert [_annotation_event_payload(event) for event in molto_annotation_events] == [
            _annotation_event_payload(event) for event in sdk_annotation_events
        ]
        assert_events_before_terminal(parity.molto_events, "response.output_text.annotation.added")

    molto_message = next(item for item in response.output if isinstance(item, OpenAIOutputItem))
    molto_text = next(
        part for part in molto_message.content if isinstance(part, OpenAIOutputTextContent)
    )
    molto_citations = [
        annotation
        for annotation in molto_text.annotations
        if annotation.get("type") == "url_citation"
    ]
    assert molto_text.text == sdk_text.text
    assert molto_citations == sdk_citation_payloads
