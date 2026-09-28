"""Citation projection shared by Messages requests and responses."""

from typing import Any

from molto.types.normalized import TextOutput


def messages_text_blocks(
    part: TextOutput,
) -> list[dict[str, Any]]:
    """Split text at citation boundaries while preserving attribution."""
    text = part.content
    boundaries = {0, len(text)}
    for citation in part.citations:
        boundaries.add(citation.start_index)
        boundaries.add(citation.end_index)

    if any(offset < 0 or offset > len(text) for offset in boundaries):
        raise ValueError("Citation offsets fall outside the associated answer text.")
    if any(citation.end_index < citation.start_index for citation in part.citations):
        raise ValueError("Citation end precedes its start.")

    ordered_boundaries = sorted(boundaries)
    intervals = set(zip(ordered_boundaries, ordered_boundaries[1:], strict=False))
    # Point annotations have no answer characters but still carry attribution.
    for citation in part.citations:
        if citation.start_index == citation.end_index:
            intervals.add((citation.start_index, citation.end_index))

    blocks: list[dict[str, Any]] = []
    for start, end in sorted(intervals):
        block: dict[str, Any] = {"type": "text", "text": text[start:end]}
        block_citations = []
        for citation in part.citations:
            if start == end:
                matches = citation.start_index == start and citation.end_index == end
            else:
                matches = citation.start_index <= start and end <= citation.end_index
            if matches:
                block_citations.append(citation.to_messages(text=text))

        if block_citations:
            block["citations"] = block_citations
        blocks.append(block)
    return blocks or [{"type": "text", "text": text}]
