from collections.abc import AsyncIterable

from .aggregates import OutputIR
from .events import OutputEvent, OutputEventSequence
from .state import StateReferences


async def collect_output_ir(
    events: AsyncIterable[OutputEvent],
    *,
    state: StateReferences | None = None,
) -> OutputIR:
    """Validate one finite provider event stream and build its terminal snapshot."""

    collected = tuple([event async for event in events])
    event_sequence = OutputEventSequence(events=collected)
    return OutputIR.from_output_events(event_sequence, state=state)
