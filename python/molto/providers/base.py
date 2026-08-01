from collections.abc import AsyncIterator
from typing import Protocol

from molto.types.ir.aggregates import InputIR
from molto.types.ir.events import OutputEvent


class InferenceProvider(Protocol):
    """OpenAI-compatible inference boundary."""

    @property
    def endpoint(self) -> str: ...

    def run(self, request: InputIR) -> AsyncIterator[OutputEvent]:
        """Execute a validated request and emit semantic output events."""
