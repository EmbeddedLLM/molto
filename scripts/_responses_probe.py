import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from molto.configs import ENV_CONFIG

MODEL = "gpt-4.1-mini"
QUESTION = "How many Rs are in strawberry?"


def openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=ENV_CONFIG.openai_api_key_plain or "missing-api-key")


def print_value(label: str, value: Any) -> None:
    print(f"{label}:")
    print("----")
    print(value)
    print("----")


async def print_stream(label: str, stream: AsyncIterator[Any]) -> None:
    print(f"{label}:")
    started = time.perf_counter()
    first_event_at: float | None = None
    event_count = 0
    async for event in stream:
        if first_event_at is None:
            first_event_at = time.perf_counter()
        event_count += 1
        print("----")
        print(event)
    print("----")
    first_event_seconds = first_event_at - started if first_event_at is not None else None
    print(
        f"Events: {event_count}; first event: {first_event_seconds!r}s; "
        f"total: {time.perf_counter() - started:.2f}s"
    )
