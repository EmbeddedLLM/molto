import asyncio

from _responses_probe import (
    MODEL,
    QUESTION,
    print_stream,
    print_value,
)

from molto import Molto
from molto.types.openai import OpenAIRequest


async def main() -> None:
    molto = Molto()
    response = await molto.responses(
        OpenAIRequest.chat(
            model=f"openai/{MODEL}",
            messages=[{"role": "user", "content": QUESTION}],
        )
    )
    print_value(
        "Molto Chat request -> Chat execution -> Responses non-stream output",
        response,
    )

    stream = await molto.responses(
        OpenAIRequest.chat(
            model=f"openai/{MODEL}",
            messages=[{"role": "user", "content": QUESTION}],
            stream=True,
        )
    )
    await print_stream(
        "Molto Chat request -> Chat execution -> Responses stream output",
        stream,
    )


asyncio.run(main())
