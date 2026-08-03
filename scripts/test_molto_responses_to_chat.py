import asyncio

from _responses_probe import (
    MODEL,
    QUESTION,
    print_stream,
    print_value,
)

from molto import Molto
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest


async def main() -> None:
    molto = Molto()
    response = await molto.chat_completions(
        OpenAIChatCompletionRequest(
            model=f"openai/{MODEL}",
            messages=[{"role": "user", "content": QUESTION}],
        ),
        upstream="responses",
    )
    print_value(
        "Molto Chat request -> Responses execution -> Chat non-stream output",
        response,
    )

    stream = await molto.chat_completions(
        OpenAIChatCompletionRequest(
            model=f"openai/{MODEL}",
            messages=[{"role": "user", "content": QUESTION}],
            stream=True,
        ),
        upstream="responses",
    )
    await print_stream(
        "Molto Chat request -> Responses execution -> Chat stream output",
        stream,
    )


asyncio.run(main())
