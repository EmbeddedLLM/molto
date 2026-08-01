import asyncio

from _responses_probe import (
    MODEL,
    QUESTION,
    print_stream,
    print_value,
)

from molto import Molto
from molto.types.openai.responses import OpenAIResponsesRequest


async def main() -> None:
    molto = Molto()
    response = await molto.chat_completions(
        OpenAIResponsesRequest(model=f"openai/{MODEL}", input=QUESTION)
    )
    print_value(
        "Molto Responses request -> Responses execution -> Chat non-stream output",
        response,
    )

    stream = await molto.chat_completions(
        OpenAIResponsesRequest(model=f"openai/{MODEL}", input=QUESTION, stream=True)
    )
    await print_stream(
        "Molto Responses request -> Responses execution -> Chat stream output",
        stream,
    )


asyncio.run(main())
