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
    response = await molto.responses(
        OpenAIResponsesRequest(
            model=f"openai/{MODEL}",
            input=QUESTION,
            upstream="chat_completions",
        ),
    )
    print_value(
        "Molto Responses request -> Chat execution -> Responses non-stream output",
        response,
    )

    stream = await molto.responses(
        OpenAIResponsesRequest(
            model=f"openai/{MODEL}",
            input=QUESTION,
            stream=True,
            upstream="chat_completions",
        ),
    )
    await print_stream(
        "Molto Responses request -> Chat execution -> Responses stream output",
        stream,
    )


asyncio.run(main())
