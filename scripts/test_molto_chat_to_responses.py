import asyncio
import os

from _responses_probe import (
    MODEL,
    QUESTION,
    print_stream,
    print_value,
)

from molto import Molto
from molto.types.openai.responses import OpenAIResponsesRequest


async def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    molto = Molto()
    response = await molto.responses(
        OpenAIResponsesRequest(
            model=f"openai/{MODEL}",
            input=QUESTION,
        ),
        api_key=api_key,
        endpoint="chat_completions",
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
        ),
        api_key=api_key,
        endpoint="chat_completions",
    )
    await print_stream(
        "Molto Responses request -> Chat execution -> Responses stream output",
        stream,
    )


asyncio.run(main())
