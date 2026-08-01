import asyncio

from _responses_probe import MODEL, QUESTION, print_stream, print_value

from molto import Molto
from molto.types.openai.responses import OpenAIResponsesRequest


async def main() -> None:
    molto = Molto()
    response = await molto.responses(
        OpenAIResponsesRequest(model=f"openai/{MODEL}", input=QUESTION)
    )
    print_value("Molto Responses non-stream response", response)

    stream = await molto.responses(
        OpenAIResponsesRequest(model=f"openai/{MODEL}", input=QUESTION, stream=True)
    )
    await print_stream("Molto Responses stream events", stream)


asyncio.run(main())
