import asyncio

from molto import Molto
from molto.types.openresponses import OpenResponsesOutputItemDone, OpenResponsesRequest

MODEL = "anthropic/claude-sonnet-4-5"
QUESTION = "How many Rs are in strawberry?"


async def main() -> None:
    molto = Molto()

    response = await molto.responses(OpenResponsesRequest(model=MODEL, input=QUESTION))
    print("Non-stream response:")
    print("----")
    print(response)
    print("----")

    stream = await molto.responses(OpenResponsesRequest(model=MODEL, input=QUESTION, stream=True))
    output_items = []

    print("\nStream events:")
    async for event in stream:
        print("----")
        print(event)
        if isinstance(event, OpenResponsesOutputItemDone):
            output_items.append(event.item)
    print("----")

    print("\nStream output:")
    for item in output_items:
        print(item)


asyncio.run(main())
