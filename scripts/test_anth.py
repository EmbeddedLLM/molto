import asyncio

from molto import Molto
from molto.types.openai.responses import OpenAIResponsesRequest

MODEL = "anthropic/claude-sonnet-4-5"
QUESTION = "How many Rs are in strawberry?"


async def main() -> None:
    molto = Molto()

    response = await molto.responses(
        OpenAIResponsesRequest(model=MODEL, input=QUESTION, endpoint="chat_completions"),
    )
    print("Non-stream response:")
    print("----")
    print(response)
    print("----")

    stream = await molto.responses(
        OpenAIResponsesRequest(
            model=MODEL,
            input=QUESTION,
            stream=True,
            endpoint="chat_completions",
        ),
    )
    output_items = []

    print("\nStream events:")
    async for event in stream:
        print("----")
        print(event)
        if event.type == "response.output_item.done":
            output_items.append(event.item)
    print("----")

    print("\nStream output:")
    for item in output_items:
        print(item)


asyncio.run(main())
