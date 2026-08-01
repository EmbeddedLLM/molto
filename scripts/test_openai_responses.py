import asyncio

from _responses_probe import MODEL, QUESTION, openai_client, print_stream, print_value


async def main() -> None:
    async with openai_client() as client:
        response = await client.responses.create(model=MODEL, input=QUESTION)
        print_value("OpenAI SDK Responses non-stream response", response)

        stream = await client.responses.create(model=MODEL, input=QUESTION, stream=True)
        await print_stream("OpenAI SDK Responses stream events", stream)


asyncio.run(main())
