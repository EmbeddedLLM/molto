import httpx
from openai import AsyncOpenAI

from molto import Molto
from molto.types.openai.responses import OpenAIResponsesRequest
from molto.utils.io import json_loads


async def test_tool_choice_required():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        body = json_loads(request.content)
        try:
            assert body["tool_choice"] == "required"
            assert len(body["messages"]) == 1
            return httpx.Response(
                200,
                json={},
            )
        except AssertionError:
            return httpx.Response(
                400,
                json={"error": "Invalid request"},
            )

    client = AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        ),
    )
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        tool_choice="required",
    )
    print(response)

    # client = Molto(
    #     http_client=httpx.AsyncClient(
    #         transport=httpx.MockTransport(handler),
    #         base_url="http://testserver",
    #     )
    # )
    # await client.chat(
    #     OpenAIResponsesRequest(
    #         model="gpt-4o",
    #         input=[{"role": "user", "content": "Hello, world!"}],
    #         tool_choice="required",
    #     )
    # )
