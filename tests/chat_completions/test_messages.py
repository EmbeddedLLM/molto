from base64 import b64encode
from pathlib import Path

import pytest

from molto.types.ir.portable import InputImagePart, InputTextPart, RequestMessageItem
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.types.openai.responses import OpenAIResponsesRequest
from tests.support import (
    STREAM_PARAMS,
    make_molto_client,
    run_chat_parity,
    stream_chat_text,
)

XLSX_IMAGE_URL = "data:image/jpeg;base64," + b64encode(
    (Path(__file__).parents[1] / "files" / "xlsx.jpg").read_bytes()
).decode("ascii")

MULTIMODAL_REQUEST = {
    "model": "gpt-4.1-nano",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image_url", "image_url": {"url": XLSX_IMAGE_URL, "detail": "auto"}},
            ],
        }
    ],
}


def test_text_and_multimodal_messages_map_to_ir() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[
            {"role": "user", "content": "Describe this image."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png", "detail": "high"},
                    },
                ],
            },
        ],
    )

    assert request.to_ir().items == (
        RequestMessageItem(
            role="user",
            content=(InputTextPart(text="Describe this image."),),
        ),
        RequestMessageItem(
            role="user",
            content=(
                InputTextPart(text="What is shown?"),
                InputImagePart(
                    image_url="https://example.com/image.png",
                    detail="high",
                ),
            ),
        ),
    )


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_multimodal_messages_round_trip_through_sdk(stream: bool, cassette_manager) -> None:
    req_body = {**MULTIMODAL_REQUEST, "stream": stream}
    if stream:
        req_body["stream_options"] = {"include_usage": True}
    parity = await run_chat_parity(cassette_manager, req_body)
    if stream:
        assert parity.sdk_chunks[-1].usage is not None
        sdk_text = stream_chat_text(parity.sdk_chunks)
        assert stream_chat_text(parity.molto_chunks) == sdk_text
    else:
        assert parity.sdk_response.usage is not None
        sdk_text = parity.sdk_response.choices[0].message.content
        assert sdk_text is not None
        assert parity.molto_response.choices[0].message.content == sdk_text

    if not stream:
        cross_result = await make_molto_client(cassette_manager).responses(
            OpenAIResponsesRequest(
                model=f"openai/{req_body['model']}",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What is shown in this image?"},
                            {
                                "type": "input_image",
                                "image_url": XLSX_IMAGE_URL,
                                "detail": "auto",
                            },
                        ],
                    }
                ],
                stream=False,
                upstream="chat_completions",
            ),
        )
        assert cross_result.object == "response"
