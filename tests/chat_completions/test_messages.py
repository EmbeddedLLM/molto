from base64 import b64encode
from pathlib import Path

import pytest

from molto.types.ir.portable import InputImagePart, InputTextPart, RequestMessageItem
from molto.types.openai import OpenAIRequest
from tests.support import (
    STREAM_PARAMS,
    make_molto_client,
    prepend_model_prefix,
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
    request = OpenAIRequest.chat(
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

    cross_result = await make_molto_client(cassette_manager).responses(
        OpenAIRequest.chat(**prepend_model_prefix(req_body))
    )
    if stream:
        cross_events = [event async for event in cross_result]
        assert cross_events[-1].type == "response.completed"
    else:
        assert cross_result.object == "response"
