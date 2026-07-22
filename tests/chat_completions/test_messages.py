from base64 import b64encode
from pathlib import Path

import pytest

from molto.types.ir import ImagePart, TextPart, UserMessage
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.utils.test import (
    STREAM_PARAMS,
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    stream_reference_cassette_suffix,
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

    assert request.to_ir().input == [
        UserMessage(content="Describe this image."),
        UserMessage(
            content=[
                TextPart(text="What is shown?"),
                ImagePart(url="https://example.com/image.png", detail="high"),
            ]
        ),
    ]


def _stream_text(chunks) -> str:
    return "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)


def _assert_usage_matches(sdk_usage, molto_usage) -> None:
    assert sdk_usage is not None
    assert molto_usage is not None
    assert molto_usage.model_dump() == sdk_usage.model_dump()


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_multimodal_messages_round_trip_through_sdk(stream: bool, cassette_manager) -> None:
    req_body = {**MULTIMODAL_REQUEST, "stream": stream}
    if stream:
        req_body["stream_options"] = {"include_usage": True}
    assert_attrs = ["model", "messages"]
    sdk_client = make_openai_client(
        cassette_manager,
        cassette_endpoint="chat_completions",
        expected_request_body=req_body,
        assert_attrs=assert_attrs,
    )
    sdk_response = await sdk_client.chat.completions.create(**req_body)
    if stream:
        sdk_chunks = [chunk async for chunk in sdk_response]
        assert sdk_chunks[-1].usage is not None
        sdk_text = _stream_text(sdk_chunks)
        stream_reference = sdk_chunks
    else:
        assert sdk_response.usage is not None
        sdk_text = sdk_response.choices[0].message.content
        assert sdk_text is not None
        stream_req_body = {
            **MULTIMODAL_REQUEST,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        reference_client = make_openai_client(
            cassette_manager,
            cassette_suffix=stream_reference_cassette_suffix(stream),
            cassette_endpoint="chat_completions",
            expected_request_body=stream_req_body,
            assert_attrs=assert_attrs,
        )
        stream_reference = [
            chunk
            async for chunk in await reference_client.chat.completions.create(**stream_req_body)
        ]

    stream_req_body = {
        **MULTIMODAL_REQUEST,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    molto = make_molto_client(
        cassette_manager,
        cassette_suffix=stream_reference_cassette_suffix(stream),
        expected_request_body=stream_req_body,
        assert_attrs=assert_attrs,
    )
    molto_response = await molto.chat(
        OpenAIChatCompletionRequest(**prepend_model_prefix(req_body))
    )
    if stream:
        molto_chunks = [chunk async for chunk in molto_response]
        assert _stream_text(molto_chunks) == sdk_text
        assert molto_chunks[-1].id == stream_reference[-1].id
        _assert_usage_matches(stream_reference[-1].usage, molto_chunks[-1].usage)
    else:
        assert molto_response.choices[0].message.content == _stream_text(stream_reference)
        assert molto_response.id == stream_reference[-1].id
        _assert_usage_matches(stream_reference[-1].usage, molto_response.usage)
