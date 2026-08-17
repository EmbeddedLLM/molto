from pydantic import TypeAdapter

from molto.types.normalized import InputItem, MultimodalInput


def test_multimodal_input_preserves_ordered_content():
    message = TypeAdapter(InputItem).validate_python(
        {
            "content_type": "multimodal-input",
            "content": [
                {"content_type": "text-input", "content": "before"},
                {
                    "content_type": "image-url-input",
                    "url": "data:image/png;base64,AA==",
                    "detail": "high",
                },
                {"content_type": "text-input", "content": "after"},
            ],
        }
    )

    assert isinstance(message, MultimodalInput)
    assert [part.content_type for part in message.content] == [
        "text-input",
        "image-url-input",
        "text-input",
    ]
    assert message.content[0].content == "before"
    assert message.content[1].url == "data:image/png;base64,AA=="
    assert message.content[1].detail == "high"
    assert message.content[2].content == "after"
