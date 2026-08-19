import pytest

from molto.adapters.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient

_CAPABILITIES = PROVIDER_CAPABILITIES["openai"]


def test_adapters_preserve_multimodal_input_boundaries_and_order():
    responses = OpenAIResponsesAdapter(_CAPABILITIES).normalize_request(
        OpenAIResponsesCreateRequest(
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "before"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                        {"type": "input_text", "text": "after"},
                    ],
                }
            ],
            model="m",
        )
    )
    chat = OpenAIChatCompletionsAdapter(_CAPABILITIES).normalize_request(
        OpenAIChatCompletionsCreateRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AA=="},
                        },
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
            model="m",
        )
    )

    assert responses.input == chat.input
    assert OpenAIResponsesClient._process_input(chat) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "before"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                {"type": "input_text", "text": "after"},
            ],
        }
    ]
    assert OpenAIChatCompletionsClient._messages(responses) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
                {"type": "text", "text": "after"},
            ],
        }
    ]


def test_responses_adapter_rejects_image_file_id():
    with pytest.raises(ValueError, match="requires image_url"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "role": "user",
                "content": [{"type": "input_image", "file_id": "file_123"}],
            }
        )


def test_responses_adapter_rejects_input_file():
    with pytest.raises(ValueError, match="Unsupported Responses user content type: input_file"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": "file_123"}],
            }
        )


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_text", "text": "hello", "prompt_cache_breakpoint": {"mode": "explicit"}},
        {
            "type": "input_image",
            "image_url": "https://example.com/a.png",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
    ],
    ids=["text", "image"],
)
def test_responses_adapter_rejects_prompt_cache_breakpoint(part):
    with pytest.raises(ValueError, match="not supported for translation"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {"role": "user", "content": [part]}
        )


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}],
        [{"type": "file", "file": {"file_id": "file_123"}}],
    ],
    ids=["audio", "file"],
)
def test_chat_adapter_rejects_unsupported_user_content(content):
    adapter = OpenAIChatCompletionsAdapter(_CAPABILITIES)
    with pytest.raises(ValueError, match="Unsupported Chat user content type"):
        adapter._normalize_message({"role": "user", "content": content})


@pytest.mark.parametrize(
    ("role", "part_type", "expected_type"),
    [
        ("developer", "input_text", "system-prompt"),
        ("assistant", "output_text", "text-output"),
    ],
)
def test_responses_adapter_validates_content_parts_by_message_role(role, part_type, expected_type):
    [item] = OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
        {"type": "message", "role": role, "content": [{"type": part_type, "text": "x"}]}
    )

    assert item.content_type == expected_type


def test_responses_adapter_rejects_wrong_content_part_for_message_role():
    with pytest.raises(ValueError, match="assistant message content cannot be translated"):
        OpenAIResponsesAdapter(_CAPABILITIES)._normalize_message(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "x"}],
            }
        )
