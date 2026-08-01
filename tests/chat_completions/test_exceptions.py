import pytest
from openai import AuthenticationError

from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from molto.utils.exceptions import ExternalAuthError
from tests.support import (
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
)

INVALID_API_KEY_REQUEST = {
    "model": "gpt-4.1-nano",
    "messages": [{"role": "user", "content": "What number am I thinking of?"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "guess_number",
                "description": "Guess a number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "description": "The guessed number.",
                        }
                    },
                    "required": ["number"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "required",
}


async def test_invalid_api_key(cassette_manager) -> None:
    """Preserve parsed Chat provider diagnostics through the public Molto route."""
    sdk_client = make_openai_client(
        cassette_manager,
        cassette_endpoint="chat_completions",
        api_key="test",
    )
    with pytest.raises(AuthenticationError) as sdk_exc_info:
        await sdk_client.chat.completions.create(**INVALID_API_KEY_REQUEST)

    sdk_exc = sdk_exc_info.value
    assert isinstance(sdk_exc.body, dict)
    sdk_error = sdk_exc.body.get("error", sdk_exc.body)
    assert isinstance(sdk_error, dict)
    assert sdk_exc.status_code == 401
    assert sdk_error["type"] == "invalid_request_error"
    assert sdk_error["code"] == "invalid_api_key"
    assert "param" in sdk_error
    assert sdk_error["param"] is None

    molto = make_molto_client(cassette_manager)
    request = OpenAIChatCompletionRequest(**prepend_model_prefix(INVALID_API_KEY_REQUEST))
    with pytest.raises(ExternalAuthError) as molto_exc_info:
        await molto.chat_completions(request, api_key="test")

    molto_exc = molto_exc_info.value
    assert str(molto_exc) == sdk_error["message"]
    assert molto_exc.status_code == sdk_exc.status_code
    assert molto_exc.request_id == sdk_exc.request_id
    assert molto_exc.provider_body == sdk_exc.body
    assert molto_exc.provider_code == sdk_error["code"]
