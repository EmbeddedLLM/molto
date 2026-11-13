"""Native request serialization preserves caller intent and model conventions."""

from copy import deepcopy
from enum import Enum
from typing import Any

import pytest
from anthropic import NOT_GIVEN as ANTHROPIC_NOT_GIVEN
from anthropic import omit as anthropic_omit
from anthropic.types import TextBlock
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai import omit as openai_omit
from openai.types.shared import Reasoning
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_serializer

from molto.types.anthropic_messages import AnthropicMessagesCreateRequest
from molto.types.openai_chat_completions import OpenAIChatCompletionsCreateRequest
from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.utils.io import request_to_dict


class _Tag(Enum):
    VALUE = "value"


@pytest.mark.parametrize(
    ("request_type", "body"),
    [
        (
            OpenAIResponsesCreateRequest,
            {
                "model": "test",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello", "part_extra": None}],
                        "message_extra": None,
                    }
                ],
            },
        ),
        (
            OpenAIChatCompletionsCreateRequest,
            {
                "model": "test",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello", "part_extra": None}],
                        "message_extra": None,
                    }
                ],
            },
        ),
        (
            AnthropicMessagesCreateRequest,
            {
                "model": "test",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello", "part_extra": None}],
                        "message_extra": None,
                    }
                ],
            },
        ),
    ],
)
def test_molto_models_keep_extensions_and_omit_declared_nulls(request_type, body):
    """Ensures allowed nested extras survive while declared None fields remain omitted."""
    body = {**body, "temperature": None, "request_extra": None, "enum_extra": _Tag.VALUE}
    request = request_type.model_validate(body, extra="allow")
    original = request.model_copy(deep=True)
    expected = {key: value for key, value in body.items() if key != "temperature"}
    expected["enum_extra"] = "value"

    assert request_to_dict(request) == expected
    assert request == original
    assert request_to_dict(request) == expected


def test_caller_model_keeps_defaults_aliases_and_json_values():
    """Ensures model defaults remain useful and aliases/enums use JSON representation."""

    class Request(BaseModel):
        model: str = "test"
        messages: list[dict[str, str]] = [{"role": "user", "content": "hello"}]
        system: str | None = None
        temperature: float = 0.7
        service_tier_: str = Field(default="standard", alias="service_tier")
        metadata: dict[str, Any] = {"tag": _Tag.VALUE}

    request = Request()
    assert not request.model_fields_set
    assert request_to_dict(request) == {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "service_tier": "standard",
        "metadata": {"tag": "value"},
    }


def test_caller_model_uses_its_field_serializer():
    """Ensures native conversion honors a caller's deliberate wire representation."""

    class Request(BaseModel):
        model: str = "test"
        metadata: dict[str, int] = {"count": 3}

        @field_serializer("metadata")
        def serialize_metadata(self, value: dict[str, int]) -> dict[str, str]:
            return {key: str(item) for key, item in value.items()}

    assert request_to_dict(Request()) == {"model": "test", "metadata": {"count": "3"}}


@pytest.mark.parametrize("extra", ["allow", "forbid"])
def test_caller_model_keeps_allowed_root_extras(extra: str):
    """Ensures root extensions survive per-call allowance, including nulls and JSON scalars."""

    class Request(BaseModel):
        model_config = ConfigDict(extra=extra)
        model: str = "test"
        system: str | None = None

    request = Request.model_validate(
        {"provider_null": None, "provider_value": _Tag.VALUE}, extra="allow"
    )
    assert request_to_dict(request) == {
        "model": "test",
        "provider_null": None,
        "provider_value": "value",
    }


def test_caller_model_serializer_owns_serialized_extras():
    """Ensures preserving extras does not overwrite a caller's custom serialization."""

    class Request(BaseModel):
        model_config = ConfigDict(extra="allow")
        model: str = "test"

        @model_serializer(mode="wrap")
        def serialize_request(self, handler):
            serialized = handler(self)
            serialized["provider_option"] = "serialized"
            return serialized

    assert request_to_dict(Request(provider_option="raw", provider_null=None)) == {
        "model": "test",
        "provider_option": "serialized",
        "provider_null": None,
    }


@pytest.mark.parametrize(
    "marker", [openai_omit, anthropic_omit, OPENAI_NOT_GIVEN, ANTHROPIC_NOT_GIVEN]
)
def test_native_dictionary_retains_sdk_omission_markers(marker):
    """Ensures the SDK can process its own omission markers before JSON serialization."""
    body = {"model": "test", "temperature": marker}

    assert request_to_dict(body)["temperature"] is marker


def test_molto_model_retains_explicit_nulls_inside_dictionaries():
    """Ensures dictionary values keep nulls while model fields retain their omission policy."""
    request = OpenAIResponsesCreateRequest(
        model="test", input="hello", reasoning={"effort": None}, temperature=None
    )

    assert request_to_dict(request) == {
        "model": "test",
        "input": "hello",
        "reasoning": {"effort": None},
    }


def test_native_dictionary_retains_nulls_and_sdk_omission():
    """Ensures raw dictionaries keep nulls while nested SDK models retain their serializers."""
    body = {
        "temperature": None,
        "provider_option": {"nested": None},
        "reasoning": Reasoning(effort="low"),
        "content": [TextBlock(type="text", text="hello")],
    }
    original = deepcopy(body)
    assert request_to_dict(body) == {
        "temperature": None,
        "provider_option": {"nested": None},
        "reasoning": {"effort": "low"},
        "content": [{"type": "text", "text": "hello"}],
    }
    assert body == original
