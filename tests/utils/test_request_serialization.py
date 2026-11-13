"""Native request serialization preserves caller intent and model conventions."""

from copy import deepcopy

import pytest
from anthropic import NOT_GIVEN as ANTHROPIC_NOT_GIVEN
from anthropic import omit as anthropic_omit
from anthropic.types import TextBlock
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai import omit as openai_omit
from openai.types.shared import Reasoning

from molto.types.openai_responses import OpenAIResponsesCreateRequest
from molto.utils.io import request_to_dict


@pytest.mark.parametrize(
    "marker", [openai_omit, anthropic_omit, OPENAI_NOT_GIVEN, ANTHROPIC_NOT_GIVEN]
)
def test_native_dictionary_retains_sdk_omission_markers(marker):
    """Ensures the SDK can process its own omission markers before JSON serialization."""
    body = {"model": "test", "temperature": marker}

    assert request_to_dict(body)["temperature"] is marker


def test_molto_model_serialization_omits_nested_model_nulls():
    """Ensures Molto serialization omits null model fields, including nested SDK fields."""
    request = OpenAIResponsesCreateRequest(
        model="test", input="hello", reasoning={"effort": None}, temperature=None
    )

    assert request_to_dict(request) == {
        "model": "test",
        "input": "hello",
        "reasoning": {},
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
