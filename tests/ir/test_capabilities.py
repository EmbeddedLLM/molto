import pytest

from molto.providers.capabilities import (
    ANTHROPIC_OPENAI_CHAT_CAPABILITIES,
    OPENAI_CHAT_CAPABILITIES,
    OPENAI_RESPONSES_CAPABILITIES,
    RequestFeature,
    derive_request_capabilities,
    validate_provider_request,
)
from molto.types.ir.portable import (
    FunctionToolDefinition,
    InputAudioPart,
    InputIR,
    InputTextPart,
    RequestMessageItem,
)
from molto.utils.exceptions import ModelCapabilityError


def _request(**updates) -> InputIR:
    request = InputIR(
        model="model",
        stream=False,
        items=(RequestMessageItem(role="user", content=(InputTextPart(text="hello"),)),),
    )
    return request.model_copy(update=updates)


def test_request_capabilities_cover_supported_openai_fields() -> None:
    required = derive_request_capabilities(
        _request(
            stream=True,
            tools=(FunctionToolDefinition(name="lookup", parameters={}),),
        )
    )
    assert RequestFeature.STREAM in required.features
    assert RequestFeature.TOOL_FUNCTION in required.features
    assert RequestFeature.INPUT_TEXT in required.features


def test_compatible_endpoint_contracts_are_explicit() -> None:
    assert RequestFeature.INPUT_AUDIO in OPENAI_CHAT_CAPABILITIES.features
    assert RequestFeature.INPUT_AUDIO not in OPENAI_RESPONSES_CAPABILITIES.features
    assert RequestFeature.INPUT_IMAGE not in ANTHROPIC_OPENAI_CHAT_CAPABILITIES.features


def test_provider_validation_rejects_unsupported_input_before_construction() -> None:
    request = _request(
        items=(
            RequestMessageItem(
                role="user",
                content=(InputAudioPart(data="UklGRg==", format="wav"),),
            ),
        )
    )
    with pytest.raises(ModelCapabilityError, match="input.audio"):
        validate_provider_request(ANTHROPIC_OPENAI_CHAT_CAPABILITIES, request)


def test_openai_contract_accepts_text() -> None:
    validate_provider_request(OPENAI_CHAT_CAPABILITIES, _request())
