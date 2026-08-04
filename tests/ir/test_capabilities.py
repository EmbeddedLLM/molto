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
    JsonSchemaOutputFormat,
    OpenAIHostedToolDefinition,
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
    assert RequestFeature.CONFIG_FREQUENCY_PENALTY in OPENAI_CHAT_CAPABILITIES.features
    assert RequestFeature.CONFIG_FREQUENCY_PENALTY not in OPENAI_RESPONSES_CAPABILITIES.features
    assert RequestFeature.CONFIG_PRESENCE_PENALTY in OPENAI_CHAT_CAPABILITIES.features
    assert RequestFeature.CONFIG_PRESENCE_PENALTY not in OPENAI_RESPONSES_CAPABILITIES.features
    assert RequestFeature.INPUT_IMAGE not in ANTHROPIC_OPENAI_CHAT_CAPABILITIES.features
    assert RequestFeature.CONFIG_STREAM_OPTIONS in ANTHROPIC_OPENAI_CHAT_CAPABILITIES.features


def test_anthropic_ignored_fields_are_not_advertised_as_capabilities() -> None:
    ignored = ANTHROPIC_OPENAI_CHAT_CAPABILITIES.ignored_features

    assert RequestFeature.OUTPUT_JSON_SCHEMA in ignored
    assert RequestFeature.TOOL_FUNCTION_STRICT in ignored
    assert RequestFeature.CONFIG_REASONING_EFFORT in ignored
    assert RequestFeature.CONFIG_STORE in ignored
    assert RequestFeature.OUTPUT_JSON_SCHEMA not in ANTHROPIC_OPENAI_CHAT_CAPABILITIES.features
    assert RequestFeature.TOOL_FUNCTION_STRICT not in ANTHROPIC_OPENAI_CHAT_CAPABILITIES.features
    assert (
        RequestFeature.CONFIG_REASONING_EFFORT not in ANTHROPIC_OPENAI_CHAT_CAPABILITIES.features
    )

    validate_provider_request(
        ANTHROPIC_OPENAI_CHAT_CAPABILITIES,
        _request(
            output_format=JsonSchemaOutputFormat(name="result", schema={}),
            tools=(FunctionToolDefinition(name="lookup", parameters={}, strict=True),),
            store=False,
            generation=_request().generation.model_copy(update={"reasoning_effort": "high"}),
        ),
    )


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
        validate_provider_request(ANTHROPIC_OPENAI_CHAT_CAPABILITIES, request, strict=True)


def test_non_strict_validation_drops_unsupported_features() -> None:
    request = _request(
        items=(
            RequestMessageItem(
                role="user",
                content=(InputAudioPart(data="UklGRg==", format="wav"),),
            ),
        )
    )

    required = validate_provider_request(ANTHROPIC_OPENAI_CHAT_CAPABILITIES, request)

    assert RequestFeature.INPUT_AUDIO in required.features


def test_strict_validation_rejects_ignored_features() -> None:
    request = _request(
        output_format=JsonSchemaOutputFormat(name="result", schema={}),
        generation=_request().generation.model_copy(update={"reasoning_effort": "high"}),
    )

    validate_provider_request(ANTHROPIC_OPENAI_CHAT_CAPABILITIES, request)
    with pytest.raises(ModelCapabilityError, match="config.reasoning_effort"):
        validate_provider_request(ANTHROPIC_OPENAI_CHAT_CAPABILITIES, request, strict=True)


def test_non_strict_validation_accepts_ignored_features() -> None:
    request = _request(
        output_format=JsonSchemaOutputFormat(name="result", schema={}),
        generation=_request().generation.model_copy(update={"reasoning_effort": "high"}),
    )

    required = validate_provider_request(ANTHROPIC_OPENAI_CHAT_CAPABILITIES, request)

    assert RequestFeature.OUTPUT_JSON_SCHEMA in required.features
    assert RequestFeature.CONFIG_REASONING_EFFORT in required.features


def test_openai_contract_accepts_text() -> None:
    validate_provider_request(OPENAI_CHAT_CAPABILITIES, _request())


def test_namespace_tool_requires_hosted_tool_capability() -> None:
    request = _request(
        tools=(
            OpenAIHostedToolDefinition(
                tool_type="namespace",
                payload_json='{"type":"namespace","name":"crm","tools":[]}',
            ),
        ),
    )

    required = derive_request_capabilities(request)
    assert RequestFeature.TOOL_HOSTED in required.features

    validate_provider_request(OPENAI_RESPONSES_CAPABILITIES, request)
    with pytest.raises(ModelCapabilityError, match="tool.hosted"):
        validate_provider_request(OPENAI_CHAT_CAPABILITIES, request, strict=True)


def test_hosted_web_search_uses_portable_web_search_capability() -> None:
    request = _request(
        tools=(
            OpenAIHostedToolDefinition(
                tool_type="web_search",
                payload_json='{"type":"web_search"}',
            ),
        ),
    )

    required = derive_request_capabilities(request)

    assert RequestFeature.TOOL_WEB_SEARCH in required.features
    assert RequestFeature.TOOL_HOSTED not in required.features
    # The Anthropic compatibility endpoint validates tools as function-only and
    # has no web search field, so web search is rejected for Anthropic.
    validate_provider_request(ANTHROPIC_OPENAI_CHAT_CAPABILITIES, request)
    with pytest.raises(ModelCapabilityError, match="tool.web_search"):
        validate_provider_request(ANTHROPIC_OPENAI_CHAT_CAPABILITIES, request, strict=True)
