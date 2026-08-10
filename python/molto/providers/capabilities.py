from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeVar

from molto.types.ir.aggregates import InputIR
from molto.types.ir.base import IRModel, ProviderName
from molto.types.ir.config import JsonObjectOutputFormat, JsonSchemaOutputFormat
from molto.types.ir.content import InputAudioPart, InputImagePart, InputTextPart
from molto.types.ir.request_items import (
    CodeInterpreterCallRequestItem,
    FunctionCallRequestItem,
    FunctionCallResultItem,
    ImageGenerationCallRequestItem,
    ReasoningRequestItem,
    RequestMessageItem,
    WebSearchCallRequestItem,
)
from molto.types.ir.tools import (
    OPENAI_WEB_SEARCH_TOOL_TYPES,
    AllowedFunctionTools,
    FunctionToolChoice,
    OpenAIHostedToolChoice,
    OpenAIHostedToolDefinition,
)
from molto.utils.exceptions import ModelCapabilityError


class RequestFeature(str, Enum):
    """Stable capabilities required by an OpenAI-compatible request."""

    STREAM = "stream"
    INPUT_TEXT = "input.text"
    INPUT_IMAGE = "input.image"
    INPUT_AUDIO = "input.audio"
    INPUT_FUNCTION_CALL = "input.function_call"
    INPUT_FUNCTION_RESULT = "input.function_result"
    INPUT_RESPONSE_HISTORY = "input.response_history"
    OUTPUT_TEXT = "output.text"
    OUTPUT_AUDIO = "output.audio"
    OUTPUT_JSON_OBJECT = "output.json_object"
    OUTPUT_JSON_SCHEMA = "output.json_schema"
    CONFIG_MAX_OUTPUT_TOKENS = "config.max_output_tokens"
    CONFIG_TEMPERATURE = "config.temperature"
    CONFIG_TOP_P = "config.top_p"
    CONFIG_FREQUENCY_PENALTY = "config.frequency_penalty"
    CONFIG_PRESENCE_PENALTY = "config.presence_penalty"
    CONFIG_TOP_LOGPROBS = "config.top_logprobs"
    CONFIG_TEXT_VERBOSITY = "config.text_verbosity"
    CONFIG_REASONING_EFFORT = "config.reasoning_effort"
    CONFIG_REASONING_SUMMARY = "config.reasoning_summary"
    CONFIG_PARALLEL_TOOL_CALLS = "config.parallel_tool_calls"
    CONFIG_MAX_TOOL_CALLS = "config.max_tool_calls"
    CONFIG_MODALITIES = "config.modalities"
    CONFIG_STREAM_OPTIONS = "config.stream_options"
    CONFIG_STREAM_INCLUDE_USAGE = "config.stream_options.include_usage"
    CONFIG_INCLUDE = "config.include"
    CONFIG_STORE = "config.store"
    TOOL_FUNCTION = "tool.function"
    TOOL_WEB_SEARCH = "tool.web_search"
    TOOL_HOSTED = "tool.hosted"
    TOOL_FUNCTION_STRICT = "tool.function.strict"
    TOOL_CHOICE_NONE = "tool.choice.none"
    TOOL_CHOICE_AUTO = "tool.choice.auto"
    TOOL_CHOICE_REQUIRED = "tool.choice.required"
    TOOL_CHOICE_FUNCTION = "tool.choice.function"
    TOOL_CHOICE_ALLOWED = "tool.choice.allowed"
    TOOL_CHOICE_ALLOWED_MULTIPLE = "tool.choice.allowed_multiple"
    STATE_RESPONSE = "state.response"
    STATE_CONVERSATION = "state.conversation"
    STATE_COMPACTION = "state.compaction"
    STATE_CONTINUATION = "state.continuation"


_CONFIG_FEATURES = {
    "max_output_tokens": RequestFeature.CONFIG_MAX_OUTPUT_TOKENS,
    "temperature": RequestFeature.CONFIG_TEMPERATURE,
    "top_p": RequestFeature.CONFIG_TOP_P,
    "frequency_penalty": RequestFeature.CONFIG_FREQUENCY_PENALTY,
    "presence_penalty": RequestFeature.CONFIG_PRESENCE_PENALTY,
    "top_logprobs": RequestFeature.CONFIG_TOP_LOGPROBS,
    "text_verbosity": RequestFeature.CONFIG_TEXT_VERBOSITY,
    "reasoning_effort": RequestFeature.CONFIG_REASONING_EFFORT,
    "reasoning_summary": RequestFeature.CONFIG_REASONING_SUMMARY,
    "parallel_tool_calls": RequestFeature.CONFIG_PARALLEL_TOOL_CALLS,
    "max_tool_calls": RequestFeature.CONFIG_MAX_TOOL_CALLS,
}
_STATE_FEATURES = {
    "response": RequestFeature.STATE_RESPONSE,
    "conversation": RequestFeature.STATE_CONVERSATION,
    "compaction": RequestFeature.STATE_COMPACTION,
    "continuation": RequestFeature.STATE_CONTINUATION,
}


class RequestRequirements(IRModel):
    features: frozenset[RequestFeature]
    state_provider: ProviderName | None = None


class ProviderContract(IRModel):
    """One endpoint's effective and explicitly ignored request features."""

    provider: ProviderName
    endpoint: str
    features: frozenset[RequestFeature]
    ignored_features: frozenset[RequestFeature] = frozenset()


def derive_request_capabilities(request: InputIR) -> RequestRequirements:
    """Derive execution-relevant features from the portable request."""

    features: set[RequestFeature] = {RequestFeature.OUTPUT_TEXT}
    if request.stream:
        features.add(RequestFeature.STREAM)
    if request.stream_options is not None:
        features.add(RequestFeature.CONFIG_STREAM_OPTIONS)
        if request.stream_options.include_usage is not None:
            features.add(RequestFeature.CONFIG_STREAM_INCLUDE_USAGE)
    if request.include is not None:
        features.add(RequestFeature.CONFIG_INCLUDE)
    if request.store is not None:
        features.add(RequestFeature.CONFIG_STORE)
    if isinstance(request.output_format, JsonObjectOutputFormat):
        features.remove(RequestFeature.OUTPUT_TEXT)
        features.add(RequestFeature.OUTPUT_JSON_OBJECT)
    elif isinstance(request.output_format, JsonSchemaOutputFormat):
        features.remove(RequestFeature.OUTPUT_TEXT)
        features.add(RequestFeature.OUTPUT_JSON_SCHEMA)
    if (
        request.generation.audio_output is not None
        or request.generation.modalities is not None
        and "audio" in request.generation.modalities
    ):
        features.add(RequestFeature.OUTPUT_AUDIO)
    if request.generation.modalities is not None:
        features.add(RequestFeature.CONFIG_MODALITIES)
    if request.generation.web_search_options is not None:
        features.add(RequestFeature.TOOL_WEB_SEARCH)

    for item in request.items:
        if isinstance(item, RequestMessageItem):
            if item.input_form == "responses_output_item":
                features.add(RequestFeature.INPUT_RESPONSE_HISTORY)
            if item.audio_id is not None:
                features.add(RequestFeature.INPUT_AUDIO)
            for part in item.content:
                if isinstance(part, InputTextPart):
                    features.add(RequestFeature.INPUT_TEXT)
                elif isinstance(part, InputImagePart):
                    features.add(RequestFeature.INPUT_IMAGE)
                elif isinstance(part, InputAudioPart):
                    features.add(RequestFeature.INPUT_AUDIO)
        elif isinstance(item, FunctionCallRequestItem):
            features.add(RequestFeature.INPUT_FUNCTION_CALL)
            if item.id is not None or item.status is not None:
                features.add(RequestFeature.INPUT_RESPONSE_HISTORY)
        elif isinstance(item, FunctionCallResultItem):
            features.add(RequestFeature.INPUT_FUNCTION_RESULT)
            if item.id is not None:
                features.add(RequestFeature.INPUT_RESPONSE_HISTORY)
        elif isinstance(
            item,
            (
                ReasoningRequestItem,
                WebSearchCallRequestItem,
                CodeInterpreterCallRequestItem,
                ImageGenerationCallRequestItem,
            ),
        ):
            features.add(RequestFeature.INPUT_RESPONSE_HISTORY)

    for field, feature in _CONFIG_FEATURES.items():
        value = getattr(request.generation, field)
        if value is None:
            continue
        if field in {"frequency_penalty", "presence_penalty"} and value == 0:
            continue
        features.add(feature)
    for tool in request.tools:
        if isinstance(tool, OpenAIHostedToolDefinition):
            if tool.tool_type in OPENAI_WEB_SEARCH_TOOL_TYPES:
                features.add(RequestFeature.TOOL_WEB_SEARCH)
            else:
                features.add(RequestFeature.TOOL_HOSTED)
            continue
        features.add(RequestFeature.TOOL_FUNCTION)
        if tool.strict is not None:
            features.add(RequestFeature.TOOL_FUNCTION_STRICT)
    if request.tool_choice == "none":
        features.add(RequestFeature.TOOL_CHOICE_NONE)
    elif request.tool_choice == "auto":
        features.add(RequestFeature.TOOL_CHOICE_AUTO)
    elif request.tool_choice == "required":
        features.add(RequestFeature.TOOL_CHOICE_REQUIRED)
    elif isinstance(request.tool_choice, FunctionToolChoice):
        features.add(RequestFeature.TOOL_CHOICE_FUNCTION)
    elif isinstance(request.tool_choice, AllowedFunctionTools):
        features.add(RequestFeature.TOOL_CHOICE_ALLOWED)
        if len(request.tool_choice.tools) > 1:
            features.add(RequestFeature.TOOL_CHOICE_ALLOWED_MULTIPLE)
    elif isinstance(request.tool_choice, OpenAIHostedToolChoice):
        features.add(RequestFeature.TOOL_HOSTED)

    state_provider = None
    if request.state is not None:
        state_provider = request.state.provider
        features.update(_STATE_FEATURES[reference.kind] for reference in request.state.values)
    return RequestRequirements(features=frozenset(features), state_provider=state_provider)


def validate_provider_request(
    provider_contract: ProviderContract,
    request: InputIR,
    *,
    strict: bool = False,
) -> RequestRequirements:
    """Reject unsupported requests before provider construction.

    With ``strict=True`` every request feature the contract does not fully
    honor raises, including the ``ignored_features`` carve-out. Without it,
    unsupported features are dropped and ignored so the request proceeds with
    the supported subset.
    """

    required = derive_request_capabilities(request)
    if (
        required.state_provider is not None
        and required.state_provider != provider_contract.provider
    ):
        raise ModelCapabilityError(
            f"State owned by provider {required.state_provider!r} cannot be used with "
            f"provider {provider_contract.provider!r}."
        )
    if strict:
        missing_features = required.features - provider_contract.features
        if missing_features:
            details = ", ".join(
                feature.value
                for feature in sorted(missing_features, key=lambda value: value.value)
            )
            raise ModelCapabilityError(
                f"Provider {provider_contract.provider!r} endpoint {provider_contract.endpoint!r} does not support: "
                f"{details}."
            )
    return required


ProviderT = TypeVar("ProviderT")


def create_provider_after_validation(
    provider_contract: ProviderContract,
    request: InputIR,
    factory: Callable[[], ProviderT],
    *,
    strict: bool = False,
) -> ProviderT:
    """Construct a provider only after capability validation succeeds."""

    validate_provider_request(provider_contract, request, strict=strict)
    return factory()


_COMMON_TEXT_FEATURES = frozenset(
    {
        RequestFeature.STREAM,
        RequestFeature.INPUT_TEXT,
        RequestFeature.INPUT_FUNCTION_CALL,
        RequestFeature.INPUT_FUNCTION_RESULT,
        RequestFeature.OUTPUT_TEXT,
        RequestFeature.OUTPUT_JSON_OBJECT,
        RequestFeature.OUTPUT_JSON_SCHEMA,
        RequestFeature.CONFIG_MAX_OUTPUT_TOKENS,
        RequestFeature.CONFIG_TEMPERATURE,
        RequestFeature.CONFIG_TOP_P,
        RequestFeature.CONFIG_REASONING_EFFORT,
        RequestFeature.CONFIG_PARALLEL_TOOL_CALLS,
        RequestFeature.TOOL_FUNCTION,
        RequestFeature.TOOL_FUNCTION_STRICT,
        RequestFeature.TOOL_CHOICE_NONE,
        RequestFeature.TOOL_CHOICE_AUTO,
        RequestFeature.TOOL_CHOICE_REQUIRED,
        RequestFeature.TOOL_CHOICE_FUNCTION,
        RequestFeature.TOOL_CHOICE_ALLOWED,
        RequestFeature.TOOL_CHOICE_ALLOWED_MULTIPLE,
    }
)

OPENAI_CHAT_CAPABILITIES = ProviderContract(
    provider="openai",
    endpoint="chat/completions",
    features=_COMMON_TEXT_FEATURES
    | {
        RequestFeature.INPUT_IMAGE,
        RequestFeature.INPUT_AUDIO,
        RequestFeature.OUTPUT_AUDIO,
        RequestFeature.CONFIG_FREQUENCY_PENALTY,
        RequestFeature.CONFIG_PRESENCE_PENALTY,
        RequestFeature.CONFIG_TOP_LOGPROBS,
        RequestFeature.CONFIG_MODALITIES,
        RequestFeature.CONFIG_STREAM_OPTIONS,
        RequestFeature.CONFIG_STREAM_INCLUDE_USAGE,
        RequestFeature.CONFIG_STORE,
        RequestFeature.TOOL_WEB_SEARCH,
    },
)

OPENAI_RESPONSES_CAPABILITIES = ProviderContract(
    provider="openai",
    endpoint="responses",
    features=_COMMON_TEXT_FEATURES
    | {
        RequestFeature.INPUT_IMAGE,
        RequestFeature.CONFIG_TOP_LOGPROBS,
        RequestFeature.CONFIG_TEXT_VERBOSITY,
        RequestFeature.CONFIG_REASONING_SUMMARY,
        RequestFeature.CONFIG_MAX_TOOL_CALLS,
        RequestFeature.CONFIG_INCLUDE,
        RequestFeature.CONFIG_STORE,
        RequestFeature.INPUT_RESPONSE_HISTORY,
        RequestFeature.TOOL_HOSTED,
        RequestFeature.STATE_RESPONSE,
        RequestFeature.STATE_CONVERSATION,
        RequestFeature.CONFIG_STREAM_OPTIONS,
        RequestFeature.CONFIG_STREAM_INCLUDE_USAGE,
    },
)

VLLM_CHAT_CAPABILITIES = ProviderContract(
    provider="vllm",
    endpoint="chat/completions",
    features=OPENAI_CHAT_CAPABILITIES.features
    - {
        RequestFeature.INPUT_AUDIO,
        RequestFeature.OUTPUT_AUDIO,
        RequestFeature.TOOL_WEB_SEARCH,
    },
)
VLLM_RESPONSES_CAPABILITIES = ProviderContract(
    provider="vllm",
    endpoint="responses",
    features=OPENAI_RESPONSES_CAPABILITIES.features,
)

# Anthropic models are addressed through their OpenAI-compatible Chat endpoint;
# no Anthropic-native request or event translator is part of the contract. Keep
# fields that the compatibility endpoint accepts but ignores out of `features`
# so the contract does not promise semantics that Anthropic does not provide.
# Hosted tools (including web search) are rejected: the compatibility endpoint
# validates `tools[].type` as function-only and has no web search field.
ANTHROPIC_OPENAI_CHAT_CAPABILITIES = ProviderContract(
    provider="anthropic",
    endpoint="chat/completions",
    features=frozenset(
        {
            RequestFeature.STREAM,
            RequestFeature.INPUT_TEXT,
            RequestFeature.INPUT_FUNCTION_CALL,
            RequestFeature.INPUT_FUNCTION_RESULT,
            RequestFeature.OUTPUT_TEXT,
            RequestFeature.CONFIG_MAX_OUTPUT_TOKENS,
            RequestFeature.CONFIG_TEMPERATURE,
            RequestFeature.CONFIG_TOP_P,
            RequestFeature.CONFIG_PARALLEL_TOOL_CALLS,
            RequestFeature.CONFIG_STREAM_OPTIONS,
            RequestFeature.CONFIG_STREAM_INCLUDE_USAGE,
            RequestFeature.TOOL_FUNCTION,
            RequestFeature.TOOL_CHOICE_NONE,
            RequestFeature.TOOL_CHOICE_AUTO,
            RequestFeature.TOOL_CHOICE_REQUIRED,
            RequestFeature.TOOL_CHOICE_FUNCTION,
        }
    ),
    ignored_features=frozenset(
        {
            RequestFeature.OUTPUT_JSON_OBJECT,
            RequestFeature.OUTPUT_JSON_SCHEMA,
            RequestFeature.CONFIG_FREQUENCY_PENALTY,
            RequestFeature.CONFIG_PRESENCE_PENALTY,
            RequestFeature.CONFIG_TOP_LOGPROBS,
            RequestFeature.CONFIG_REASONING_EFFORT,
            RequestFeature.CONFIG_STORE,
            RequestFeature.TOOL_FUNCTION_STRICT,
        }
    ),
)
ANTHROPIC_CAPABILITIES = ANTHROPIC_OPENAI_CHAT_CAPABILITIES


def provider_contract_for(model: str, endpoint: str) -> ProviderContract:
    """Select the OpenAI-compatible contract for a model prefix and endpoint."""

    if model.startswith("openai/"):
        provider_contracts = {
            "chat/completions": OPENAI_CHAT_CAPABILITIES,
            "responses": OPENAI_RESPONSES_CAPABILITIES,
        }
    elif model.startswith("vllm/"):
        provider_contracts = {
            "chat/completions": VLLM_CHAT_CAPABILITIES,
            "responses": VLLM_RESPONSES_CAPABILITIES,
        }
    elif model.startswith("anthropic/"):
        provider_contracts = {"chat/completions": ANTHROPIC_OPENAI_CHAT_CAPABILITIES}
    else:
        raise ValueError(f"Unsupported model: {model}")
    try:
        return provider_contracts[endpoint]
    except KeyError as exc:
        raise ValueError(f"Model {model!r} does not support endpoint {endpoint!r}.") from exc
