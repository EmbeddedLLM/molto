"""Explicit client selection for every accepted provider/protocol combination."""

from molto.providers import Endpoint, Provider
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.base import BaseClient
from molto.upstream.bedrock import (
    BedrockChatCompletionsClient,
    BedrockMessagesClient,
    BedrockResponsesClient,
)
from molto.upstream.compatible import (
    AnthropicChatCompletionsClient,
    AnthropicResponsesClient,
    OpenAIMessagesClient,
)
from molto.upstream.fireworks import (
    FireworksChatCompletionsClient,
    FireworksMessagesClient,
    FireworksResponsesClient,
)
from molto.upstream.gemini import GeminiChatCompletionsClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from molto.upstream.openrouter import (
    OpenRouterChatCompletionsClient,
    OpenRouterMessagesClient,
    OpenRouterResponsesClient,
)
from molto.upstream.together import (
    TogetherAIChatCompletionsClient,
    TogetherAIMessagesClient,
    TogetherAIResponsesClient,
)
from molto.upstream.vllm import VLLMChatCompletionsClient, VLLMMessagesClient, VLLMResponsesClient

CLIENTS: dict[tuple[Provider, Endpoint], type[BaseClient]] = {
    ("openai", "responses"): OpenAIResponsesClient,
    ("openai", "chat/completions"): OpenAIChatCompletionsClient,
    ("openai", "messages"): OpenAIMessagesClient,
    ("anthropic", "responses"): AnthropicResponsesClient,
    ("anthropic", "chat/completions"): AnthropicChatCompletionsClient,
    ("anthropic", "messages"): AnthropicMessagesClient,
    ("vllm", "responses"): VLLMResponsesClient,
    ("vllm", "chat/completions"): VLLMChatCompletionsClient,
    ("vllm", "messages"): VLLMMessagesClient,
    ("bedrock", "responses"): BedrockResponsesClient,
    ("bedrock", "chat/completions"): BedrockChatCompletionsClient,
    ("bedrock", "messages"): BedrockMessagesClient,
    ("fireworks", "responses"): FireworksResponsesClient,
    ("fireworks", "chat/completions"): FireworksChatCompletionsClient,
    ("fireworks", "messages"): FireworksMessagesClient,
    ("openrouter", "responses"): OpenRouterResponsesClient,
    ("openrouter", "chat/completions"): OpenRouterChatCompletionsClient,
    ("openrouter", "messages"): OpenRouterMessagesClient,
    ("togetherai", "responses"): TogetherAIResponsesClient,
    ("togetherai", "chat/completions"): TogetherAIChatCompletionsClient,
    ("togetherai", "messages"): TogetherAIMessagesClient,
    ("gemini", "chat/completions"): GeminiChatCompletionsClient,
}
