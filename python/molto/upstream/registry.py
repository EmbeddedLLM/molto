"""Explicit client selection for every accepted provider/protocol combination."""

from molto.providers import Endpoint, Provider
from molto.upstream.anthropic import AnthropicMessagesClient
from molto.upstream.base import BaseClient
from molto.upstream.bedrock import BedrockResponsesClient
from molto.upstream.fireworks import FireworksResponsesClient
from molto.upstream.gemini import GeminiChatCompletionsClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from molto.upstream.openrouter import OpenRouterResponsesClient
from molto.upstream.together import TogetherAIResponsesClient
from molto.upstream.vllm import VLLMChatCompletionsClient, VLLMMessagesClient, VLLMResponsesClient

CLIENTS: dict[tuple[Provider, Endpoint], type[BaseClient]] = {
    ("openai", "responses"): OpenAIResponsesClient,
    ("openai", "chat/completions"): OpenAIChatCompletionsClient,
    ("anthropic", "messages"): AnthropicMessagesClient,
    ("vllm", "responses"): VLLMResponsesClient,
    ("vllm", "chat/completions"): VLLMChatCompletionsClient,
    ("vllm", "messages"): VLLMMessagesClient,
    ("bedrock", "responses"): BedrockResponsesClient,
    ("fireworks", "responses"): FireworksResponsesClient,
    ("openrouter", "responses"): OpenRouterResponsesClient,
    ("togetherai", "responses"): TogetherAIResponsesClient,
    ("gemini", "chat/completions"): GeminiChatCompletionsClient,
}
