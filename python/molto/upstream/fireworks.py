"""Fireworks clients using the shared OpenAI protocol implementations."""

from dataclasses import replace
from typing import Any

from molto.providers import ProviderCapabilities
from molto.types.normalized import NormalizedRequest, Thinking, ThinkingDelta
from molto.upstream.openai import ChatCompletionsClient, ResponsesClient


class FireworksResponsesClient(ResponsesClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"

    @classmethod
    def resolve_capabilities(cls, profile: ProviderCapabilities) -> ProviderCapabilities:
        return replace(profile, thinking="raw")

    def _reasoning_part(self, item: Any) -> Thinking:
        part = super()._reasoning_part(item)
        # Molto treats Fireworks summary_text as raw reasoning.
        return Thinking(
            content=part.content or part.summary,
            summary="",
            signature=part.signature,
            id=part.id,
        )

    def _reasoning_delta(self, text: str, *, summary: bool) -> ThinkingDelta:
        return ThinkingDelta(content_delta=text)

    @staticmethod
    def _process_input(request: NormalizedRequest) -> list[dict[str, Any]] | None:
        items = ResponsesClient._process_input(request)
        # Preserve the recorded continuation shape, rather than assuming that
        # acceptance of reasoning_text proves the provider uses that history.
        for item in items or []:
            if item.get("type") == "reasoning" and item.get("content"):
                item["summary"] = [
                    {"type": "summary_text", "text": part["text"]} for part in item.pop("content")
                ]
        return items


class FireworksChatCompletionsClient(ChatCompletionsClient):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    REQUIRED_API_KEY = "FIREWORKS_API_KEY"
    OMIT_NULL_MESSAGE_FIELDS = True
    OMIT_INCLUDE_OBFUSCATION = True

    @classmethod
    def resolve_capabilities(cls, profile: ProviderCapabilities) -> ProviderCapabilities:
        return replace(profile, thinking="raw")
