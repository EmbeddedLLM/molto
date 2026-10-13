"""Together AI transport using the OpenAI-compatible Chat Completions endpoint."""

from typing import Any

from molto.types.exceptions import UnsupportedInput
from molto.types.normalized import TextOutput, Thinking, ToolCall
from molto.upstream.openai import OpenAIChatCompletionsClient


class TogetherAIClient(OpenAIChatCompletionsClient):
    """Translate Together's `reasoning` field as unsigned raw thinking."""

    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    reasoning_field = "reasoning"

    @classmethod
    def _assistant_message(cls, parts: list[Thinking | TextOutput | ToolCall]) -> dict[str, Any]:
        thinking_parts = [part for part in parts if isinstance(part, Thinking)]
        if any(part.summary for part in thinking_parts):
            raise UnsupportedInput(
                "Together Chat Completions cannot represent reasoning summaries."
            )
        # Together preserved thinking needs the original bytes and block sequence.
        # Chat history has one reasoning string, so joining distinct blocks loses fidelity.
        if len(thinking_parts) > 1:
            raise UnsupportedInput(
                "Together Chat Completions cannot preserve multiple reasoning blocks."
            )
        return super()._assistant_message(parts)
