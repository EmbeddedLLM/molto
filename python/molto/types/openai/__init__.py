"""OpenAI public wire formats.

Each subpackage (`responses`, `chat_completions`) re-exports its full public
surface with a computed `__all__`.
"""

from . import chat_completions as chat_completions
from . import responses as responses
from .chat_completions import OpenAIChatCompletionRequest
from .responses import OpenAIResponsesRequest

__all__ = [
    "OpenAIChatCompletionRequest",
    "OpenAIResponsesRequest",
    "chat_completions",
    "responses",
]
