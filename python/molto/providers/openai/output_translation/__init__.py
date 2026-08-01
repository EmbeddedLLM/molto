"""OpenAI SDK payload to portable output event translation.

Public translators are re-exported so `molto.providers.openai.output_translation`
import paths survive the module split.
"""

from .chat import chat_response_to_output_events, chat_stream_to_output_events  # noqa: F401
from .responses import (  # noqa: F401
    responses_response_to_output_events,
    responses_stream_to_output_events,
)

__all__ = [
    "chat_response_to_output_events",
    "chat_stream_to_output_events",
    "responses_response_to_output_events",
    "responses_stream_to_output_events",
]
