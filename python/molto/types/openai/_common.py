"""Concepts shared verbatim across the OpenAI Chat Completions and Responses API models.

Only genuinely identical definitions belong here. The two endpoints deliberately
differ in wire shape (nested vs flat formats, tool nesting, optionality), so most
of their models stay endpoint-specific; this module holds the rare exact twins so
a supported-value change is one edit, not two.
"""

from typing import Literal, TypeAlias

# Reasoning effort levels accepted by both endpoints' request models.
OpenAIReasoningEffortLevel: TypeAlias = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh"
]
