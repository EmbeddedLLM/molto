from base64 import b64encode
from collections.abc import Callable, Iterator
from textwrap import dedent
from time import time
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import BeforeValidator, Discriminator, Field, model_validator

from molto.types.base import ExtraModel, StrictModel, ZeroIfNoneInt

PromptCacheTtl: TypeAlias = Literal["5m", "1h", "30m"]


class CacheBreakpoint(StrictModel):
    """The end of a reusable prefix, attached to its final content part."""

    ttl: Annotated[
        PromptCacheTtl | None,
        Field(
            description="Requested minimum cache lifetime, overriding the request lifetime if set."
        ),
    ] = None


class SystemPrompt(StrictModel):
    """A system prompt, generally written by the application developer."""

    content: Annotated[str, Field(description="The content of the prompt.")]
    cache_breakpoint: Annotated[
        CacheBreakpoint | None, Field(description="Cache the prefix ending at this part.")
    ] = None
    content_type: Annotated[
        Literal["system-prompt"],
        Field(description="Content type identifier as a discriminator."),
    ] = "system-prompt"


class TextInput(StrictModel):
    """Text content sent to the model."""

    content: Annotated[str, Field(description="The content sent to the model.")]
    cache_breakpoint: Annotated[
        CacheBreakpoint | None, Field(description="Cache the prefix ending at this part.")
    ] = None
    content_type: Annotated[
        Literal["text-input"],
        Field(description="Content type identifier as a discriminator."),
    ] = "text-input"


class ImageUrlInput(StrictModel):
    """An image supplied by remote URL or data URL."""

    url: Annotated[str, Field(description="The remote URL or data URL of the image.")]
    cache_breakpoint: Annotated[
        CacheBreakpoint | None, Field(description="Cache the prefix ending at this part.")
    ] = None
    detail: Annotated[
        Literal["auto", "low", "high", "original"] | None,
        Field(description="The requested image fidelity, when specified."),
    ] = None
    content_type: Annotated[
        Literal["image-url-input"],
        Field(description="Content type identifier as a discriminator."),
    ] = "image-url-input"

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        media_type: str,
        detail: Literal["auto", "low", "high", "original"] | None = None,
    ) -> Self:
        """Construct a base64 image input with an optional fidelity setting."""
        return cls(
            url=f"data:{media_type};base64,{b64encode(data).decode('ascii')}", detail=detail
        )


UserContent = Annotated[
    TextInput | ImageUrlInput,
    Discriminator("content_type"),
]

ToolReturnContent = Annotated[
    TextInput | ImageUrlInput,
    Discriminator("content_type"),
]


class UserInput(StrictModel):
    """One user message containing its ordered input content parts."""

    content: Annotated[
        list[UserContent],
        Field(description="The ordered content parts in the user message."),
    ]
    content_type: Annotated[
        Literal["user-input"],
        Field(description="Content type identifier as a discriminator."),
    ] = "user-input"


# class _FileUrlInput(ExtraModel):
#     """Common fields for URL-based user content."""

#     url: Annotated[str, Field(description="The URL of the file.")]
#     vendor_metadata: Annotated[
#         dict[str, Any] | None,
#         Field(description="Vendor-specific metadata for the file."),
#     ] = None
#     media_type: Annotated[
#         str | None,
#         Field(description="The media type of the file."),
#     ] = None
#     identifier: Annotated[
#         str | None,
#         Field(description="An optional identifier for the file."),
#     ] = None


# class ImageUrl(_FileUrl):
#     """An image referenced by URL."""

#     content_type: Annotated[
#         Literal["image-url"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "image-url"


# class AudioUrl(_FileUrl):
#     """An audio file referenced by URL."""

#     content_type: Annotated[
#         Literal["audio-url"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "audio-url"


# class DocumentUrl(_FileUrl):
#     """A document referenced by URL."""

#     content_type: Annotated[
#         Literal["document-url"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "document-url"


# class VideoUrl(_FileUrl):
#     """A video referenced by URL."""

#     content_type: Annotated[
#         Literal["video-url"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "video-url"


# class BinaryInput(ExtraModel):
#     """Binary user content."""

#     data: Annotated[bytes, Field(description="The binary file data.")]
#     media_type: Annotated[str, Field(description="The media type of the binary data.")]
#     vendor_metadata: Annotated[
#         dict[str, Any] | None,
#         Field(description="Vendor-specific metadata for the file."),
#     ] = None
#     identifier: Annotated[
#         str | None,
#         Field(description="An optional identifier for the file."),
#     ] = None
#     content_type: Annotated[
#         Literal["binary"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "binary"


# UploadedFileProviderName: TypeAlias = Literal[
#     "anthropic",
#     "openai",
#     "google",
#     "google-cloud",
#     "google-gla",
#     "google-vertex",
#     "bedrock",
#     "xai",
# ]


# class UploadedFileInput(ExtraModel):
#     """A reference to a provider-uploaded file."""

#     file_id: Annotated[str, Field(description="The provider-specific file identifier.")]
#     provider_name: Annotated[
#         UploadedFileProviderName,
#         Field(description="The provider that owns the file."),
#     ]
#     vendor_metadata: Annotated[
#         dict[str, Any] | None,
#         Field(description="Vendor-specific metadata for the file."),
#     ] = None
#     media_type: Annotated[
#         str | None,
#         Field(description="The media type of the file."),
#     ] = None
#     identifier: Annotated[
#         str | None,
#         Field(description="An optional identifier for the file."),
#     ] = None
#     content_type: Annotated[
#         Literal["uploaded-file"],
#         Field(description="Content type identifier as a discriminator."),
#     ] = "uploaded-file"


class ToolReturn(StrictModel):
    """Tool return parts."""

    id: Annotated[
        str | None,
        Field(description="The unique ID of the function tool call output."),
    ] = None
    tool_name: Annotated[
        str | None,
        Field(description="The name of the tool that was called."),
    ] = None
    content: Annotated[
        list[ToolReturnContent],
        Field(description="The ordered content returned by the tool."),
    ]
    tool_call_id: Annotated[
        str,
        Field(description="The tool call identifier."),
    ]
    outcome: Annotated[
        Literal["success", "failed", "denied", "interrupted"],
        Field(
            description=(
                "The outcome of the tool call. "
                "Only `'failed'` is mapped to a provider's native error channel "
                "(e.g. Anthropic `is_error`, Bedrock `status='error'`)"
            )
        ),
    ] = "success"
    content_type: Annotated[
        Literal["tool-return"],
        Field(description="Content type identifier as a discriminator."),
    ] = "tool-return"


class WebCitation(StrictModel):
    """URL attribution to a range of Unicode code points in one text part."""

    url: Annotated[str, Field(description="The attributed source URL.")]
    title: Annotated[str | None, Field(description="The observed source title.")] = None
    start_index: Annotated[int, Field(description="Inclusive offset within the text part.")]
    end_index: Annotated[int, Field(description="Exclusive offset within the text part.")]
    cited_text: Annotated[
        str | None, Field(description="The original source excerpt, when supplied.")
    ] = None
    encrypted_index: Annotated[
        str | None, Field(description="Opaque citation state retained for history.")
    ] = None


class WebSearchSource(StrictModel):
    """An observed search result, retaining available opaque history context."""

    url: Annotated[str, Field(description="The retrieved source URL.")]
    title: Annotated[str | None, Field(description="The observed source title.")] = None
    page_age: Annotated[str | None, Field(description="The observed source age.")] = None
    encrypted_content: Annotated[
        str | None, Field(description="Opaque retrieved context retained for history.")
    ] = None


class SearchCall(ExtraModel):
    """A provider-executed operation, distinct from a client function call."""

    id: Annotated[str, Field(description="The hosted operation identifier.")]
    action: Annotated[
        Literal["search", "open_page", "find_in_page"],
        Field(description="The observed hosted operation."),
    ] = "search"
    query: Annotated[str | None, Field(description="The observed singular search query.")] = None
    queries: Annotated[
        list[str] | None, Field(description="The observed list of search queries.")
    ] = None
    url: Annotated[str | None, Field(description="The target of a browsing operation.")] = None
    pattern: Annotated[str | None, Field(description="The text sought within a page.")] = None
    status: Annotated[
        Literal["in_progress", "searching", "completed", "failed"],
        Field(description="The observed execution status of the hosted operation."),
    ] = "in_progress"
    content_type: Annotated[
        Literal["search-call"], Field(description="Content type identifier.")
    ] = "search-call"


class SearchResult(ExtraModel):
    """The result of a hosted search, linked to its provider-executed call."""

    tool_call_id: Annotated[str, Field(description="The corresponding hosted call identifier.")]
    sources: Annotated[
        list[WebSearchSource] | None,
        Field(description="Observed results; null means unavailable, not an empty result list."),
    ] = None
    error: Annotated[str | None, Field(description="The observed search error, if any.")] = None
    content_type: Annotated[
        Literal["search-result"], Field(description="Content type identifier.")
    ] = "search-result"


class TextOutput(ExtraModel):
    """A plain text response from a model, including one used as conversation history."""

    content: Annotated[
        str,
        Field(description="The text content of the response."),
    ]
    cache_breakpoint: Annotated[
        CacheBreakpoint | None,
        Field(description="Cache the prefix ending at this part when used as history."),
    ] = None
    id: Annotated[
        str | None,
        Field(description="An optional identifier of the text part."),
    ] = None
    citations: Annotated[
        list[WebCitation], Field(description="URL attributions local to this text part.")
    ] = []
    defer_citations: Annotated[
        bool,
        Field(description="Delay block-based projection until the citation snapshot is complete."),
    ] = False
    content_type: Annotated[
        Literal["text-output"],
        Field(description="Content type identifier as a discriminator."),
    ] = "text-output"


class ToolCall(ExtraModel):
    """A tool call from a model."""

    tool_name: Annotated[
        str,
        Field(description="The name of the tool to call."),
    ]
    args: Annotated[
        str,
        Field(description="The arguments to pass to the tool."),
    ]
    tool_call_id: Annotated[
        str,
        Field(description="The tool call identifier."),
    ]
    id: Annotated[
        str | None,
        Field(description="An optional identifier of the tool call part"),
    ] = None
    content_type: Annotated[
        Literal["tool-call"],
        Field(description="Content type identifier as a discriminator."),
    ] = "tool-call"


class Thinking(ExtraModel):
    """A thinking response from a model."""

    content: Annotated[
        str,
        Field(description="The reasoning content."),
    ]
    summary: Annotated[
        str,
        Field(description="A summary of the reasoning."),
    ]
    id: Annotated[
        str | None,
        Field(description="The identifier of the thinking part."),
    ] = None
    signature: Annotated[
        str | None,
        Field(description="Provider-specific opaque state for the thinking."),
    ] = None
    content_type: Annotated[
        Literal["thinking"],
        Field(description="Content type identifier as a discriminator."),
    ] = "thinking"

    @model_validator(mode="after")
    def _validate_representation(self) -> Self:
        if self.content and self.summary:
            raise ValueError("Thinking cannot contain both raw reasoning and a summary.")
        if self.content and self.signature:
            raise ValueError("A thinking signature cannot accompany raw reasoning.")
        return self


OutputItem = Annotated[
    Thinking | TextOutput | ToolCall | SearchCall | SearchResult,
    # | ToolSearchCallPart
    # | LoadCapabilityCallPart
    # | NativeToolSearchCallPart
    # | NativeToolCallPart
    # | NativeToolSearchReturnPart
    # | NativeToolReturnPart
    # | CompactionPart
    # | FilePart,
    Discriminator("content_type"),
]


def _reject_input_extras(model_type: type[ExtraModel]) -> Callable[[Any], Any]:
    def validate(value: Any) -> Any:
        if isinstance(value, dict):
            extra_fields = {
                key for key in value if key not in model_type.model_fields or key == "extra_fields"
            }
        elif isinstance(value, model_type):
            extra_fields = value.extra_fields.keys()
        else:
            return value

        if extra_fields:
            raise ValueError(f"Extra request fields are not permitted: {sorted(extra_fields)}")
        return value

    return validate


_ThinkingInput = Annotated[
    Thinking,
    BeforeValidator(_reject_input_extras(Thinking)),
]
_TextOutputInput = Annotated[
    TextOutput,
    BeforeValidator(_reject_input_extras(TextOutput)),
]
_ToolCallInput = Annotated[
    ToolCall,
    BeforeValidator(_reject_input_extras(ToolCall)),
]
_SearchCallInput = Annotated[
    SearchCall,
    BeforeValidator(_reject_input_extras(SearchCall)),
]
_SearchResultInput = Annotated[
    SearchResult,
    BeforeValidator(_reject_input_extras(SearchResult)),
]


AssistantContent = Annotated[
    _ThinkingInput | _TextOutputInput | _ToolCallInput | _SearchCallInput | _SearchResultInput,
    Discriminator("content_type"),
]


class AssistantInput(StrictModel):
    """One assistant history message containing its ordered output content parts."""

    content: Annotated[
        list[AssistantContent],
        Field(description="The ordered content parts in the assistant message."),
    ]
    id: Annotated[
        str | None,
        Field(description="An optional identifier of the assistant message."),
    ] = None
    content_type: Annotated[
        Literal["assistant-input"],
        Field(description="Content type identifier as a discriminator."),
    ] = "assistant-input"


InputItem = Annotated[
    SystemPrompt
    | UserInput
    | AssistantInput
    | _ThinkingInput
    | _ToolCallInput
    | _SearchCallInput
    | _SearchResultInput
    | ToolReturn,
    # | ToolSearchReturnPart
    # | LoadCapabilityReturnPart
    # | ToolAvailabilityDeltaPart
    Discriminator("content_type"),
]
Input: TypeAlias = list[InputItem]


ReasoningEffort: TypeAlias = (
    Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
)


class Reasoning(StrictModel):
    """
    Configuration options for
    [reasoning models](https://platform.openai.com/docs/guides/reasoning).
    """

    effort: Annotated[
        ReasoningEffort | None,
        Field(description="Constrains effort on reasoning for reasoning models."),
    ] = None
    summary: Annotated[
        Literal["auto", "concise", "detailed"] | None,
        Field(description="A summary of the reasoning performed by the model."),
    ] = None


class FunctionTool(StrictModel):
    """A client-defined function available to the model."""

    name: Annotated[str, Field(description="The function name.")]
    description: Annotated[
        str | None,
        Field(description="A description of what the function does."),
    ] = None
    parameters: Annotated[
        dict[str, Any],
        Field(description="The JSON Schema for the function arguments."),
    ]
    strict: Annotated[
        bool,
        Field(description="Whether to enforce strict schema validation."),
    ] = False
    type: Annotated[
        Literal["function"],
        Field(description="Tool type identifier."),
    ] = "function"


class FunctionToolChoice(StrictModel):
    """Require the model to call one specific client-defined function."""

    name: Annotated[str, Field(description="The required function name.")]
    type: Annotated[
        Literal["function"],
        Field(description="Tool choice type identifier."),
    ] = "function"


class SearchLocation(StrictModel):
    """Approximate location used to tailor a hosted search."""

    city: Annotated[str | None, Field(description="The user's approximate city.")] = None
    country: Annotated[str | None, Field(description="The user's approximate country.")] = None
    region: Annotated[str | None, Field(description="The user's approximate region.")] = None
    timezone: Annotated[str | None, Field(description="The user's timezone.")] = None


class WebSearchTool(StrictModel):
    """Basic provider-hosted search configuration."""

    allowed_domains: Annotated[
        list[str] | None, Field(description="Restrict search to these domains.")
    ] = None
    blocked_domains: Annotated[
        list[str] | None, Field(description="Exclude these domains from search.")
    ] = None
    search_context_size: Annotated[
        Literal["low", "medium", "high"] | None,
        Field(
            description="Explicit search context budget; omission retains the provider default."
        ),
    ] = None
    user_location: Annotated[
        SearchLocation | None, Field(description="Approximate search location.")
    ] = None
    type: Annotated[Literal["web_search"], Field(description="Tool type identifier.")] = (
        "web_search"
    )


class WebSearchToolChoice(StrictModel):
    """Require hosted search specifically, independently of client function names."""

    type: Annotated[Literal["web_search"], Field(description="Required hosted tool.")] = (
        "web_search"
    )


Tool = Annotated[FunctionTool | WebSearchTool, Discriminator("type")]
ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"] | FunctionToolChoice | WebSearchToolChoice
)


class StreamOptions(StrictModel):
    """Options for streaming responses. Only set this when you set `stream: true`."""

    include_obfuscation: Annotated[
        bool | None,
        Field(
            description=dedent("""\
            When true, stream obfuscation will be enabled.

            Stream obfuscation adds random characters to an `obfuscation` field on streaming
            delta events to normalize payload sizes as a mitigation to certain side-channel
            attacks. These obfuscation fields are included by default, but add a small
            amount of overhead to the data stream.
            """)
        ),
    ] = None


PromptCacheMode: TypeAlias = Literal["implicit", "explicit"]
PromptCacheRetention: TypeAlias = Literal["in_memory", "24h"]


class PromptCache(StrictModel):
    """Prompt-caching intent for one request.

    Caching changes cost and latency, never model-visible content, so every field is
    optional and each target protocol projects only the subset it represents. Providers
    disagree on the vocabulary: `key` and `retention` exist solely on the OpenAI side,
    and the supported `ttl` values do not overlap between OpenAI (`30m`) and Anthropic
    (`5m`, `1h`). Clients map requested lifetimes under the documented cache policy:
    OpenAI uses `30m` even for a requested `1h` minimum, while Messages rounds `30m`
    up to `1h`. Messages drops the caller key without enabling caching on its own.
    """

    key: Annotated[
        str | None,
        Field(description="A caller-supplied key that partitions the provider's cache."),
    ] = None
    mode: Annotated[
        PromptCacheMode | None,
        Field(
            description=("Whether the provider or the request marks the cached prefix boundary.")
        ),
    ] = None
    ttl: Annotated[
        PromptCacheTtl | None,
        Field(
            description="Requested minimum lifetime of a cached prefix; target mappings may approximate it."
        ),
    ] = None
    retention: Annotated[
        PromptCacheRetention | None,
        Field(description="Where the provider may retain the cached prefix."),
    ] = None


class NormalizedRequest(StrictModel):
    input: Annotated[
        Input | None,
        Field(description="Text, image, or file inputs to the model."),
    ] = None
    max_output_tokens: Annotated[
        int | None,
        Field(description="The upper bound for generated tokens."),
    ] = None
    max_tool_calls: Annotated[
        int | None,
        Field(description="The maximum number of built-in tool calls."),
    ] = None
    model: Annotated[
        str | None,
        Field(description="The model ID used to generate the response."),
    ] = None
    parallel_tool_calls: Annotated[
        bool | None,
        Field(description="Whether the model may run tool calls in parallel."),
    ] = None
    prompt_cache: Annotated[
        PromptCache | None,
        Field(description="Prompt-caching intent for the request."),
    ] = None
    reasoning: Annotated[
        Reasoning | None,
        Field(description="Reasoning configuration for supported models."),
    ] = None
    stream: Annotated[
        bool | None,
        Field(description="Whether to stream the response."),
    ] = None
    stream_options: Annotated[
        StreamOptions | None,
        Field(description="Options for streaming responses."),
    ] = None
    temperature: Annotated[
        float | None,
        Field(description="Sampling temperature between 0 and 2."),
    ] = None
    tool_choice: Annotated[
        ToolChoice | None,
        Field(description="How the model should select tools."),
    ] = None
    tools: Annotated[
        list[Tool] | None,
        Field(description="Tools available to the model."),
    ] = None
    top_p: Annotated[
        float | None,
        Field(description="The nucleus-sampling probability mass."),
    ] = None

    def iter_cache_breakpoints(self) -> Iterator[CacheBreakpoint]:
        """Yield explicit markers in input order, including nested content parts."""
        for item in self.input or []:
            if isinstance(item, (UserInput, AssistantInput, ToolReturn)):
                for part in item.content:
                    if (
                        isinstance(part, (TextInput, ImageUrlInput, TextOutput))
                        and part.cache_breakpoint is not None
                    ):
                        yield part.cache_breakpoint
            elif isinstance(item, SystemPrompt) and item.cache_breakpoint is not None:
                yield item.cache_breakpoint


class TextDelta(ExtraModel):
    """A partial update (delta) for a `TextOutput` to append new text content."""

    content_delta: Annotated[
        str,
        Field(description="The incremental text content."),
    ]
    delta_type: Annotated[
        Literal["text-delta"],
        Field(description="Part delta type identifier, used as a discriminator."),
    ] = "text-delta"


class CitationDelta(ExtraModel):
    """An attribution added to the associated text part."""

    citation: Annotated[WebCitation, Field(description="The completed citation association.")]
    delta_type: Annotated[
        Literal["citation-delta"], Field(description="Part delta type identifier.")
    ] = "citation-delta"


class ThinkingDelta(ExtraModel):
    """An update to append thinking text or replace its provider-specific state."""

    content_delta: Annotated[
        str | None,
        Field(description="The incremental reasoning content."),
    ] = None
    summary_delta: Annotated[
        str | None,
        Field(description="The incremental summary of the reasoning."),
    ] = None
    signature_delta: Annotated[
        str | None,
        Field(description="An optional replacement for provider-specific opaque state."),
    ] = None
    delta_type: Annotated[
        Literal["thinking-delta"],
        Field(description="Part delta type identifier, used as a discriminator."),
    ] = "thinking-delta"

    @model_validator(mode="after")
    def _validate_representation(self) -> Self:
        if self.content_delta and self.summary_delta:
            raise ValueError("A thinking delta cannot contain both raw reasoning and a summary.")
        if self.content_delta and self.signature_delta:
            raise ValueError("A signature delta cannot accompany raw reasoning.")
        return self


class ToolCallDelta(ExtraModel):
    """A partial update (delta) for a `ToolCall` to modify tool name, arguments, or tool call ID."""

    tool_name_delta: Annotated[
        str | None,
        Field(description="Incremental text to add to the existing tool name, if any."),
    ] = None
    args_delta: Annotated[
        str | None,
        Field(description="Incremental data to add to the tool arguments."),
    ] = None
    tool_call_id: Annotated[
        str | None,
        Field(description="Optional tool call identifier."),
    ] = None
    delta_type: Annotated[
        Literal["tool-call-delta"],
        Field(description="Part delta type identifier, used as a discriminator."),
    ] = "tool-call-delta"


DeltaItem = Annotated[
    TextDelta | ThinkingDelta | ToolCallDelta | CitationDelta,
    Discriminator("delta_type"),
]


class InputTokenDetails(ExtraModel):
    cached_tokens: Annotated[
        ZeroIfNoneInt,
        Field(description="The number of cached input tokens."),
    ] = 0
    cache_write_tokens: Annotated[
        ZeroIfNoneInt,
        Field(description="The number of input tokens written to a prompt cache."),
    ] = 0


class OutputTokenDetails(ExtraModel):
    reasoning_tokens: Annotated[
        ZeroIfNoneInt,
        Field(description="The number of output tokens used for reasoning."),
    ] = 0


class Usage(ExtraModel):
    input_tokens: Annotated[
        ZeroIfNoneInt,
        Field(description="The number of input tokens used."),
    ]
    input_tokens_details: Annotated[
        InputTokenDetails,
        Field(description="A breakdown of the input token usage."),
    ]
    output_tokens: Annotated[
        ZeroIfNoneInt,
        Field(description="The number of output tokens used."),
    ]
    output_tokens_details: Annotated[
        OutputTokenDetails,
        Field(description="A breakdown of the output token usage."),
    ]
    total_tokens: Annotated[
        ZeroIfNoneInt,
        Field(description="The total number of tokens used."),
    ]


class ResponseStartEvent(ExtraModel):
    """Provider-backed response metadata available when a response stream starts."""

    id: Annotated[
        str,
        Field(description="The upstream response identifier."),
    ]
    created_at: Annotated[
        int,
        Field(description="The upstream response creation time."),
    ] = Field(default_factory=lambda: int(time()))
    model: Annotated[
        str,
        Field(description="The model that produced the response."),
    ]
    service_tier: Annotated[
        str | None,
        Field(description="The service tier used for the response."),
    ] = None
    web_search_enabled: Annotated[
        bool | None,
        Field(
            description="Whether the effective upstream request enables hosted web search, when known."
        ),
    ] = None
    event_kind: Annotated[
        Literal["response_start"],
        Field(description="Event type identifier, used as a discriminator."),
    ] = "response_start"


TerminationReason: TypeAlias = Literal[
    "content_filter",
    "max_output_tokens",
    "model_context_window_exceeded",
    "pause",
    "refusal",
    "stop",
    "stop_sequence",
    "tool_call",
    "unknown",
]


class ResponseEndEvent(ExtraModel):
    """Provider-backed response metadata available when a response stream ends."""

    id: Annotated[
        str,
        Field(description="The upstream response identifier."),
    ]
    created_at: Annotated[
        int,
        Field(description="The upstream response creation time."),
    ]
    model: Annotated[
        str,
        Field(description="The model that produced the response."),
    ]
    service_tier: Annotated[
        str | None,
        Field(description="The service tier used for the response."),
    ] = None
    termination_reason: Annotated[
        TerminationReason,
        Field(
            description="The provider-neutral reason generation terminated, or 'unknown' if unavailable."
        ),
    ]
    stop_sequence: Annotated[
        str | None,
        Field(description="The matched stop sequence, when one terminated generation."),
    ] = None
    usage: Annotated[
        Usage | None,
        Field(description="The response token usage."),
    ] = None
    event_kind: Annotated[
        Literal["response_end"],
        Field(description="Event type identifier, used as a discriminator."),
    ] = "response_end"


class PartStartEvent(ExtraModel):
    """An event indicating that a new part has started.

    If multiple `PartStartEvent`s are received with the same index,
    the new one should fully replace the old one.
    """

    index: Annotated[
        int,
        Field(description="The index of the part within the overall response parts list."),
    ]
    part: Annotated[
        OutputItem,
        Field(description="The newly started response part."),
    ]
    previous_part_kind: Annotated[
        Literal[
            "text",
            "thinking",
            "tool-call",
            "builtin-tool-call",
            "builtin-tool-return",
            "compaction",
            "file",
        ]
        | None,
        Field(description="The kind of the previous part, if any."),
    ] = None
    event_kind: Annotated[
        Literal["part_start"],
        Field(description="Event type identifier, used as a discriminator."),
    ] = "part_start"


class PartDeltaEvent(ExtraModel):
    """An event indicating a delta update for an existing part."""

    index: Annotated[
        int,
        Field(description="The index of the part within the overall response parts list."),
    ]
    delta: Annotated[
        DeltaItem,
        Field(description="The delta to apply to the specified part."),
    ]
    event_kind: Annotated[
        Literal["part_delta"],
        Field(description="Event type identifier, used as a discriminator."),
    ] = "part_delta"


class PartEndEvent(ExtraModel):
    """An event indicating that a part is complete."""

    index: Annotated[
        int,
        Field(description="The index of the part within the overall response parts list."),
    ]
    part: Annotated[
        OutputItem,
        Field(description="The complete response part."),
    ]
    next_part_kind: Annotated[
        Literal[
            "text",
            "thinking",
            "tool-call",
            "builtin-tool-call",
            "builtin-tool-return",
            "compaction",
            "file",
        ]
        | None,
        Field(description="The kind of the next part, if any."),
    ] = None
    event_kind: Annotated[
        Literal["part_end"],
        Field(description="Event type identifier, used as a discriminator."),
    ] = "part_end"


class FinalResultEvent(ExtraModel):
    """An event indicating the response to the current model request matches the output schema and will produce a result."""

    tool_name: Annotated[
        str | None,
        Field(
            description=(
                "The name of the output tool that was called. "
                "None if the result is from text content."
            )
        ),
    ]
    tool_call_id: Annotated[
        str | None,
        Field(description="The tool call ID associated with this result, if any."),
    ]
    event_kind: Annotated[
        Literal["final_result"],
        Field(description="Event type identifier, used as a discriminator."),
    ] = "final_result"


ModelResponseStreamEvent = Annotated[
    ResponseStartEvent
    | PartStartEvent
    | PartDeltaEvent
    | PartEndEvent
    | FinalResultEvent
    | ResponseEndEvent,
    Discriminator("event_kind"),
]
