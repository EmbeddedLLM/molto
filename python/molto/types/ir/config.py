from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import ConfigDict, Field

from .base import IRModel

Probability = Annotated[float, Field(ge=0, le=1)]
TokenLimit = Annotated[int, Field(ge=1)]
Penalty = Annotated[float, Field(ge=-2, le=2)]
TopLogprobs = Annotated[int, Field(ge=0)]


class TextOutputFormat(IRModel):
    type: Literal["text"] = "text"


class JsonObjectOutputFormat(IRModel):
    type: Literal["json_object"] = "json_object"


class JsonSchemaOutputFormat(IRModel):
    model_config = ConfigDict(
        serialize_by_alias=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    type: Literal["json_schema"] = "json_schema"
    name: str
    schema_: dict[str, object] = Field(alias="schema", serialization_alias="schema")
    description: str | None = None
    strict: bool | None = None


OutputFormat = Annotated[
    Union[TextOutputFormat, JsonObjectOutputFormat, JsonSchemaOutputFormat],
    Field(discriminator="type"),
]


class CustomVoice(IRModel):
    id: str


class AudioOutputConfig(IRModel):
    format: Literal["wav", "aac", "mp3", "flac", "opus", "pcm16"]
    voice: str | CustomVoice


class WebSearchApproximateLocation(IRModel):
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None


class WebSearchUserLocation(IRModel):
    type: Literal["approximate"]
    approximate: WebSearchApproximateLocation


class WebSearchOptions(IRModel):
    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None


class StreamOptions(IRModel):
    include_usage: bool | None = None
    include_obfuscation: bool | None = None


class GenerationConfig(IRModel):
    """Portable generation controls accepted by the OpenAI-compatible boundary."""

    max_output_tokens: TokenLimit | None = None
    temperature: float | None = None
    top_p: Probability | None = None
    frequency_penalty: Penalty | None = None
    presence_penalty: Penalty | None = None
    top_logprobs: TopLogprobs | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None
    parallel_tool_calls: bool | None = None
    max_tool_calls: TokenLimit | None = None
    modalities: tuple[Literal["text", "audio"], ...] | None = None
    audio_output: AudioOutputConfig | None = None
    web_search_options: WebSearchOptions | None = None
