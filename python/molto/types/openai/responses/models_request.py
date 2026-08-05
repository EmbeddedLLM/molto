"""OpenAI Responses request API models."""

from collections import defaultdict
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from molto.types.ir.portable import (
    InputIR,
)

from .models_items import (
    OpenAIAllowedToolsChoice,
    OpenAIConversation,
    OpenAIInput,
    OpenAIReasoningEffort,
    OpenAIResponsesCodeTool,
    OpenAIResponsesCodeToolBase,
    OpenAIResponsesFunctionTool,
    OpenAIResponsesImageGenerationTool,
    OpenAIResponsesInclude,
    OpenAIResponsesMcpTool,
    OpenAIResponsesNamespaceTool,
    OpenAIResponsesPlainTextConfig,
    OpenAIResponsesRestrictedTool,
    OpenAIResponsesTool,
    OpenAIResponsesWebSearchTool,
    OpenAIStreamOptions,
    OpenAITextConfig,
    OpenAIToolChoice,
)


def _validate_tools_unique(
    v: list[OpenAIResponsesTool] | None,
) -> list[OpenAIResponsesTool] | None:
    """Reject duplicate tool declarations across both request variants."""

    if v:
        tool_count = defaultdict(int)
        for t in v:
            if isinstance(t, OpenAIResponsesFunctionTool):
                tool_name = t.name
            elif isinstance(t, OpenAIResponsesCodeToolBase):
                tool_name = t.type
            elif isinstance(t, OpenAIResponsesWebSearchTool):
                tool_name = "web_search"
            elif isinstance(t, OpenAIResponsesImageGenerationTool):
                tool_name = t.type
            elif isinstance(t, OpenAIResponsesNamespaceTool):
                tool_name = t.name
            elif isinstance(t, OpenAIResponsesMcpTool):
                # MCP duplicate-by-server validation is handled in `as_run_settings(...)` where request-level
                # validation can include runtime manager availability checks.
                continue
            else:
                raise ValueError(f"Invalid tool type: {type(t)}")
            tool_count[tool_name] += 1
        if any(count > 1 for count in tool_count.values()):
            raise ValueError(
                f"Duplicate tool names found in tools: {[tool for tool, count in tool_count.items() if count > 1]}."
            )
    return v


class _ResponsesRequestBase(BaseModel):
    """Shared Responses request contract for the restricted and full variants."""

    model_config = ConfigDict(extra="forbid")

    input: Annotated[
        str | list[OpenAIInput],
        Field(
            description="Text, image, or file inputs to the model, used to generate a response."
        ),
    ]
    instructions: Annotated[
        str | None,
        Field(
            description=(
                "A system (or developer) message inserted into the model's context. "
                "When using along with `previous_response_id`, the instructions from a previous response will not be carried over to the next response. "
                "This makes it simple to swap out system (or developer) messages in new responses."
            ),
        ),
    ] = None
    # OpenAI Responses API reference: `instructions` are not carried over when using `previous_response_id`.
    # https://platform.openai.com/docs/guides/text#message-roles-and-instruction-following
    max_output_tokens: Annotated[
        int | None,
        Field(
            description=(
                "An upper bound for the number of tokens that can be generated for a response, "
                "including visible output tokens and reasoning tokens."
            ),
        ),
    ] = None
    max_tool_calls: Annotated[
        int | None,
        Field(
            description=(
                "The maximum number of total calls to native tools that can be processed in a response. "
                "This maximum number applies across all native tool calls, not per individual tool. "
                "Any further attempts to call a tool by the model will be ignored. "
                "As of 2026-02-22, observed OpenAI runtime behavior is not always consistent with this documented rule across model/tool paths."
            ),
        ),
    ] = None
    extra_body: Annotated[
        dict[str, Any] | None,
        Field(description="Additional provider-specific request body fields."),
    ] = None
    frequency_penalty: Annotated[
        float | None,
        Field(
            ge=-2,
            le=2,
            description="Penalize tokens according to their existing frequency.",
        ),
    ] = None
    parallel_tool_calls: Annotated[
        bool | None,
        Field(
            description="Whether to allow the model to run tool calls in parallel.",
        ),
    ] = None
    presence_penalty: Annotated[
        float | None,
        Field(
            ge=-2,
            le=2,
            description="Penalize tokens according to whether they have appeared.",
        ),
    ] = None
    previous_response_id: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("previous_response_id", "previous_responses_id"),
            description=(
                "The unique ID of the previous response to the model. Use this to create multi-turn conversations."
            ),
        ),
    ] = None
    reasoning: Annotated[
        OpenAIReasoningEffort | None,
        Field(
            description=(
                "Configuration options for reasoning models. "
                "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response."
            )
        ),
    ] = None
    store: Annotated[
        bool | None,
        Field(
            description="Whether to store the generated model response for later retrieval via API.",
        ),
    ] = None
    stream: Annotated[
        bool | None,
        Field(
            description=(
                "If set to `true`, the model response data will be streamed to the client as it is generated using server-sent events."
            ),
        ),
    ] = None
    temperature: Annotated[
        float | None,
        Field(
            gt=0,
            lt=2,
            description=(
                "What sampling temperature to use, between 0 and 2. "
                "Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. "
                "We generally recommend altering this with `top_p` but not both."
            ),
        ),
    ] = None
    tool_choice: Annotated[
        OpenAIToolChoice | None,
        Field(
            description=(
                "Controls which (if any) tool is called by the model. "
                "`none` means the model will not call any tool and instead generates a message. "
                "`auto` means the model can pick between generating a message or calling one or more tools. "
                "`required` means the model must call one or more tools. "
                "Alternatively, you can provide an `allowed_tools` list to constrain the model to a specific set of tools."
            ),
        ),
    ] = None
    top_logprobs: Annotated[
        int | None,
        Field(
            ge=0,
            description=(
                "An integer specifying the number of most likely tokens to return at each token position, "
                "each with an associated log probability."
            ),
        ),
    ] = None
    top_p: Annotated[
        float | None,
        Field(
            ge=0,
            le=1,
            description=(
                "An alternative to sampling with `temperature`, called nucleus sampling, "
                "where the model considers the results of the tokens with `top_p` probability mass. "
                "So 0.1 means only the tokens comprising the top 10% probability mass are considered. "
                "We generally recommend altering this with `temperature` but not both."
            ),
        ),
    ] = None
    user: Annotated[
        str | None,
        Field(description="A stable identifier for your end users."),
    ] = None


class OpenAIResponsesRestrictedRequest(_ResponsesRequestBase):
    """Responses request contract restricted to the vLLM-supported subset."""

    include: Annotated[
        list[OpenAIResponsesInclude] | None,
        Field(
            description=(
                "Specify additional output data to include in the model response. "
                "MVP: only `code_interpreter_call.outputs` is currently acted upon by the gateway."
            ),
        ),
    ] = None
    text: Annotated[
        OpenAIResponsesPlainTextConfig | None,
        Field(
            description=(
                "Configuration options for a text response from the model. "
                "Can be plain text or structured JSON data."
            ),
        ),
    ] = None
    tools: Annotated[
        list[OpenAIResponsesRestrictedTool] | None,
        Field(
            description=(
                "An array of tools the model may call while generating a response. "
                "You can specify which tool to use by setting the `tool_choice` parameter."
            ),
            min_length=1,
            examples=[
                [
                    OpenAIResponsesCodeToolBase(),
                    OpenAIResponsesFunctionTool(
                        name="get_weather",
                        parameters=dict(
                            type="object",
                            properties=dict(
                                location=dict(
                                    type="string",
                                    description="City and country, e.g. `Bogotá, Colombia`",
                                )
                            ),
                            required=["location"],
                            additionalProperties=False,
                        ),
                        description="Get current temperature for a given location.",
                    ),
                ],
            ],
        ),
    ] = None

    @field_validator("tools", mode="after")
    @classmethod
    def validate_tools(
        cls, v: list[OpenAIResponsesRestrictedTool] | None
    ) -> list[OpenAIResponsesRestrictedTool] | None:
        return _validate_tools_unique(v)

    @field_validator("tool_choice", mode="after")
    @classmethod
    def validate_tool_choice(cls, v: OpenAIToolChoice) -> OpenAIToolChoice:
        if isinstance(v, OpenAIAllowedToolsChoice):
            v.validate_hosted_tools()
        return v


class OpenAIResponsesBody(_ResponsesRequestBase):
    """Responses JSON body, without Molto routing information."""

    background: Annotated[
        Literal[False] | None,
        Field(
            description=(
                "Background execution is not supported because Molto returns the result of "
                "the current request. Only false may be supplied."
            )
        ),
    ] = None
    conversation: Annotated[
        str | OpenAIConversation | None,
        Field(
            description=(
                "The conversation that this response belongs to. "
                "Items from this conversation are prepended to `input_items` for this response request. "
                "Input items and output items from this response are automatically added to this conversation after this response completes."
            )
        ),
    ] = None
    include: Annotated[
        list[OpenAIResponsesInclude] | None,
        Field(
            description=(
                "Specify additional output data to include in the model response. Currently supported values are:\n"
                "`web_search_call.action.sources`: Include the sources of the web search tool call.\n"
                "`code_interpreter_call.outputs`: Includes the outputs of python code execution in code interpreter tool call items.\n"
                "`computer_call_output.output.image_url`: Include image urls from the computer call output.\n"
                "`file_search_call.results`: Include the search results of the file search tool call.\n"
                "`message.input_image.image_url:` Include image urls from the input message.\n"
                "`message.output_text.logprobs`: Include logprobs with assistant messages.\n"
                "`reasoning.encrypted_content`: Includes an encrypted version of reasoning tokens in reasoning item outputs. "
                "This enables reasoning items to be used in multi-turn conversations when using the Responses API statelessly "
                "(like when the store parameter is set to false, or when an organization is enrolled in the zero data retention program)."
            ),
        ),
    ] = None
    prompt_cache_key: Annotated[
        str | None,
        Field(
            description=(
                "Used to cache responses for similar requests to optimize your cache hit rates. "
                "Replaces the `user` field."
            ),
        ),
    ] = None
    prompt_cache_retention: Annotated[
        Literal["in_memory", "24h"] | None,
        Field(
            description=(
                "The retention policy for the prompt cache. "
                "Set to `24h` to enable extended prompt caching, "
                "which keeps cached prefixes active for longer, up to a maximum of 24 hours."
            ),
        ),
    ] = None
    safety_identifier: Annotated[
        str | None,
        Field(
            description=(
                "A stable identifier used to help detect users of your application that may be violating usage policies. "
                "The IDs should be a string that uniquely identifies each user. "
                "We recommend hashing their username or email address, in order to avoid sending us any identifying information."
            ),
        ),
    ] = None
    service_tier: Annotated[
        Literal["auto", "default", "flex", "scale", "priority"] | None,
        Field(
            description=(
                "Specifies the processing type used for serving the request.\n\n"
                "- If set to 'auto', then the request will be processed with the service tier configured in the Project settings. "
                "Unless otherwise configured, the Project will use 'default'.\n"
                "- If set to 'default', then the request will be processed with the standard pricing and performance for the selected model.\n"
                "- If set to 'flex' or 'priority', then the request will be processed with the corresponding service tier.\n"
                "- When not set, the default behavior is 'auto'.\n"
                "When the `service_tier` parameter is set, the response body will include the `service_tier` value based on the processing mode actually used to serve the request. "
                "This response value may be different from the value set in the parameter."
            ),
        ),
    ] = None
    stream_options: Annotated[
        OpenAIStreamOptions | None,
        Field(
            description=(
                "Options for streaming responses. Only set this when you set `stream: true`."
            ),
        ),
    ] = None
    text: Annotated[
        OpenAITextConfig | None,
        Field(
            description=(
                "Configuration options for a text response from the model. "
                "Can be plain text or structured JSON data."
            ),
        ),
    ] = None
    tools: Annotated[
        list[OpenAIResponsesTool] | None,
        Field(
            description=(
                "An array of tools the model may call while generating a response. "
                "You can specify which tool to use by setting the `tool_choice` parameter."
            ),
            min_length=1,
            examples=[
                [
                    OpenAIResponsesWebSearchTool(),
                    OpenAIResponsesCodeTool(),
                    OpenAIResponsesFunctionTool(
                        name="get_weather",
                        parameters=dict(
                            type="object",
                            properties=dict(
                                location=dict(
                                    type="string",
                                    description="City and country, e.g. `Bogotá, Colombia`",
                                )
                            ),
                            required=["location"],
                            additionalProperties=False,
                        ),
                        description="Get current temperature for a given location.",
                    ),
                ],
            ],
        ),
    ] = None
    truncation: Annotated[
        Literal["auto", "disabled"] | None,
        Field(
            description=(
                "The truncation strategy to use for the model response. "
                "`auto`: If the input to this Response exceeds the model's context window size, "
                "the model will truncate the response to fit the context window by dropping items from the beginning of the conversation. "
                "`disabled` (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error."
            ),
        ),
    ] = None

    @field_validator("tools", mode="after")
    @classmethod
    def validate_tools(
        cls, v: list[OpenAIResponsesTool] | None
    ) -> list[OpenAIResponsesTool] | None:
        return _validate_tools_unique(v)


class OpenAIResponsesRequest(OpenAIResponsesBody):
    """Legacy complete Responses request retained for direct adapter use."""

    model: Annotated[
        str,
        Field(
            description="The model used to generate the response; an optional provider prefix selects a non-OpenAI adapter."
        ),
    ]
    endpoint: Annotated[
        Literal["responses", "chat_completions"] | None,
        Field(
            default=None,
            exclude=True,
            validation_alias=AliasChoices("endpoint", "upstream"),
            description="Molto execution endpoint. This routing field is consumed before IR conversion.",
        ),
    ] = None

    def to_ir(self) -> InputIR:
        from .to_ir import _responses_request_to_ir

        return _responses_request_to_ir(self)
