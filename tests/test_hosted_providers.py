"""Offline destination/authentication probes; these do not certify hosted models."""

from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic, omit
from openai import AsyncOpenAI, OpenAIError
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import SecretStr

from molto import Molto
from molto.configs import ENV_CONFIG, EnvConfig
from molto.providers import PROVIDER_CAPABILITIES
from molto.types.exceptions import AuthenticationError
from molto.upstream.registry import CLIENTS
from molto.utils.io import dumps_sse, json_loads, read_yaml
from molto.utils.test import (
    assert_stream_lifecycle,
    collect_sdk_response,
    reconstructable_chat_completions_chunks,
    reconstructable_responses_event,
)

DESTINATIONS = [
    ("fireworks", "responses", "https://api.fireworks.ai/inference/v1", "/responses"),
    ("openrouter", "responses", "https://openrouter.ai/api/v1", "/responses"),
    ("togetherai", "responses", "https://api.together.ai/v1", "/responses"),
    (
        "fireworks",
        "chat/completions",
        "https://api.fireworks.ai/inference/v1",
        "/chat/completions",
    ),
    ("togetherai", "chat/completions", "https://api.together.ai/v1", "/chat/completions"),
    (
        "bedrock",
        "chat/completions",
        "https://bedrock-runtime.ap-southeast-1.amazonaws.com/openai/v1",
        "/chat/completions",
    ),
    (
        "gemini",
        "chat/completions",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "/chat/completions",
    ),
    (
        "bedrock",
        "responses",
        "https://bedrock-runtime.ap-southeast-1.amazonaws.com/openai/v1",
        "/responses",
    ),
    (
        "bedrock",
        "messages",
        "https://bedrock-runtime.ap-southeast-1.amazonaws.com/anthropic",
        "/v1/messages",
    ),
    (
        "bedrock",
        "messages",
        "https://bedrock-mantle.us-east-1.api.aws/anthropic",
        "/v1/messages",
    ),
]


@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize(
    "names",
    [
        ["AWS_BEARER_TOKEN_BEDROCK"],
        ["BEDROCK_API_KEY"],
        ["AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_API_KEY"],
    ],
)
def test_bedrock_credential_alias_preserves_secret_and_name_precedence(
    source, names, monkeypatch, tmp_path
):
    """Ensures both names load one secret field, preferring the AWS name within each source."""
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    values = {name: f"synthetic-{index}" for index, name in enumerate(names)}
    env_file = None
    if source == "environment":
        for name, value in values.items():
            monkeypatch.setenv(name, value)
    else:
        env_file = tmp_path / ".env"
        env_file.write_text("\n".join(f"{name}={value}" for name, value in values.items()))
    config = EnvConfig(_env_file=env_file)
    credential = config.aws_bearer_token_bedrock
    assert isinstance(credential, SecretStr)
    assert credential.get_secret_value() == "synthetic-0"
    assert str(credential) == "**********"
    assert config.model_dump()["aws_bearer_token_bedrock"] is credential


@pytest.mark.parametrize("field", ["aws_bearer_token_bedrock", "BEDROCK_API_KEY"])
def test_bedrock_constructor_credential_overrides_environment(field, monkeypatch):
    """Ensures explicit settings construction keeps precedence over either environment name."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-aws")
    monkeypatch.setenv("BEDROCK_API_KEY", "synthetic-alias")
    config = EnvConfig(_env_file=None, **{field: "synthetic-explicit"})
    assert config.aws_bearer_token_bedrock.get_secret_value() == "synthetic-explicit"


@pytest.mark.parametrize(
    ("environment_name", "dotenv_name", "expected"),
    [
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_BEARER_TOKEN_BEDROCK", "synthetic-environment"),
        ("BEDROCK_API_KEY", "BEDROCK_API_KEY", "synthetic-environment"),
        ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_API_KEY", "synthetic-environment"),
        ("BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "synthetic-file"),
    ],
)
def test_bedrock_alias_and_source_precedence(
    environment_name, dotenv_name, expected, monkeypatch, tmp_path
):
    """Ensures the AWS name takes priority while same-name environment values override dotenv."""
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(environment_name, "synthetic-environment")
    env_file = tmp_path / ".env"
    env_file.write_text(f"{dotenv_name}=synthetic-file\n")
    config = EnvConfig(_env_file=env_file)
    assert config.aws_bearer_token_bedrock.get_secret_value() == expected


def cassette(protocol, stream):
    mode = "stream" if stream else "non_stream"
    path = Path(__file__).parent / "cassettes/matched/simple_text"
    provider = "anthropic" if protocol == "messages" else "openai"
    return read_yaml(path / f"{provider}-{protocol.replace('/', '_')}-{mode}-00.yaml")


def response(recording):
    recorded = recording["response"]
    if "sse" in recorded:
        return httpx.Response(
            200, content="".join(recorded["sse"]), headers={"content-type": "text/event-stream"}
        )
    return httpx.Response(200, json=recorded["body"])


@pytest.mark.parametrize(("provider", "endpoint", "base", "path"), DESTINATIONS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("override", [False, True])
async def test_hosted_native_destination_auth_and_sdk_parity(
    provider, endpoint, base, path, stream, override, monkeypatch
):
    """Ensures configured keys/default URLs and explicit overrides retain native SDK fidelity."""
    monkeypatch.setattr(
        ENV_CONFIG,
        CLIENTS[provider, endpoint].REQUIRED_API_KEY.lower(),
        SecretStr("configured-key"),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated")
    monkeypatch.setenv("OPENAI_ORG_ID", "unrelated")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated")
    base = "https://custom.example/prefix" if override else base
    key = "override-key" if override else "configured-key"
    recording = cassette(endpoint, stream)
    body = deepcopy(recording["request"]["body"])
    body["stream"] = stream
    expected_body = deepcopy(body)
    if provider in {"gemini", "fireworks"} and endpoint == "chat/completions":
        expected_body.get("stream_options", {}).pop("include_obfuscation", None)
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == base + path
        assert request.method == "POST"
        assert json_loads(request.content) == expected_body
        if endpoint == "messages":
            assert request.headers["x-api-key"] == key
            assert "authorization" not in request.headers
        else:
            assert request.headers["authorization"] == f"Bearer {key}"
            assert "x-api-key" not in request.headers
        assert "openai-organization" not in request.headers
        assert "openai-project" not in request.headers
        return response(recording)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        async with Molto(http_client=transport) as molto:
            actual = await getattr(molto, endpoint.replace("/", "_"))(
                body,
                endpoint=endpoint,
                provider=provider,
                base_url=base if override or provider == "bedrock" else None,
                api_key=key if override else None,
                extra_headers={"Authorization": "inbound", "x-api-key": "inbound"},
            )
            if stream:
                actual = [event.model_dump() async for event in actual]
            else:
                actual = actual.model_dump()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response(recording))
    ) as transport:
        sdk = (
            AsyncAnthropic(api_key="sdk-key", http_client=transport)
            if endpoint == "messages"
            else AsyncOpenAI(api_key="sdk-key", http_client=transport)
        )
        create = (
            sdk.messages.create
            if endpoint == "messages"
            else sdk.responses.create
            if endpoint == "responses"
            else sdk.chat.completions.create
        )
        expected = await create(**expected_body)
        if stream:
            async with expected:
                expected = [event.model_dump() async for event in expected]
        else:
            expected = expected.model_dump()
    assert actual == expected
    assert len(requests) == 1


@pytest.mark.parametrize("downstream", ["responses", "chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
async def test_bedrock_messages_translation_matches_openai_sdk(downstream, stream, monkeypatch):
    """Ensures Bedrock routing preserves matched Messages-to-OpenAI output in both modes."""
    # These matched protocol fixtures establish offline translation parity, not
    # live Bedrock acceptance of their synthetic model or response values.
    upstream = cassette("messages", True)
    reference = cassette(downstream, True)
    body = deepcopy(reference["request"]["body"])
    body["stream"] = stream
    if not stream:
        body.pop("stream_options", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-token")
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://bedrock.example/anthropic/v1/messages"
        assert json_loads(request.content) == upstream["request"]["body"]
        assert request.headers["x-api-key"] == "selected-key"
        assert "authorization" not in request.headers
        return response(upstream)

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as molto:
        actual = await getattr(molto, downstream.replace("/", "_"))(
            body,
            endpoint="messages",
            provider="bedrock",
            base_url="https://bedrock.example/anthropic",
            api_key="selected-key",
        )
        if stream:
            actual = [event async for event in actual]
            assert_stream_lifecycle(downstream, actual)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response(reference))
    ) as transport:
        sdk = AsyncOpenAI(api_key="sdk-key", http_client=transport)
        create = sdk.responses.create if downstream == "responses" else sdk.chat.completions.create
        expected = await create(**reference["request"]["body"])
        async with expected:
            expected = [event async for event in expected]

    if stream:
        if downstream == "responses":
            actual = [reconstructable_responses_event(dumps_sse(e), body=body) for e in actual]
            expected = [reconstructable_responses_event(dumps_sse(e), body=body) for e in expected]
        else:
            actual = [
                chunk
                for event in actual
                for chunk in reconstructable_chat_completions_chunks(dumps_sse(event), body=body)
            ]
            expected = [
                chunk
                for event in expected
                for chunk in reconstructable_chat_completions_chunks(dumps_sse(event), body=body)
            ]
            # Messages has no source timestamp to project onto Chat chunks.
            for chunk in actual + expected:
                chunk.pop("created", None)
        assert actual == expected
    else:
        expected = collect_sdk_response(downstream, expected)
        assert actual.id == expected.id
        assert actual.model == expected.model
        if downstream == "responses":
            assert actual.status == expected.status
            assert actual.incomplete_details == expected.incomplete_details
            actual_output = [item.model_dump(exclude={"id", "phase"}) for item in actual.output]
            expected_output = [
                item.model_dump(exclude={"id", "phase"}) for item in expected.output
            ]
            assert actual_output == expected_output
            assert actual.usage.model_dump() == expected.usage.model_dump()
        else:
            assert len(actual.choices) == len(expected.choices) == 1
            assert actual.choices[0].index == expected.choices[0].index
            assert actual.choices[0].finish_reason == expected.choices[0].finish_reason
            actual_message = actual.choices[0].message.model_dump(exclude={"parsed"})
            expected_message = expected.choices[0].message.model_dump(exclude={"parsed"})
            assert actual_message == expected_message
            assert actual.usage.prompt_tokens == expected.usage.prompt_tokens
            assert actual.usage.completion_tokens == expected.usage.completion_tokens
            assert actual.usage.total_tokens == expected.usage.total_tokens
            assert actual.usage.prompt_tokens_details.cached_tokens == (
                expected.usage.prompt_tokens_details.cached_tokens
            )
            assert actual.usage.completion_tokens_details.reasoning_tokens == (
                expected.usage.completion_tokens_details.reasoning_tokens
            )
    assert len(requests) == 1


@pytest.mark.parametrize(("provider", "endpoint", "base", "path"), DESTINATIONS)
@pytest.mark.parametrize("downstream", ["responses", "chat_completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_missing_provider_credentials_never_use_other_accounts(
    provider, endpoint, base, path, downstream, stream, monkeypatch
):
    """Ensures unrelated environment keys cannot authorize any hosted route or result mode."""
    monkeypatch.setattr(
        ENV_CONFIG, CLIENTS[provider, endpoint].REQUIRED_API_KEY.lower(), SecretStr("")
    )
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_ADMIN_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    ):
        monkeypatch.setenv(name, "unrelated-key")
    body = {
        "model": "example",
        "stream": stream,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    if downstream == "responses":
        body = {"model": "example", "stream": stream, "input": "Hi", "max_output_tokens": 20}

    def unexpected(request):
        pytest.fail("Missing provider credential reached HTTP")

    async with Molto(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(unexpected))
    ) as molto:
        with pytest.raises(Exception, match=CLIENTS[provider, endpoint].REQUIRED_API_KEY) as error:
            await getattr(molto, downstream)(
                body, endpoint=endpoint, provider=provider, base_url=base
            )
    assert type(error.value).__name__ == "AuthenticationError"
    assert str(error.value.request.url) == base + path


@pytest.mark.parametrize(("provider", "endpoint", "base", "path"), DESTINATIONS)
async def test_direct_hosted_client_defaults_and_missing_credentials(
    provider, endpoint, base, path, monkeypatch
):
    """Ensures direct clients own their URL and reject unrelated SDK credentials without the facade."""
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    async with httpx.AsyncClient() as transport:
        kwargs = {"capabilities": PROVIDER_CAPABILITIES[provider]}
        if provider == "bedrock":
            with pytest.raises(ValueError, match="explicit"):
                CLIENTS[provider, endpoint](transport, **kwargs)
            kwargs["base_url"] = base
        client = CLIENTS[provider, endpoint](transport, **kwargs)
        factory = "_make_anthropic_client" if endpoint == "messages" else "_make_sdk_client"
        error_type = AuthenticationError if endpoint == "messages" else OpenAIError
        with pytest.raises(error_type, match=CLIENTS[provider, endpoint].REQUIRED_API_KEY):
            getattr(client, factory)()
        selected = CLIENTS[provider, endpoint](transport, api_key="selected-key", **kwargs)
        sdk = getattr(selected, factory)()
        assert str(sdk.base_url).rstrip("/") == base
        if endpoint == "messages":
            assert sdk.default_headers["X-Api-Key"] == "selected-key"
            assert sdk.default_headers.get("Authorization", omit) is omit
        else:
            assert sdk.default_headers["Authorization"] == "Bearer selected-key"


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        (provider, endpoint)
        for provider in PROVIDER_CAPABILITIES
        for endpoint in ("responses", "chat/completions", "messages")
        if (provider, endpoint) not in CLIENTS
    ],
)
@pytest.mark.parametrize("downstream", ["responses", "chat_completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("base_url", [None, "https://custom.example/prefix"])
async def test_unregistered_routes_rejected_before_http(
    provider, endpoint, downstream, stream, base_url
):
    """Ensures native and translated routes require registration even with an explicit URL."""
    body = {"model": "example", "stream": stream}
    if downstream == "responses":
        body["input"] = "Hi"
    else:
        body["messages"] = [{"role": "user", "content": "Hi"}]
        if downstream == "messages":
            body["max_tokens"] = 20
    before = deepcopy(body)
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Unregistered provider protocol reached HTTP")
            )
        )
    ) as client:
        with pytest.raises(ValueError, match="Unsupported upstream endpoint") as error:
            await getattr(client, downstream)(
                body,
                provider=provider,
                endpoint=endpoint,
                api_key="selected-key",
                base_url=base_url,
            )
    assert repr(provider) in str(error.value)
    assert repr(endpoint) in str(error.value)
    assert body == before


async def test_native_bedrock_messages_requires_explicit_destination():
    """Ensures Bedrock Messages cannot send its credential to a default Anthropic URL."""
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Missing Bedrock destination reached HTTP")
            )
        )
    ) as client:
        with pytest.raises(ValueError, match="explicit base_url"):
            await client.messages(
                {"model": "test", "messages": [], "max_tokens": 20},
                provider="bedrock",
                api_key="selected-key",
            )


def tool_chunk(calls, finish=None):
    return {
        "id": "chat-1",
        "model": "example",
        "created": 1,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": finish}],
    }


def gemini_http(events):
    data = "".join(dumps_sse(event) for event in events) + "data: [DONE]\n\n"
    return httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
        )
    )


def call_start(call_id, arguments="{", **fields):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "add", "arguments": arguments},
        **fields,
    }


@pytest.mark.parametrize(
    ("downstream", "stream"),
    [("chat_completions", True), ("responses", True), ("responses", False)],
)
@pytest.mark.parametrize("parallel", [False, True])
async def test_gemini_missing_indices_preserve_distinct_calls(downstream, stream, parallel):
    """Ensures repaired single and contiguous parallel calls finish with stable IDs and arguments."""
    events = [
        tool_chunk([call_start("call-a", index=None)]),
        tool_chunk([{"id": "call-a", "function": {"arguments": '"a":1}'}}]),
    ]
    if parallel:
        events.extend(
            [
                tool_chunk([call_start("call-b")]),
                tool_chunk([{"id": "call-b", "function": {"arguments": '"a":2}'}}]),
            ]
        )
    events.append(tool_chunk([], "tool_calls"))
    async with Molto(http_client=gemini_http(events)) as molto:
        body = {"model": "example", "stream": stream}
        body.update(
            {"input": "Hi"}
            if downstream == "responses"
            else {"messages": [{"role": "user", "content": "Hi"}]}
        )
        result = await getattr(molto, downstream)(
            body, endpoint="chat/completions", provider="gemini", api_key="key"
        )
        if stream:
            output = [event async for event in result]
        if downstream == "chat_completions":
            state = ChatCompletionStreamState()
            for event in output:
                state.handle_chunk(event)
            calls = state.get_final_completion().choices[0].message.tool_calls
            actual = [(c.id, c.function.arguments) for c in calls]
        else:
            response = output[-1].response if stream else result
            actual = [(item.call_id, item.arguments) for item in response.output]
            if stream:
                added = [
                    event.item.call_id
                    for event in output
                    if event.type == "response.output_item.added"
                ]
                done = [
                    event.item.call_id
                    for event in output
                    if event.type == "response.output_item.done"
                ]
                assert added == done == [call_id for call_id, _ in actual]
    assert actual == [("call-a", '{"a":1}')] + ([("call-b", '{"a":2}')] if parallel else [])


async def test_gemini_native_interleaved_calls_use_ids_not_array_positions():
    """Ensures native accumulation preserves parallel identities when fragment order changes."""
    first = [call_start("call-a"), call_start("call-b")]
    for call in first:
        call["extra_content"] = {"google": {"thought_signature": f"state-{call['id']}"}}
    events = [
        tool_chunk(first),
        tool_chunk(
            [
                {"id": "call-b", "function": {"arguments": '"a":2}'}},
                {"id": "call-a", "function": {"arguments": '"a":1}'}},
            ]
        ),
        tool_chunk([], "tool_calls"),
    ]
    original = deepcopy(events)
    async with Molto(http_client=gemini_http(events)) as molto:
        output = await molto.chat_completions(
            {"model": "example", "stream": True, "messages": [{"role": "user", "content": "Hi"}]},
            provider="gemini",
            api_key="key",
        )
        state = ChatCompletionStreamState()
        async for event in output:
            state.handle_chunk(event)
    calls = state.get_final_completion().choices[0].message.tool_calls
    assert [(c.id, c.function.name, c.function.arguments) for c in calls] == [
        ("call-a", "add", '{"a":1}'),
        ("call-b", "add", '{"a":2}'),
    ]
    assert [c.extra_content for c in calls] == [
        {"google": {"thought_signature": "state-call-a"}},
        {"google": {"thought_signature": "state-call-b"}},
    ]
    assert events == original


@pytest.mark.parametrize("stream", [False, True])
async def test_gemini_translated_noncontiguous_calls_reject(stream):
    """Ensures inferred indices do not bypass the translated stream's contiguous-call contract."""
    events = [
        tool_chunk([call_start("call-a"), call_start("call-b")]),
        tool_chunk([{"id": "call-a", "function": {"arguments": "}"}}]),
    ]
    async with Molto(http_client=gemini_http(events)) as molto:
        with pytest.raises(ValueError, match="contiguous"):
            result = await molto.responses(
                {"model": "example", "input": "Hi", "stream": stream},
                endpoint="chat/completions",
                provider="gemini",
                api_key="key",
            )
            if stream:
                [event async for event in result]


async def test_gemini_unidentified_call_rejects():
    """Ensures a call lacking both an index and identity cannot be repaired."""
    calls = [{"index": None, "function": {"arguments": "{}"}}]
    async with Molto(http_client=gemini_http([tool_chunk(calls)])) as molto:
        with pytest.raises(ValueError, match="unambiguous"):
            result = await molto.responses(
                {"model": "example", "input": "Hi", "stream": True},
                endpoint="chat/completions",
                provider="gemini",
                api_key="key",
            )
            [event async for event in result]


async def test_gemini_anonymous_fragment_after_parallel_calls_rejects():
    """Ensures a fragment with several possible owners never defaults to the latest or zero index."""
    events = [
        tool_chunk([call_start("call-a"), call_start("call-b")]),
        tool_chunk([{"function": {"arguments": "}"}}]),
    ]
    async with Molto(http_client=gemini_http(events)) as molto:
        with pytest.raises(ValueError, match="unambiguous"):
            result = await molto.chat_completions(
                {
                    "model": "example",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                provider="gemini",
                api_key="key",
            )
            [event async for event in result]


async def test_gemini_anonymous_single_call_continuation_is_unambiguous():
    """Ensures an anonymous argument-only fragment can continue the sole established call."""
    events = [
        tool_chunk([call_start("call-a")]),
        tool_chunk([{"function": {"arguments": '"a":1}'}}]),
        tool_chunk([], "tool_calls"),
    ]
    async with Molto(http_client=gemini_http(events)) as molto:
        result = await molto.responses(
            {"model": "example", "input": "Hi", "stream": False},
            endpoint="chat/completions",
            provider="gemini",
            api_key="key",
        )
    assert [(item.call_id, item.arguments) for item in result.output] == [("call-a", '{"a":1}')]


@pytest.mark.parametrize(
    "calls",
    [
        [call_start("call-a", index=0)],
        [call_start("call-a", index=-1)],
        [call_start("call-a", index="invalid")],
        [call_start("call-a", index=0), call_start("call-b", index=0)],
        [call_start("call-a", index=0), call_start("call-a", index=1)],
    ],
)
async def test_gemini_indexed_chunks_retain_native_fidelity(calls):
    """Ensures already indexed output retains SDK parity without Molto identity validation."""
    events = [
        tool_chunk(calls),
        tool_chunk([{"id": "call-a", "index": 0, "function": {"arguments": "}"}}]),
        tool_chunk([], "tool_calls"),
    ]
    async with Molto(http_client=gemini_http(events)) as molto:
        result = await molto.chat_completions(
            {"model": "example", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
            provider="gemini",
            api_key="key",
        )
        actual = [event.model_dump() async for event in result]
    async with AsyncOpenAI(
        api_key="key", base_url="https://example.test", http_client=gemini_http(events)
    ) as sdk:
        reference = await sdk.chat.completions.create(
            model="example", messages=[{"role": "user", "content": "Hi"}], stream=True
        )
        async with reference:
            expected = [event.model_dump() async for event in reference]
    assert actual == expected
