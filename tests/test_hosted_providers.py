"""Offline destination/authentication probes; these do not certify hosted models."""

from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI, OpenAIError
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import SecretStr

from molto import Molto
from molto.configs import ENV_CONFIG
from molto.providers import PROVIDER_CAPABILITIES, PROVIDER_KEY_NAMES
from molto.upstream.registry import CLIENTS
from molto.utils.io import dumps_sse, json_loads, read_yaml

DESTINATIONS = [
    ("fireworks", "responses", "https://api.fireworks.ai/inference/v1", "/responses"),
    ("openrouter", "responses", "https://openrouter.ai/api/v1", "/responses"),
    ("togetherai", "responses", "https://api.together.ai/v1", "/responses"),
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
]


def cassette(protocol, stream):
    mode = "stream" if stream else "non_stream"
    path = Path(__file__).parent / "cassettes/matched/simple_text"
    return read_yaml(path / f"openai-{protocol.replace('/', '_')}-{mode}-00.yaml")


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
        ENV_CONFIG, PROVIDER_KEY_NAMES[provider].lower(), SecretStr("configured-key")
    )
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated")
    monkeypatch.setenv("OPENAI_ORG_ID", "unrelated")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated")
    base = "https://custom.example/prefix" if override else base
    key = "override-key" if override else "configured-key"
    recording = cassette(endpoint, stream)
    body = deepcopy(recording["request"]["body"])
    body["stream"] = stream
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == base + path
        assert request.method == "POST"
        assert json_loads(request.content) == body
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
        sdk = AsyncOpenAI(api_key="sdk-key", http_client=transport)
        expected = await (
            sdk.responses.create if endpoint == "responses" else sdk.chat.completions.create
        )(**body)
        if stream:
            async with expected:
                expected = [event.model_dump() async for event in expected]
        else:
            expected = expected.model_dump()
    assert actual == expected
    assert len(requests) == 1


@pytest.mark.parametrize(("provider", "endpoint", "base", "path"), DESTINATIONS)
@pytest.mark.parametrize("downstream", ["responses", "chat_completions", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_missing_provider_credentials_never_use_other_accounts(
    provider, endpoint, base, path, downstream, stream, monkeypatch
):
    """Ensures unrelated environment keys cannot authorize any hosted route or result mode."""
    monkeypatch.setattr(ENV_CONFIG, PROVIDER_KEY_NAMES[provider].lower(), SecretStr(""))
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
        with pytest.raises(Exception, match=PROVIDER_KEY_NAMES[provider]) as error:
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
        with pytest.raises(OpenAIError, match=PROVIDER_KEY_NAMES[provider]):
            client._make_sdk_client()
        selected = CLIENTS[provider, endpoint](transport, api_key="selected-key", **kwargs)
        sdk = selected._make_sdk_client()
        assert str(sdk.base_url).rstrip("/") == base
        assert sdk.default_headers["Authorization"] == "Bearer selected-key"


@pytest.mark.parametrize(("provider", "supported", "base", "path"), DESTINATIONS)
async def test_hosted_providers_reject_unselected_protocols(provider, supported, base, path):
    """Ensures a custom URL cannot silently enable an unsupported provider protocol."""
    async with Molto(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Unsupported provider protocol reached HTTP")
            )
        )
    ) as client:
        for endpoint in {"responses", "chat/completions", "messages"} - {supported}:
            with pytest.raises(ValueError):
                await client.responses(
                    {"model": "example", "input": "Hi"},
                    provider=provider,
                    endpoint=endpoint,
                    api_key="selected-key",
                    base_url=base,
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


@pytest.mark.parametrize("mixed_indices", [False, True])
async def test_gemini_native_interleaved_calls_use_ids_not_array_positions(mixed_indices):
    """Ensures native accumulation preserves parallel identities when fragment order changes."""
    first = [call_start("call-a"), call_start("call-b")]
    for call in first:
        call["extra_content"] = {"google": {"thought_signature": f"state-{call['id']}"}}
    if mixed_indices:
        first[1]["index"] = 1
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


@pytest.mark.parametrize(
    "calls",
    [
        [{"index": None, "function": {"arguments": "{}"}}],
        [call_start("call-a", index=-1)],
        [call_start("call-a", index="invalid")],
        [call_start("call-a", index=0), call_start("call-b", index=0)],
        [call_start("call-a", index=0), call_start("call-a", index=1)],
    ],
)
async def test_gemini_ambiguous_or_conflicting_identity_rejects(calls):
    """Ensures malformed or conflicting identity fails before a fabricated successful completion."""
    async with Molto(http_client=gemini_http([tool_chunk(calls)])) as molto:
        with pytest.raises(ValueError, match="unambiguous|nonnegative|identity|index"):
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


async def test_gemini_indexed_chunks_retain_native_fidelity():
    """Ensures Gemini compatibility repair does not alter already indexed provider chunks."""
    from openai.types.chat import ChatCompletionChunk

    events = [
        tool_chunk([call_start("call-a", index=0)]),
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
    assert actual == [ChatCompletionChunk.model_validate(event).model_dump() for event in events]
