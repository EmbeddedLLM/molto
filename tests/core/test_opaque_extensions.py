"""Opaque upstream extensions must not participate in protocol translation."""

from copy import deepcopy

import pytest
from openai import AsyncOpenAI

from molto import Molto
from tests.core.test_interface import (
    PROTOCOLS,
    Protocol,
    _coordinator,
    _get_http_client,
    _matched_cassette_path,
    _read_cassette,
)
from tests.core.test_web_search import _assert_lifecycle, _collect


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True])
async def test_opaque_extensions_preserve_sdk_output(upstream: Protocol, stream: bool):
    """Ensures unvalidated extensions reach transport unchanged without altering SDK output parity."""
    downstream = "messages" if upstream == "responses" else "responses"
    downstream_path = _matched_cassette_path("simple_text", downstream, "stream-00.yaml")
    upstream_path = _matched_cassette_path("simple_text", upstream, "stream-00.yaml")
    body = deepcopy(_read_cassette(downstream_path)["request"]["body"])
    body["stream"] = stream
    extra_body = {"tools": [{"name": "provider_specific_tool"}], "include": ["provider.detail"]}
    if upstream == "responses":
        # Nonstandard provider input must remain opaque even when it is not a
        # Responses history dictionary and cannot be inspected for citations.
        extra_body["input"] = ["provider-specific input"]
    original = deepcopy(extra_body)
    async with _get_http_client(
        [upstream_path], upstream, expected_extra_body=extra_body
    ) as http_client:
        result = await _coordinator(Molto(http_client=http_client), downstream)(
            body=body,
            endpoint=upstream,
            api_key="test-api-key",
            extra_body=extra_body,
        )
        if stream:
            events = [event async for event in result]
            _assert_lifecycle(downstream, events)
            result = _collect(downstream, events)
    assert extra_body == original

    # Replay the unchanged matched Responses recording through the SDK; the
    # synthetic extensions test forwarding, not provider acceptance of them.
    reference_path = _matched_cassette_path("simple_text", "responses", "stream-00.yaml")
    reference_body = _read_cassette(reference_path)["request"]["body"]
    async with _get_http_client([reference_path]) as http_client:
        sdk = AsyncOpenAI(api_key="test-api-key", http_client=http_client)
        async with await sdk.responses.create(**reference_body) as reference_stream:
            reference = _collect("responses", [event async for event in reference_stream])

    text = (
        "".join(block.text for block in result.content if block.type == "text")
        if downstream == "messages"
        else result.output_text
    )
    assert text == reference.output_text
    assert result.id == reference.id
    assert result.model == reference.model
    assert result.usage.input_tokens == reference.usage.input_tokens
    assert result.usage.output_tokens == reference.usage.output_tokens
