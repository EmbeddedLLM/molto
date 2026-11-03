"""SDK parity for unchanged historical Together error captures from PR #76."""

from copy import deepcopy

import pytest
from openai import AsyncOpenAI
from openai import AuthenticationError as SDKAuthenticationError
from openai import BadRequestError as SDKBadRequestError
from openai import NotFoundError as SDKNotFoundError

from molto import Molto
from molto.utils.io import json_loads, read_yaml
from tests.core.test_exceptions import (
    CASSETTE_DIR,
    DOWNSTREAM_ERROR_TYPES,
    DOWNSTREAM_STATUS_ERROR_TYPES,
    ERROR_SCENARIOS,
    PROTOCOLS,
    _get_http_client,
    _read_cassette,
    scenario_ids,
)


@pytest.mark.parametrize("scenario", ERROR_SCENARIOS, ids=scenario_ids(ERROR_SCENARIOS))
@pytest.mark.parametrize("downstream", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True])
async def test_together_error_cassette_sdk_parity(scenario, downstream, stream):
    """Ensures all public routes preserve recorded SDK errors and exact executed requests."""
    mode = "stream" if stream else "non_stream"
    cassette = read_yaml(
        CASSETTE_DIR / scenario.name / f"togetherai-chat_completions-{mode}-00.yaml"
    )
    status = cassette["response"]["status_code"]
    sdk_error_type = {
        400: SDKBadRequestError,
        401: SDKAuthenticationError,
        404: SDKNotFoundError,
    }[status]
    native_body = deepcopy(cassette["request"]["body"])
    extra_body = {"reasoning": native_body.pop("reasoning")}
    sdk_transport, sdk_requests = _get_http_client(cassette)
    async with sdk_transport:
        sdk = AsyncOpenAI(
            http_client=sdk_transport,
            api_key="test-api-key",
            base_url="https://api.together.ai/v1",
            max_retries=0,
        )
        with pytest.raises(sdk_error_type) as reference:
            await sdk.chat.completions.create(**native_body, extra_body=extra_body)
    assert len(sdk_requests) == 1
    assert json_loads(sdk_requests[0].content) == cassette["request"]["body"]

    translated = downstream != "chat/completions"
    body = (
        deepcopy(_read_cassette(scenario.name, downstream, stream=stream)["request"]["body"])
        if translated
        else native_body
    )
    expected_body = deepcopy(cassette["request"]["body"])
    if translated and not stream:
        expected_body["stream"] = True
        expected_body.setdefault("stream_options", {})["include_usage"] = True
        if downstream == "messages":
            expected_body["stream_options"]["include_obfuscation"] = False

    expected_error_type = DOWNSTREAM_STATUS_ERROR_TYPES[downstream].get(
        status, DOWNSTREAM_ERROR_TYPES[downstream]
    )
    transport, requests = _get_http_client(cassette)
    async with transport:
        client = Molto(http_client=transport)
        coordinator = getattr(client, downstream.replace("/", "_"))
        with pytest.raises(expected_error_type) as result:
            await coordinator(
                body=body,
                endpoint="chat/completions",
                provider="togetherai",
                api_key="test-api-key",
                extra_body=extra_body,
            )
        error = result.value
        normalized = error.__cause__
        original = normalized.__cause__
        assert type(normalized) is scenario.normalized_error
        assert type(original) is sdk_error_type
        assert error is not original
        assert original.__cause__ is None
        assert error.message == normalized.message == original.message == reference.value.message
        assert error.body is normalized.body is original.body
        assert error.body == reference.value.body
        assert error.response is normalized.response is original.response
        assert error.response.status_code == status
        assert error.response.headers == reference.value.response.headers
        assert error.response.content == reference.value.response.content
        assert error.response.is_closed
        assert error.request is normalized.request is original.request is requests[0]
        assert not transport.is_closed
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://api.together.ai/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-api-key"
    assert json_loads(requests[0].content) == expected_body
