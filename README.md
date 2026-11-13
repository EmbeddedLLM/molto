# Molto

[![Lint](https://github.com/EmbeddedLLM/molto/actions/workflows/lint.yml/badge.svg?branch=main&event=push)](https://github.com/EmbeddedLLM/molto/actions/workflows/lint.yml?query=branch%3Amain+event%3Apush) [![CI](https://github.com/EmbeddedLLM/molto/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/EmbeddedLLM/molto/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush) [![OpenAI schema](https://github.com/EmbeddedLLM/molto/actions/workflows/weekly-provider-tests.yml/badge.svg?branch=main&event=schedule)](https://github.com/EmbeddedLLM/molto/actions/workflows/weekly-provider-tests.yml?query=branch%3Amain+event%3Aschedule)

A Python library for translating between LLM API protocols without giving up native SDK compatibility.

## Design goals

- **Any protocol to any protocol.** Accept any supported downstream LLM protocol and translate it to any supported upstream LLM protocol through a provider-neutral representation.
- **Native when no translation is needed.** When the downstream and upstream protocols match, Molto preserves request dictionaries, including unknown properties and explicit nulls, and leaves request validation to the SDK and provider. Returned events remain native SDK classes and objects.
- **Strict translation inputs, lenient outputs.** Cross-protocol requests are closed and lossless-or-rejected, with documented [cache-lifetime approximations](docs/README.md#prompt-caching). Normalized outputs tolerate provider extensions; retaining unknown output data does not authorize projecting it into another protocol.

Start with the [Molto documentation](docs/README.md) for protocol guides, translation guarantees, and current support. See [Molto Architecture](python/ARCHITECTURE.md) for the internal translation and native-passthrough design.

## Quickstart

Use the method for the protocol your application speaks and select the upstream protocol with `endpoint`. With `ANTHROPIC_API_KEY` set, this accepts an OpenAI Responses request and sends it to an Anthropic Messages endpoint:

```python
import asyncio

from molto import Molto


async def main() -> None:
    # Using context managers
    async with Molto() as molto:
        async with await molto.responses(
            body={
                "model": "claude-sonnet-5",
                "input": "Hello!",
                "max_output_tokens": 1024,
                "stream": True,
            },
            endpoint="messages",
        ) as stream:
            async for event in stream:
                print(event)  # Native OpenAI Responses event

    # Manual cleanup
    molto = Molto()
    try:
        stream = await molto.responses(
            body={
                "model": "claude-sonnet-5",
                "input": "Hello!",
                "max_output_tokens": 1024,
                "stream": True,
            },
            endpoint="messages",
        )
        try:
            async for event in stream:
                print(event)  # Native OpenAI Responses event
        finally:
            await stream.aclose()
    finally:
        await molto.aclose()


asyncio.run(main())
```

- `endpoint` selects the upstream wire protocol.
- `provider` (optional) selects the serving implementation and its capabilities.
- `base_url` (optional) selects only the HTTP destination.
- `extra_body` (optional) merges provider extensions into the projected upstream body.
- `extra_headers` (optional) adds upstream headers after Molto removes client credentials and transport-specific fields.

These settings are independent. Use the same protocol for the method and `endpoint` to get native SDK passthrough. The returned `MoltoStream.response_headers` retains successful upstream HTTP headers for callers such as proxy servers.

## Error handling

Molto exposes recognized errors in the downstream SDK namespace, with a normalized Molto cause for portable classification. HTTP errors preserve their upstream status and body.

For example, an Anthropic missing-model `404` becomes an OpenAI `NotFoundError` with its original Anthropic body:

```python
import openai

from molto.types.exceptions import ModelNotFoundError

try:
    async for event in stream:
        ...
except openai.APIStatusError as error:
    if isinstance(error.__cause__, ModelNotFoundError):
        print("The upstream provider does not recognize this model.")
```

Unknown errors pass through unchanged. See the [shared error contract](docs/errors.md) for HTTP mappings, Responses failures during streaming, and the deliberate synthetic `401` for missing SDK credentials.

## Test coverage

CI runs the offline replay suite on multiple Python versions and posts a coverage comment for each version on pull requests. Each comment includes test results and coverage details for changed files, and is updated on subsequent runs. Comment posting is best-effort because fork pull requests may have read-only tokens; a posting failure does not hide test failures.

After installing `.[test]`, generate the same reports locally with:

```bash
python -m pytest --cov=python/molto --cov-report=term-missing \
  --cov-report=xml:coverage.xml --junitxml=junit/pytest.xml
```

## OpenAI schema monitoring

The [OpenAI OpenAPI compatibility workflow](.github/workflows/weekly-provider-tests.yml) runs weekly on Tuesdays at 09:00 UTC from the default branch. It compares a pinned OpenAPI snapshot against upstream `main` for Chat Completions and Responses. It needs no API credentials and makes no model requests.

The check reports programmatic schema changes, including additions, while ignoring documentation changes. A failure signals upstream drift to review; it does not by itself prove a runtime regression. Update `BASELINE_OPENAPI_URL` to a new immutable commit only after reviewing the differences and verifying affected Molto behavior. Advancing this reference acknowledges existing drift without changing SDK dependencies or adding support for new API features. Offline cassette tests remain a separate CI check.
