# Molto

A Python library for translating between LLM API protocols without giving up native SDK compatibility.

## Design goals

- **Any protocol to any protocol.** Accept any supported downstream LLM protocol and translate it to any supported upstream LLM protocol through a provider-neutral representation.
- **Native when no translation is needed.** When the downstream and upstream protocols match, Molto bypasses translation via normalized representation and acts as a thin wrapper around the corresponding native SDK. Requests are still subject to Molto's validated public contract, while returned events remain native SDK classes and objects.
- **Strict inputs, lenient outputs.** Cross-protocol requests are closed and lossless-or-rejected, while normalized outputs tolerate provider extensions. Retaining unknown output data does not authorize projecting it into another protocol.

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

On every route, Molto recognizes selected provider errors and raises them in the downstream SDK namespace. This translation is semantic, not a rewrite of the provider's HTTP response: the status code, message, response, and body still come from the upstream protocol.

For example, Anthropic reports a missing model with status `404`, while OpenAI Responses reports it with status `400`. An Anthropic Messages upstream routed through the OpenAI Responses downstream therefore raises an OpenAI `APIStatusError` whose status remains `404` and whose body remains Anthropic-shaped.

The translated exception's cause identifies the normalized error more precisely:

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

Normalized API errors inherit from `APIException` and retain the original upstream SDK exception as their cause. Native passthrough routes receive the same normalized cause, so classification works identically for every upstream; only unrecognized errors do not receive it. See [Error translation](python/ARCHITECTURE.md#error-translation) for the complete mapping and cause chain.
