# Molto

A Python library for translating between LLM API protocols without giving up native SDK compatibility.

## Design goals

- **Any protocol to any protocol.** Accept any supported downstream LLM protocol and translate it to any supported upstream LLM protocol through a provider-neutral representation.
- **Native when no translation is needed.** When the downstream and upstream protocols match, Molto bypasses translation and acts as a thin wrapper around the corresponding native SDK. All native features and events remain available, and results are returned as native SDK classes and objects.

See [Molto Architecture](python/ARCHITECTURE.md) for the translation and native-passthrough design.

## Quickstart

Use the method for the protocol your application speaks and select the upstream protocol with `endpoint`. Inside an async function, with `ANTHROPIC_API_KEY` set, this accepts an OpenAI Responses request and sends it to an Anthropic Messages endpoint:

```python
from molto import Molto

molto = Molto()
stream = await molto.responses(
    body={
        "model": "claude-sonnet-5",
        "input": "Hello!",
        "max_output_tokens": 1024,
        "stream": True,
    },
    endpoint="messages",
    provider="anthropic",
)

async for event in stream:
    print(event)  # Native OpenAI Responses event

await molto.close()
```

`endpoint` selects the upstream wire protocol, `provider` selects the serving implementation and its capabilities, and `base_url` selects only the HTTP destination. These settings are independent. Use the same protocol for the method and `endpoint` to get native SDK passthrough.
