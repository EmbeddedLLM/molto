# Molto Architecture

## Overview

Molto is a provider-agnostic gateway that accepts requests in various API formats (OpenAI Responses, OpenAI Chat Completion, etc.) and routes them to LLM providers via [pydantic-ai](https://ai.pydantic.dev/).

## Data Flow

```
  API Request (OpenAI Responses, Chat Completion, ...)
       │
       ▼
  ┌──────────┐
  │  types/  │  Parse & validate the vendor-specific request.
  │          │  Convert it into the Intermediate Representation (IR).
  └────┬─────┘
       │  Input IR
       ▼
  ┌──────────────┐
  │ providers.py │  Accept input IR. Internally: map to pydantic-ai
  │              │  settings, create a Runner, call run/run_stream,
  │              │  convert the response back into output IR.
  └────┬─────────┘
       │  Output IR stream
       ▼
  ┌──────────┐
  │  types/  │  Convert the IR response stream back into the
  │          │  vendor-specific response format.
  └──────────┘
```

### Step 1 — Types (request path): vendor request → IR

Each vendor format lives under `types/<vendor>/` (e.g. `types/openai/responses.py`).
The type modules are responsible for:

- Pydantic models that mirror the vendor's API schema (request, response, streaming events).
- Validation logic specific to that vendor format.
- A method (e.g. `as_ir()`) that converts the validated request into the **Intermediate Representation**.

The IR is a provider-agnostic bundle of everything needed to run an Agent.

**Key rule:** type modules must NOT import provider-specific pydantic-ai settings (e.g. `OpenAIChatModelSettings`). They produce only the common IR. This keeps the `types/` layer decoupled from any single provider.

### Step 2 — Runner: pydantic-ai wrapper

`runner.py` is a thin wrapper around pydantic-ai's `Agent`. It is instantiated with the agent, model, and provider-specific settings, then exposes two methods:

```python
# runner.py  (sketch)

class Runner:
     def __init__(self, agent: Agent, model: Model, settings: ModelSettings):
          ...

     async def run(self, ...) -> RunResult:
          """Non-streaming execution."""
          ...

     async def run_stream(self, ...) -> AsyncIterator[StreamEvent]:
          """Streaming execution."""
          ...
```

The Runner is the only module that directly interacts with the pydantic-ai execution API. It can be swapped or mocked independently (e.g. for testing or recording).

### Step 3 — Providers: input IR → output IR

`providers.py` contains one class per provider. Each provider exposes `run` and `run_stream` that accept input IR and return output IR.

```python
# providers.py  (sketch)

class OpenAIProvider:
     async def run(ir: InputIR) -> OutputIR:
          settings = self._map_model_settings(ir.model_settings)
          runner = Runner(agent, model, settings)
          result = await runner.run(...)
          return self._map_response(result)

     async def run_stream(ir: InputIR) -> AsyncIterator[OutputIR]:
          settings = self._map_model_settings(ir.model_settings)
          runner = Runner(agent, model, settings)
          async for event in runner.run_stream(...):
               yield self._map_stream_event(event)


class AnthropicProvider:
     async def run(ir: InputIR) -> OutputIR:
          ...

     async def run_stream(ir: InputIR) -> AsyncIterator[OutputIR]:
          ...
```

Internally each provider:
1. Maps IR model settings to provider-specific pydantic-ai settings (eg: field renaming like `max_tokens` → `max_output_tokens`, provider-specific features like `openai_reasoning_effort`, omitting unsupported fields).
2. Creates a Runner with the appropriate agent, model, and settings.
3. Converts the pydantic-ai response/stream events back into provider-agnostic output IR.

### Step 4 — Types (response path): IR → vendor response

The same `types/<vendor>/` modules that parsed the request also convert the output IR stream back into the vendor-specific response format:

- For streaming: each output IR event is mapped to the appropriate vendor streaming event (e.g. `response.output_text.delta` for OpenAI Responses).
- For non-streaming: the full output IR is assembled into the vendor's response object (e.g. `OpenAIResponsesResponse`).

**Key rule:** type modules consume and produce only the common IR — they never interact with pydantic-ai types directly.

### Step 5 — Core: orchestrate

`core.py` (`Molto.chat()`) ties it together:

1. Call the type's `as_ir()` to get the input IR.
2. Select the provider based on the model identifier (or explicit config).
3. Call the provider's `run()` or `run_stream()` to get output IR.
4. Pass the output IR to the type's response conversion to produce the vendor-specific response.

## Migration Checklist (current → target)

**Input IR & request path:**
- [ ] Define the input IR (provider-agnostic sampling params, message history, tools, etc.).
- [ ] Refactor `as_run_settings` to return the input IR instead of `OpenAIChatModelSettings`. Rename to `as_ir` or similar.
- [ ] Remove the `from pydantic_ai.models.openai import OpenAIChatModelSettings` import from `types/openai/responses.py`.

**Output IR & response path:**
- [ ] Define the output IR (provider-agnostic response events for both streaming and non-streaming).
- [ ] Add response conversion methods to `types/openai/` that accept output IR and produce vendor-specific responses and streaming events.

**Runner:**
- [ ] Implement `runner.py` with a `Runner` class that wraps pydantic-ai Agent instantiation and exposes `run` / `run_stream`.

**Providers:**
- [ ] Move `as_openai_chat_settings` from `vLLMResponsesRequest` into the OpenAI provider as an internal settings mapper.
- [ ] Implement provider `run` / `run_stream` methods (input IR → Runner → output IR).
- [ ] Add provider implementations for additional backends (Anthropic, vLLM, etc.).

**Core:**
- [ ] Wire the full pipeline into `Molto.chat()`: types → provider → types.
