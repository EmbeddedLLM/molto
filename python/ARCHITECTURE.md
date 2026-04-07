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
  │          │  Type.to_ir() → Input IR
  └────┬─────┘
       │  Input IR
       ▼
  ┌──────────────┐
  │  providers/  │  Provider(body) — instantiated with vendor request.
  │              │  provider.run() / provider.run_stream()
  │              │  Internally: parse settings, create Runner,
  │              │  compose output via Type.from_ir().
  └────┬─────────┘
       │  Vendor-specific response
       ▼
  Client
```

### Step 1 — Types (request path): vendor request → IR

Each vendor format lives under `types/<vendor>/` (e.g. `types/openai/responses.py`).
The type modules are responsible for:

- Pydantic models that mirror the vendor's API schema (request, response, streaming events).
- Validation logic specific to that vendor format.
- `to_ir()` — converts the validated request into the **Input IR**.
- `from_ir()` — converts Output IR events back into vendor-specific response objects and streaming events.

The IR is a provider-agnostic bundle of everything needed to run an Agent (input side) and everything produced by a run (output side).

**Key rule:** type modules must NOT import provider-specific pydantic-ai settings (e.g. `OpenAIChatModelSettings`). They produce and consume only the common IR. This keeps the `types/` layer decoupled from any single provider.

### Step 2 — Providers: body → run / run_stream

Each provider is instantiated with the vendor-specific request `body` and exposes two public methods — `run` and `run_stream`:

```python
# providers.py  (sketch)

class _Provider:
     def __init__(self, body: VendorRequest):
          self.body = body

     async def run(self) -> VendorResponse:
          """Non-streaming execution."""
          self._parse_run_settings()
          self._parse_agent_settings()
          runner = Runner(self._run_settings, self._agent_settings)
          output_ir = await runner.run()
          return self._compose_output(output_ir)

     async def run_stream(self) -> AsyncIterator[VendorResponse]:
          """Streaming execution."""
          self._parse_run_settings()
          self._parse_agent_settings()
          runner = Runner(self._run_settings, self._agent_settings)
          async for event in runner.run_stream():
               yield self._compose_output(event)

     def _parse_run_settings(self) -> AgentRunSettings:
          """Parse the vendor request body into pydantic-ai run settings
          (message history, instructions, toolsets, usage limits)."""
          ...

     def _parse_agent_settings(self) -> ModelSettings:
          """Parse the vendor request body into provider-specific model settings
          (temperature, max_tokens → max_output_tokens, reasoning_effort, etc.)."""
          ...

     def _compose_output(self, output_ir: OutputIR) -> VendorResponse:
          """Convert Output IR to vendor-specific response format
          by calling Type.from_ir()."""
          ...
```

Internally each provider:
1. Is **instantiated with the vendor-specific request body**.
2. **`run`** / **`run_stream`** — the public interface. Orchestrates the full pipeline: parse settings, create Runner, stream output, compose response.
3. **`_parse_run_settings`** — parses the body into the agent run configuration (message history, instructions, toolsets, usage limits).
4. **`_parse_agent_settings`** — parses the body's sampling parameters into provider-specific pydantic-ai settings (e.g. field renaming like `max_tokens` → `max_output_tokens`, provider-specific features like `openai_reasoning_effort`, omitting unsupported fields).
5. **`_compose_output`** — takes Output IR from the Runner and produces the vendor-specific response by calling `Type.from_ir()`.

### Step 3 — Runner: pydantic-ai wrapper

`runner.py` is a thin wrapper around pydantic-ai's `Agent`. The provider instantiates it with the run settings and model settings, and the Runner streams **Output IR** back.

```python
# runner.py  (sketch)

class Runner:
     def __init__(self, agent: Agent, model: Model, settings: ModelSettings):
          ...

     async def run(self, ...) -> OutputIR:
          """Non-streaming execution. Returns Output IR."""
          ...

     async def run_stream(self, ...) -> AsyncIterator[OutputIR]:
          """Streaming execution. Yields Output IR events."""
          ...
```

The Runner is the only module that directly interacts with the pydantic-ai execution API. It normalises pydantic-ai's raw response/stream events into the Output IR before yielding them. It can be swapped or mocked independently (e.g. for testing or recording).

### Step 4 — Core: orchestrate

`core.py` (`Molto.chat()`) ties it together:

1. Call the type's `to_ir()` to get the Input IR.
2. Select the provider based on the model identifier (or explicit config).
3. Instantiate the provider with the vendor-specific request body.
4. Call `provider.run()` or `provider.run_stream()` to get the vendor-specific response.

## Migration Checklist (current → target)

**Input IR & request path:**
- [ ] Define the Input IR (provider-agnostic sampling params, message history, tools, etc.).
- [ ] Implement `to_ir()` on vendor request types. Refactor the current `as_run_settings()` on `vLLMResponsesRequest` — the provider-agnostic parts become `to_ir()`, the provider-specific parts move to the provider.
- [ ] Remove the `from pydantic_ai.models.openai import OpenAIChatModelSettings` import from `types/openai/responses.py`.

**Output IR & response path:**
- [ ] Define the Output IR (provider-agnostic response events for both streaming and non-streaming). Event start and end must be pure markers that carry no content — content is only delivered in delta events between them. One specific motivation: pydantic-ai sometimes mixes content into event markers — e.g. `TextPart` inside a `PartStartEvent` can already contain text content instead of being a pure start signal. The Output IR normalises these inconsistencies so that downstream type converters don't need to handle them.
- [ ] Implement `from_ir()` on vendor type modules — accepts Output IR events and produces vendor-specific response objects and streaming events.

**Runner:**
- [ ] Implement `runner.py` with a `Runner` class that wraps pydantic-ai Agent instantiation and exposes `run` / `run_stream`, normalising pydantic-ai output into Output IR.

**Providers:**
- [ ] Implement provider constructors that accept the vendor-specific request body.
- [ ] Implement `run()` / `run_stream()` as the public interface.
- [ ] Implement `_parse_run_settings()` — parse the body into pydantic-ai agent run configuration (move provider-agnostic logic from `vLLMResponsesRequest.as_run_settings()` here).
- [ ] Implement `_parse_agent_settings()` — parse the body into provider-specific agent configuration (move `as_openai_chat_settings` logic from `vLLMResponsesRequest` here).
- [ ] Implement `_compose_output()` — convert Output IR to vendor-specific responses via `Type.from_ir()`.
- [ ] Add provider implementations for additional backends (Anthropic, vLLM, etc.).

**Core:**
- [ ] Wire the full pipeline into `Molto.chat()`: `to_ir()` → `Provider(body)` → `provider.run()` / `provider.run_stream()`.
