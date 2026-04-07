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
  ┌──────────┐
  │  core.py │  Molto.chat() — orchestrates the pipeline:
  │  (Molto) │  1. body.to_ir()
  │          │  2. Select provider based on ir.model
  │          │  3. provider.parse_agent_settings(ir)
  │          │  4. provider.parse_run_settings(ir)
  │          │  5. Runner(agent_settings, run_settings)
  │          │  6. provider.compose_outputs() or get_output()
  └────┬─────┘
       │
       ▼
  ┌──────────────┐
  │  providers/  │  Stateless helpers — each implements:
  │              │  parse_agent_settings, parse_run_settings,
  │              │  compose_outputs, get_output
  └──────────────┘
       │
       ▼
  ┌──────────┐
  │  runner/  │  Runner wraps pydantic-ai Agent.
  │          │  run_stream() → Output IR events
  └────┬─────┘
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

### Step 2 — Core: orchestrate

`core.py` (`Molto.chat()`) is the orchestrator. It accepts any vendor-specific request type and chains together types, providers, and the runner:

```python
# core.py  (sketch)

class Molto:
     async def chat(self, body: VendorRequest) -> VendorResponse:
          # 1. Convert vendor request to provider-agnostic IR
          ir = body.to_ir()

          # 2. Select provider based on the model requested
          if ir.model.startswith("openai/"):
               provider = OpenAIResponsesProvider
          elif ir.model.startswith("vllm/"):
               provider = OpenAIChatCompletionsProvider
          ...

          # 3. Provider parses IR into pydantic-ai settings
          agent_settings = await provider.parse_agent_settings(ir)
          run_settings = await provider.parse_run_settings(ir)

          # 4. Create Runner and execute
          runner = Runner(agent_settings, run_settings)

          # 5. Provider composes the final vendor-specific output
          if ir.stream:
               return await provider.compose_outputs(runner.run_stream())
          else:
               return await provider.get_output(runner.run_stream())
```

Molto owns the pipeline — it calls `to_ir()`, selects the provider, delegates settings parsing and output composition to the provider, and wires the Runner in between. Providers do not instantiate the Runner themselves.

### Step 3 — Providers: settings parsing & output composition

Providers are stateless helpers. Each implements four methods that Molto calls at specific points in the pipeline:

```python
# providers.py  (sketch)

class _Provider:
     async def parse_agent_settings(self, ir: InputIR) -> AgentSettings:
          """Parse input IR into provider-specific pydantic-ai agent settings
          (model, model_settings with temperature, max_tokens → max_output_tokens,
          reasoning_effort, etc.)."""
          ...

     async def parse_run_settings(self, ir: InputIR) -> RunSettings:
          """Parse input IR into pydantic-ai run settings
          (message_history, instructions, toolsets, usage_limits)."""
          ...

     async def compose_outputs(self, output_stream) -> AsyncIterator[VendorStreamEvent]:
          """Compose vendor-specific streaming response from Output IR event stream.
          Calls Type.from_ir() on each event."""
          ...

     async def get_output(self, output_stream) -> VendorResponse:
          """Compose vendor-specific non-streaming response from Output IR event stream.
          Consumes the full stream and calls Type.from_ir()."""
          ...
```

1. **`parse_agent_settings`** — translates the IR's sampling parameters into provider-specific pydantic-ai settings (e.g. field renaming like `max_tokens` → `max_output_tokens`, provider-specific features like `openai_reasoning_effort`, omitting unsupported fields).
2. **`parse_run_settings`** — translates the IR into the agent run configuration (message history, instructions, toolsets, usage limits).
3. **`compose_outputs`** — takes an Output IR event stream from the Runner and yields vendor-specific streaming events via `Type.from_ir()`.
4. **`get_output`** — takes an Output IR event stream from the Runner and returns a single vendor-specific response object via `Type.from_ir()`.

### Step 4 — Runner: pydantic-ai wrapper

`runner.py` is a thin wrapper around pydantic-ai's `Agent`. Molto instantiates it with the agent settings and run settings parsed by the provider. The Runner only exposes `run_stream()`, which yields **Output IR** events. Non-streaming output is obtained by the provider consuming the same stream and assembling a single response (via `get_output`).

```python
# runner.py  (sketch)

class Runner:
     def __init__(self, agent_settings: AgentSettings, run_settings: RunSettings):
          ...

     async def run_stream(self) -> AsyncIterator[OutputIR]:
          """Yields Output IR events. This is the only execution method —
          non-streaming responses are derived by consuming this stream."""
          ...
```

The Runner is the only module that directly interacts with the pydantic-ai execution API. It normalises pydantic-ai's raw response/stream events into the Output IR before yielding them. It can be swapped or mocked independently (e.g. for testing or recording).

## Migration Checklist (current → target)

**Input IR & request path:**
- [ ] Define the Input IR (provider-agnostic sampling params, message history, tools, etc.).
- [ ] Implement `to_ir()` on vendor request types. Refactor the current `as_run_settings()` on `vLLMResponsesRequest` — the provider-agnostic parts become `to_ir()`, the provider-specific parts move to the provider.
- [ ] Remove the `from pydantic_ai.models.openai import OpenAIChatModelSettings` import from `types/openai/responses.py`.

**Output IR & response path:**
- [ ] Define the Output IR (provider-agnostic response events for both streaming and non-streaming). Event start and end must be pure markers that carry no content — content is only delivered in delta events between them. One specific motivation: pydantic-ai sometimes mixes content into event markers — e.g. `TextPart` inside a `PartStartEvent` can already contain text content instead of being a pure start signal. The Output IR normalises these inconsistencies so that downstream type converters don't need to handle them.
- [ ] Implement `from_ir()` on vendor type modules — accepts Output IR events and produces vendor-specific response objects and streaming events.

**Runner:**
- [ ] Implement `runner.py` with a `Runner` class that wraps pydantic-ai Agent instantiation and exposes `run_stream()`, normalising pydantic-ai output into Output IR. Non-streaming output is derived by providers consuming the stream.

**Providers:**
- [ ] Implement `parse_agent_settings(ir)` — translate IR sampling parameters into provider-specific pydantic-ai settings (move `as_openai_chat_settings` logic from `vLLMResponsesRequest` here).
- [ ] Implement `parse_run_settings(ir)` — translate IR into pydantic-ai run configuration (move provider-agnostic logic from `vLLMResponsesRequest.as_run_settings()` here).
- [ ] Implement `compose_outputs(output_stream)` — consume Output IR event stream and yield vendor-specific streaming events via `Type.from_ir()`.
- [ ] Implement `get_output(output_stream)` — consume Output IR event stream and return a single vendor-specific response via `Type.from_ir()`.
- [ ] Add provider implementations for additional backends (Anthropic, vLLM, etc.).

**Core:**
- [ ] Wire the full pipeline into `Molto.chat()`: `body.to_ir()` → select provider by `ir.model` → `provider.parse_agent_settings(ir)` + `provider.parse_run_settings(ir)` → `Runner(agent_settings, run_settings)` → `provider.compose_outputs()` / `provider.get_output()`.
