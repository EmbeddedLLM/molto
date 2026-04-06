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
       │  IR
       ▼
  ┌──────────────┐
  │ providers.py │  Map the IR into provider-specific pydantic-ai settings.
  │              │  One function per provider (OpenAI, Anthropic, vLLM, ...).
  └────┬─────────┘
       │  Provider-specific pydantic-ai kwargs (AgentRunSettings + ModelSettings)
       ▼
  ┌──────────┐
  │ core.py  │  Instantiate the pydantic-ai Agent, call run/stream,
  │ (Molto)  │  return the response.
  └──────────┘
```

### Step 1 — Types: vendor request → IR

Each vendor format lives under `types/<vendor>/` (e.g. `types/openai/responses.py`).
The type modules are responsible for:

- Pydantic models that mirror the vendor's API schema (request, response, streaming events).
- Validation logic specific to that vendor format.
- A method (e.g. `as_ir()`) that converts the validated request into the **Intermediate Representation**.

The IR is a provider-agnostic bundle of everything needed to run an Agent. |

**Key rule:** type modules must NOT import provider-specific pydantic-ai settings (e.g. `OpenAIChatModelSettings`). They produce only the common IR. This keeps the `types/` layer decoupled from any single provider.

### Step 2 — Providers: IR → provider-specific settings

`providers.py` contains one function per provider that accepts the IR and returns the concrete pydantic-ai settings:

```python
# providers.py  (sketch)

class OpenAIProvider:
     def model_settings(ir: ModelSettings) -> OpenAIChatModelSettings:
          """Map IR sampling params to OpenAI's pydantic-ai settings."""
          ...


class AnthropicProvider:
     def model_settings(ir: ModelSettings) -> AnthropicModelSettings:
          """Map IR sampling params to Anthropic's pydantic-ai settings."""
          ...
```

Provider functions handle field renaming (`max_output_tokens` → `max_tokens`), provider-specific features (e.g. `openai_reasoning_effort`), and omitting fields the provider doesn't support.

### Step 3 — Core: orchestrate

`core.py` (`Molto.chat()`) ties it together:

1. Call the type's `as_ir()` to get the IR.
2. Select the provider based on the model identifier (or explicit config).
3. Call the provider function to get pydantic-ai model settings.
4. Build and run the pydantic-ai `Agent`.

## Migration Checklist (current → target)

- [ ] Define the IR `ModelSettings` TypedDict (provider-agnostic sampling params).
- [ ] Move `as_openai_chat_settings` from `vLLMResponsesRequest` into `providers.py` as a standalone function that accepts `ModelSettings`.
- [ ] Refactor `as_run_settings` to return the IR `ModelSettings` instead of `OpenAIChatModelSettings`. Rename to `as_ir` or similar.
- [ ] Remove the `from pydantic_ai.models.openai import OpenAIChatModelSettings` import from `types/openai/responses.py`.
- [ ] Add provider functions for additional backends (Anthropic, vLLM, etc.).
- [ ] Wire the provider dispatch into `Molto.chat()`.
