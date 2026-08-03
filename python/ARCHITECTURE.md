# Molto architecture

Molto has one mandatory translation path for every supported request:

```text
public request adapter
        |
        v
    InputIR -> OpenAI-compatible provider endpoint -> OutputEvent stream
                                                        |
public output adapter <- OutputIR <---------------------+
```

There is no native-provider bypass. The provider receives `InputIR`, and public
responses are composed only after the output event stream has been validated by
`OutputEventSequence` and collected into `OutputIR`.

## Portable IR

The IR is intentionally OpenAI-shaped rather than a union of provider APIs. It
contains:

- ordered text, image, and audio input parts where the selected endpoint supports
  them;
- assistant messages, reasoning items, and developer-hosted function calls/results;
- text, JSON object, and JSON Schema output formats;
- common generation controls, state references, usage, errors, and terminal status;
- stateless Responses history for assistant output, reasoning, web search, code
  interpreter, and image generation items;
- semantic `response.*` events for response lifecycle, items, content parts,
  function arguments, and terminal results.
- typed `openai.stream_frame` and `openai.response_snapshot` events containing
  canonical JSON from objects already validated by the OpenAI SDK.

`OutputIR` stores the validated items, usage, OpenAI-compatible response metadata,
and the complete `OutputEventSequence`. Unknown or richer provider data is an
explicit rejection. It is never hidden in an opaque passthrough field.

## Provider boundary

`molto.providers.openai.OpenAIProvider` is the only SDK execution adapter. It
uses the official OpenAI SDK endpoint selected by the request model:

| Model prefix | Chat request | Responses request |
| --- | --- | --- |
| `openai/` | OpenAI Chat Completions | OpenAI Responses |
| `vllm/` | OpenAI-compatible Chat Completions | OpenAI-compatible Responses |
| `anthropic/` | Anthropic OpenAI-compatible Chat | rejected |

`validate_provider_request()` is the pre-construction capability gate. It derives
requirements from `InputIR`, checks state ownership, and rejects unsupported
features before an SDK client is created. The native provider translator then
builds a canonical OpenAI request dictionary.

The Anthropic-native Messages adapter has been removed. An `anthropic/` model is
supported only through the provider's OpenAI-compatible Chat endpoint, so all
provider output uses the same OpenAI-shaped translator and event vocabulary.

Responses requests can manage history without provider state by appending the
complete supported prior `response.output` to the next input. The request IR
retains assistant message IDs/status/phase, reasoning state, web-search actions,
code-interpreter code and outputs, image-generation call IDs/results/status, and
`store=False`. Complete image output items are accepted at the public boundary,
but output-only metadata is not sent in the narrower OpenAI input item. Image
generation may instead use an `image_generation_call` containing only the prior
call ID. Other hosted history item types reject until they have an equally
complete typed representation.

## Output translation

Chat and Responses payloads are translated into the same validated event
sequence. The public adapters then compose the requested destination model:

- same-format responses preserve exact parsed-SDK shape from the typed OpenAI
  frame or snapshot event while still traversing `OutputIR`;
- cross-endpoint responses compose only representable text, refusal, function
  calls, terminal status, and usage; Responses reasoning is intentionally omitted
  when the destination is Chat Completions;
- hosted tools, MCP, image generation, unsupported annotations, and endpoint-only
  fields fail closed when the chosen destination cannot carry them. Generated Chat
  audio is retained in the portable Chat output item and is rejected only when the
  selected destination is not Chat Completions.

Streaming and non-streaming output use the same event-sequence boundary. Exact
OpenAI frames coexist with semantic events: matching-format adapters consume the
frames, while cross-format and OpenResponses adapters consume only semantics.
No source frame appears in a cross-format public response.

## Module ownership

- `python/molto/types/openai/*`: public OpenAI request/response models and their
  request/output adapters.
- `python/molto/types/openresponses/*`: the portable OpenResponses-shaped public
  models, backed by the same IR event stream.
- `python/molto/types/ir/*`: strict portable models, event invariants, emitter,
  usage, state, and collection.
- `python/molto/providers/capabilities.py`: provider contracts and
  pre-provider validation.
- `python/molto/providers/openai/*`: SDK construction, request translation,
  output translation, usage mapping, and error mapping.
- `python/molto/core.py`: public routing, execution, and output adapter selection.

## Resource and error contracts

Caller-provided HTTP clients remain caller-owned. SDK-created clients close after
the response completes or the stream is exhausted. SDK failures are mapped to
Molto exceptions; provider-returned failed Responses snapshots remain typed
failed output events.

## Verification policy

Approved OpenAI SDK cassettes are the parity oracle for supported fields and
ordered events. Tests must exercise the public request adapter, `InputIR`, the
OpenAI-compatible endpoint, `OutputEvent`, `OutputIR`, and the public output
adapter. Unsupported provider features receive pre-execution or output-composition
rejection tests. Cassette recording and replay are user/CI operations, not local
implementation steps in this repository workflow.
