# Test cassettes

These YAML files capture sanitized HTTP requests and provider responses for deterministic test playback. Most cassettes are faithful recordings apart from secret masking and the filtering of unstable transport metadata.

Normal test runs are playback-only. Do not record or rewrite cassettes during a test run.

## Recorded cassettes

The cassettes under provider directories such as `openai/` preserve the recorded request and response as closely as possible. Authorization values and secret-looking request fields are masked, and only selected headers are retained so the fixtures remain safe and deterministic.

Recorded headers are informational. In particular, `content-length` is not guaranteed to match the serialized body after sanitization or other approved cassette edits and must not be used as a fixture integrity check.

Treat these files as provider evidence. Do not edit a recorded value merely to make an implementation pass a test. If a committed recording is demonstrably wrong, document the correction and review its diff explicitly.

## Error cassettes

The cassettes under `errors/{scenario}/` preserve the native request and error response for each participating protocol:

```text
errors/{scenario}/{provider}-{endpoint}-{mode}-{turn}.yaml
```

Unlike matched success cassettes, equivalent errors are not rewritten to share a status code, message, or body shape. Those differences are the behavior under test. Cross-protocol tests combine the request from the downstream protocol's cassette with the response from the upstream protocol's cassette. Error replay covers same-protocol SDK passthrough and every translated route, asserting the normalized cause chain and preservation of the upstream response on all of them.

## Matched cassettes

The cassettes under `matched/` are deliberately edited fixtures, not untouched recordings. Each scenario contains semantically equivalent native requests and responses for multiple protocols. A correct translator should be able to translate between any pair while preserving every representable semantic value and generating protocol-required structure according to the documented parity policy.

They use the scenario-first naming pattern:

```text
matched/{scenario}/{provider}-{endpoint}-{mode}-{turn}.yaml
```

Responses is the canonical behavioral reference. A matched Chat Completions or Anthropic Messages cassette remains valid native wire data, but selected values are aligned with the corresponding Responses cassette. Intentional edits include:

- using each protocol's equivalent request fields and lifecycle values;
- aligning observable delta boundaries, model and usage values, and other representable response state; and
- replacing Chat Completions (CC) chunk IDs and the Messages message ID with the corresponding Responses response ID.

These edits make translation errors observable without requiring unrelated provider-generated values to coincide by chance.

### Encrypted reasoning

For encrypted reasoning from closed-source models, matched Responses and Messages cassettes assume that OpenAI serves both endpoints. Under that same-provider fixture contract, Responses `encrypted_content` and a Messages thinking `signature` are two protocol representations of the same OpenAI-issued opaque state. The Messages cassette is therefore edited to carry the encrypted content or signature from the Responses cassette. This does not imply that an OpenAI-issued value is valid against Anthropic's API, or that opaque reasoning state is portable across providers.

Responses exposes three encrypted snapshots for one reasoning item:

1. `response.output_item.added.item.encrypted_content`;
2. `response.output_item.done.item.encrypted_content`; and
3. the reasoning item's `encrypted_content` inside `response.completed.response.output`.

A Messages stream has only two corresponding signature slots: the thinking block start and a later replacement `signature_delta`. The matched Responses cassette therefore edits the terminal `response.completed` encrypted content to equal the `response.output_item.done` value. This lets tests assert the terminal state while keeping translation incremental; reproducing a separately observed third value would require buffering the entire stream.

The `signature_delta` is a complete replacement snapshot, not text to append to the initial signature.

In the current recordings the three ciphertexts cover the same number of reasoning tokens, but their byte differences may reflect opaque envelope metadata such as a version, timestamp, or nonce. This equivalence is empirical and non-contractual: applications must not inspect or synthesize these values. Matthew Green's [“Let's talk about encrypted reasoning”](https://blog.cryptographyengineering.com/2026/05/29/fooling-around-with-encrypted-reasoning-blobs/) provides cryptographic background on treating the blobs as opaque authenticated ciphertext and on the metadata their lengths can reveal; it does not establish equivalence among the three snapshots.

### Scenario listing

The `matched/{scenario}/` directories currently exercise these scenarios. One scenario may span several turns, so each turn subfile carries `-01`, `-02`, and so on:

| Scenario | What it exercises |
| --- | --- |
| `simple_text` | A single plain-text exchange over every protocol pair. |
| `reasoning_raw` | Raw reasoning projected through Chat and Messages as well as Responses. Uses `vllm-*` filenames (`provider="vllm"`); see below. |
| `reasoning_encrypted` | A summarized reasoning item whose continuation is opaque encrypted state, exercising the encrypted-snapshot mapping. |
| `input_image_base64` | Base64-data-URL image user input. |
| `input_image_remote_url` | Remote-URL image user input. |
| `input_image_with_reasoning_encrypted` | An image user turn followed by encrypted reasoning continuation. |
| `tool_call` | Three turns: a function call, then a result plus another call, then a result plus final text; exercises tool-call history over translated routes. |
| `tool_call_choice` | Named tool selection (`tool_choice: {name: read_file}`) followed by a disabled tool use (`tool_choice: none`). |
| `tool_call_parallel` | Two function calls and results with `parallel_tool_calls` enabled. |
| `tool_call_image_result_parallel` | Parallel tool calls where a result carries an image, exercising the Chat image-tool-result relocation. |
| `tool_call_with_reasoning_encrypted` | A tool call whose conversation carries encrypted reasoning continuation across three turns. |
| `prompt_cache_implicit` | Two requests with automatic caching of a user prefix. |
| `prompt_cache_explicit` | Two requests with an explicit breakpoint on a system prefix and a changing user suffix. |
| `error-invalid_request_error` | A native invalid-request error scenario (`error-invalid_request_error`) used to exercise native error passthrough on the relevant route. |

The `reasoning_raw` scenario is named with `vllm-*` filenames and its test derives `provider="vllm"` from that filename prefix, selecting the raw reasoning profile across all three protocols. A base URL never selects either behavior.

### Prompt caching

`matched/prompt_cache_implicit/` and `matched/prompt_cache_explicit/` each contain two streaming requests per protocol, numbered `-00` and `-01` in recording order. Responses remains unmodified and supplies the canonical request content, response ID, model, text deltas, and token counts. Chat also carries the Responses creation timestamp; Messages has no timestamp field.

Both OpenAI protocols use `30m`; matching projects that lifetime to Messages' `1h` under the translation policy, including in the implicit scenario. Automatic caching stays at the request level, and the explicit system breakpoint retains its marked block boundary. Both OpenAI counterparts represent every instruction as a text-part list, including unmarked instructions.

All four canonical Responses streams report cache hits and zero cache writes. The matched counterparts use those counts, including Messages' separate uncached input count and cache-creation lifetime breakdown. These fixtures cover warm-cache replay; positive cache creation is covered separately by synthetic usage tests.

## Specifications and tooling

The detailed contracts are documented in:

- [Python conventions](../../python/CONVENTIONS.md#matched-cassettes), including naming, maintenance, and verification rules;
- [Python architecture](../../python/ARCHITECTURE.md#replay-and-parity-testing), including how parity assertions differ by route;
- the [shared translation contract](../../docs/README.md#translation-contract);
- the [Responses fidelity contract](../../docs/protocols/openai-responses.md#fidelity) and [Chat Completions fidelity contract](../../docs/protocols/openai-chat-completions.md#fidelity); and
- the [Anthropic Messages opaque thinking policy](../../docs/protocols/anthropic-messages.md#opaque-thinking-state), including the encrypted-reasoning lifecycle policy.

Use `scripts/match_openai_cassette_events.py` and `scripts/match_anthropic_cassette_events.py` for deterministic alignment. Start from approved recordings, keep the original recordings committed before editing, review every transformed diff, and ensure each result still validates as native wire data for its protocol.

Relevant test entry points and canonicalization helpers:

- `python/molto/utils/test.py` defines the deterministic comparison helpers `reconstructable_responses_event`, `reconstructable_chat_completions_chunks`, and `reconstructable_anthropic_event`;
- `tests/test_match_openai_cassette_events.py` and `tests/test_match_anthropic_cassette_events.py` verify the matcher scripts against committed cassettes; and
- `tests/utils/test_test_utils.py` covers the canonicalization helpers themselves.

When a matcher changes an SSE payload discriminator, update the corresponding `event:` header as part of the same change. Keep JSON key order and whitespace stable unless the changed value requires reserialization; this keeps fixture diffs focused on intentional semantic alignment.
