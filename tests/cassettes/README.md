# Test cassettes

These YAML files capture sanitized HTTP requests and provider responses for deterministic test playback. Most cassettes are faithful recordings apart from secret masking and the filtering of unstable transport metadata.

Normal test runs are playback-only. Do not record or rewrite cassettes during a test run.

## Recorded cassettes

The cassettes under provider directories such as `openai/` preserve the recorded request and response as closely as possible. Authorization values and secret-looking request fields are masked, and only selected headers are retained so the fixtures remain safe and deterministic.

Recorded headers are informational. In particular, `content-length` is not guaranteed to match the serialized body after sanitization or other approved cassette edits and must not be used as a fixture integrity check.

Treat these files as provider evidence. Do not edit a recorded value merely to make an implementation pass a test. If a committed recording is demonstrably wrong, document the correction and review its diff explicitly.

### Multi-turn upstream IR recordings

`upstream_ir/{scenario}/{provider}-{endpoint}-stream-{turn}.yaml` contains independent, unchanged provider recordings for `tests/upstream/test_multi_turn.py`. The scenarios are two-turn `text_chat`, `text_image_chat`, and `consecutive_user_text`, plus three-turn `tool_use` and `tool_use_reasoning`. Text, multimodal input, consecutive user messages, and tool use cover all three upstream clients; reasoning tool use covers Responses and Messages.

`text_image_chat` sends text, an image, and trailing text in one user message on each turn, using the existing `tests/files/github.png` and `tests/files/xlsx.jpg` fixtures as data URLs. `consecutive_user_text` sends two adjacent user messages, each with two text parts, before each assistant response. Both scenarios retain actual assistant output as history and assert that the input IR is not mutated. They also inspect outgoing message roles and content grouping independently of cassette equality, so the expected fixtures cannot silently permit merged user messages or split multimodal parts. These are upstream-client projection checks; incoming request normalization remains covered by the adapter tests.

These requests were recorded outside pytest through `scripts/recorder_proxy.py`, forwarding to the official OpenAI and Anthropic APIs. A standalone driver used `httpx.ASGITransport(app=recorder_proxy.app)` to call the recording proxy in process; the proxy's HTTP client performed the live requests. Each response's completed IR parts became assistant history for the next request. Tool results referenced the returned call IDs, and reasoning continuations retained the returned opaque state. The proxy files were renamed without changing their bytes. Authorization is masked, and Anthropic API-key headers are excluded by the recorder's header whitelist.

Replay asserts exact outgoing request bodies and provider-neutral invariants: balanced event lifecycles, reconstruction from deltas, valid tool arguments and call identities, consistent usage, terminal reasons, and preserved reasoning signatures. Generated answer wording and chunk counts are not fixed expectations. These recordings are not matched across providers, and opaque reasoning state is only replayed to its original provider. Existing SDK parity tests remain in `tests/core/test_interface.py`.

## Error cassettes

The cassettes under `errors/{scenario}/` preserve the native request and error response for each participating protocol:

```text
errors/{scenario}/{provider}-{endpoint}-{mode}-{turn}.yaml
```

`errors/incomplete_response/` contains HTTP `200` Responses recordings that stop at `max_output_tokens`. These are incomplete results, not API exceptions, so they are covered in `tests/core/test_interface.py` rather than `ERROR_SCENARIOS`. The tests cover native Responses and Responses-to-Chat translation in both downstream modes, comparing partial text and token usage with SDK-parsed recordings and asserting incomplete/length termination. Native non-streaming replay uses `openai-responses-non_stream-00.yaml`; native streaming and both translated modes share `openai-responses-stream-00.yaml`, because translated non-streaming calls still consume an upstream stream. Both files were moved unchanged from `openai/responses/test_incomplete_responses/`; the redundant `test_incomplete_response-non-stream-sse.yaml` recording was removed.

Unlike matched success cassettes, equivalent errors are not rewritten to share a status code, message, or body shape. Those differences are the behavior under test. Cross-protocol tests combine the request from the downstream protocol's cassette with the response from the upstream protocol's cassette. Add error categories to `ERROR_SCENARIOS` in `tests/core/test_exceptions.py`. The shared test replays streaming requests through the native SDK and every Molto route, asserting SDK error parity, the normalized cause chain, and preservation of the upstream response. It consumes returned streams to cover errors delivered after successful HTTP headers as well as immediate HTTP failures.

Non-streaming error tests use the `non_stream` recordings for every error scenario across all three protocols, covering 12 native and 24 translated cases. Match the Chat Completions and Messages requests to the canonical Responses request with the request-only scripts before adding route coverage. Tests compare the matched model and input directly. The SDK reference replays each cassette in its recorded mode. Molto still streams translated calls upstream, so the expected request accounts only for the execution-mode differences: `stream=true`, Chat Completions usage reporting, and disabled obfuscation when translating from Messages. Recorded HTTP error responses and transport metadata remain unchanged.

Match error-cassette requests before adding a scenario; streaming route tests compare request bodies directly without model or streaming-option adjustments. Use the Responses request as the canonical input, with `stream_options.include_obfuscation: false` when matching Messages routes. The matching scripts support `--request-only` to align counterpart requests while preserving error responses and transport metadata:

```bash
python scripts/match_openai_cassette_events.py <chat-cassette> <responses-cassette> --request-only
python scripts/match_anthropic_cassette_events.py <messages-cassette> <responses-cassette> --request-only
```

The context-length streaming requests use the same model and equivalent streaming settings after matching. The Anthropic model in these replay requests is therefore a matching value, not a model to send to the live Anthropic endpoint. Recorded error bodies and SSE events remain unchanged.

The streaming and non-streaming recordings under `errors/context_length_exceeded/` have their repeated request text shortened to a single `x` to keep fixtures small and replay fast. The original instruction and recorded responses, including token counts and SSE events, are preserved. These shortened requests test error handling through replay; they no longer reproduce context overflow against a live provider.

An error delivered inside an HTTP `200` stream is recorded under `response.sse`, retaining status `200`; it must not be rewritten as an HTTP `400` error. The OpenAI SDK may raise a plain `APIError` for such an event without attaching an HTTP status to the exception. If the SDK disconnects before the upstream stream finishes, the recorder saves the SSE lines received so far without adding a disconnect marker. That recording may be partial and does not establish which events the provider would have sent afterward.

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

Use `scripts/match_openai_cassette_events.py` and `scripts/match_anthropic_cassette_events.py` to align streaming SSE fixtures. Use `scripts/match_openai_cassette_responses.py` and `scripts/match_anthropic_cassette_responses.py` to align non-streaming JSON response fixtures. Start from approved recordings, keep the original recordings committed before editing, review every transformed diff, and ensure each result still validates as native wire data for its protocol.

Relevant test entry points and canonicalization helpers:

- `python/molto/utils/test.py` defines the deterministic comparison helpers `reconstructable_responses_event`, `reconstructable_chat_completions_chunks`, and `reconstructable_anthropic_event`;
- `tests/test_match_openai_cassette_events.py`, `tests/test_match_anthropic_cassette_events.py`, `tests/test_match_openai_cassette_responses.py`, and `tests/test_match_anthropic_cassette_responses.py` verify the matcher scripts against committed cassettes; and
- `tests/utils/test_test_utils.py` covers the canonicalization helpers themselves.

When a matcher changes an SSE payload discriminator, update the corresponding `event:` header as part of the same change. Keep JSON key order and whitespace stable unless the changed value requires reserialization; this keeps fixture diffs focused on intentional semantic alignment.
