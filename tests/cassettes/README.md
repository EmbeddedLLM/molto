# Test cassettes

These YAML files capture sanitized HTTP requests and provider responses for deterministic test playback. Most cassettes are faithful recordings apart from secret masking and the filtering of unstable transport metadata.

Normal test runs are playback-only. Do not record or rewrite cassettes during a test run. Pytest has no recording option or cassette manager; its default async HTTP transport is blocked unconditionally. Approved recordings use standalone tooling such as `scripts/recorder_proxy.py`, which retains its authorization masking and retry policy.

## Conversation discovery

Public-route and matcher tests discover files directly inside scenario directories using `{provider}-{endpoint}-{mode}-{turn}.yaml`. The filename endpoint `chat_completions` maps to `chat/completions`. Conversations are grouped by scenario, provider, endpoint, and mode, with turns ordered numerically from zero. Adding a provider cannot overwrite another provider's turn with the same number.

Discovery has separate purposes for each root:

- `matched/` supplies compatible protocol conversations and Responses-referenced translation checks.
- `native/` optionally holds faithful provider-deviation recordings for native SDK parity in supported scenarios. They do not require matched counterparts or establish translated behavior.
- `errors/` supplies only the explicitly registered `ERROR_SCENARIOS`. The Responses-only incomplete-result recordings retain their focused interface tests.
- `upstream_ir/` retains its separate projection and continuation runner; archives and legacy provider directories do not join generic discovery.

Each additional matched upstream provider runs against the canonical downstream companion and Responses reference, preferring the endpoint's default provider or requiring a unique available conversation. Native routes use that upstream provider's own request and SDK result. This preserves the existing routes without taking a cross-product of providers. Both native SDK runners pass the discovered provider explicitly. The existing native SDK baseline scenarios remain selected; other supported matched scenarios retain their native public-route assertions.

Malformed filenames, duplicate numeric identities, missing turns or required counterparts, unsupported scenarios/modes, and incompatible registered capability profiles fail collection with the affected identity. Matched scenarios retain their required two-turn conversations, three-turn sequential tool exchanges, and single-turn unknown-error cases. A new error category or unsupported scenario still needs explicit assertions. Raw reasoning requires the raw profile; encrypted matched fixtures retain the single-issuer assumption described below, including the existing `anthropic` Messages filename label. Filenames select replay behavior; they do not establish live provider support or provenance.

## Recorded cassettes

Faithful originals live under `native/`, `upstream_ir/`, or `errors/`, according to their test purpose. Preserve each original separately from any deliberately edited matched companion. Authorization values are masked, and only selected headers are retained so the fixtures remain safe and deterministic. Review request bodies for secrets before retaining recordings.

For each provider deviation, a nearby README must identify the source provider, endpoint, model and server version when available, response mode, purpose, and recording provenance. Retain the complete sanitized conversation. Native tests compare requests and SDK-parsed results with that original, asserting the observed deviation separately from any documented Molto adjustment. If translation is affected, retain the difference in the upstream replay and compare against Responses-referenced expectations; matching must not erase the deviation under test.

The matched `reasoning_raw` omission regression separately checks that the OpenAI SDK observes `cache_write_tokens` as `None` and Molto adjusts it to zero. It has no verified faithful original establishing the omission's provider provenance. Its filename selects the raw-reasoning profile; this regression does not establish live deployment behavior. Replay verifies the saved wire contract, while live provider acceptance requires approved source evidence.

Recorded headers are informational. In particular, `content-length` is not guaranteed to match the serialized body after sanitization or other approved cassette edits and must not be used as a fixture integrity check.

Treat these files as provider evidence. Do not edit a recorded value merely to make an implementation pass a test. If a committed recording is demonstrably wrong, document the correction and review its diff explicitly.

### SDK schema mismatches

When a cassette fails validation against the pinned SDK because a recorded field is missing or incompatible, identify the cassette path, SDK version, and validation error, and ask the developer to record a fresh cassette from an approved source. Do not synthesize missing values, add compatibility defaults or coercions, weaken runtime or test-helper validation, or fabricate fields in the fixture solely to make that recording pass. For example, a missing SDK-required `cache_write_tokens` count requires a new recording rather than an assumed zero.

Preserve the original evidence, continue unaffected work, and report dependent verification as awaiting the new recording. New recordings must be sanitized, reviewed, and committed before matching edits; normal test runs remain replay-only. If a fresh recording reproduces the mismatch, establish the supported provider contract with the developer before implementing compatibility behavior.

### Multi-turn upstream IR recordings

`upstream_ir/{scenario}/{provider}-{endpoint}-stream-{turn}.yaml` contains independent, unchanged provider recordings for `tests/upstream/test_multi_turn.py`. The scenarios are two-turn `text_chat`, `text_image_chat`, and `consecutive_user_text`, plus three-turn `tool_use` and `tool_use_reasoning`. Text, multimodal input, consecutive user messages, and tool use cover all three upstream clients; reasoning tool use covers Responses and Messages.

`text_image_chat` sends text, an image, and trailing text in one user message on each turn, using the existing `tests/files/github.png` and `tests/files/xlsx.jpg` fixtures as data URLs. `consecutive_user_text` sends two adjacent user messages, each with two text parts, before each assistant response. Both scenarios retain actual assistant output as history and assert that the input IR is not mutated. They also inspect outgoing message roles and content grouping independently of cassette equality, so the expected fixtures cannot silently permit merged user messages or split multimodal parts. These are upstream-client projection checks; incoming request normalization remains covered by the adapter tests.

These requests were recorded outside pytest through `scripts/recorder_proxy.py`, forwarding to the official OpenAI and Anthropic APIs. A standalone driver used `httpx.ASGITransport(app=recorder_proxy.app)` to call the recording proxy in process; the proxy's HTTP client performed the live requests. Each response's completed IR parts became assistant history for the next request. Tool results referenced the returned call IDs, and reasoning continuations retained the returned opaque state. The proxy files were renamed without changing their bytes. Authorization is masked, and Anthropic API-key headers are excluded by the recorder's header whitelist.

Replay asserts exact outgoing request bodies and provider-neutral invariants: balanced event lifecycles, reconstruction from deltas, valid tool arguments and call identities, consistent usage, terminal reasons, and preserved reasoning signatures. Generated answer wording and chunk counts are not fixed expectations. These recordings are not matched across providers, and opaque reasoning state is only replayed to its original provider. Existing SDK parity tests remain in `tests/core/test_interface.py`.

### Anthropic hosted-search recordings

These unmodified Anthropic `claude-sonnet-4-6` recordings provide independent provider evidence:

- `anthropic/web_search/*-00.yaml`: streamed search inputs, citations, and search usage.
- `anthropic/web_search/*-01.yaml`: successful continuation of the same search with original source and citation state restored.

`tests/core/test_anthropic_web_search.py` verifies SDK parity and exact history replay in both response modes. See the [provider metadata contract](../../docs/web-search.md#preserving-anthropic-search-history).

## Error cassettes

The cassettes under `errors/{scenario}/` preserve the native request and error response for each participating protocol:

```text
errors/{scenario}/{provider}-{endpoint}-{mode}-{turn}.yaml
```

`errors/incomplete_response/` contains HTTP `200` Responses recordings that stop at `max_output_tokens`. These are incomplete results, not API exceptions, so they are covered in `tests/core/test_interface.py` rather than `ERROR_SCENARIOS`. The tests cover native Responses and Responses-to-Chat translation in both downstream modes, comparing partial text and token usage with SDK-parsed recordings and asserting incomplete/length termination. Native non-streaming replay uses `openai-responses-non_stream-00.yaml`; native streaming and both translated modes share `openai-responses-stream-00.yaml`, because translated non-streaming calls still consume an upstream stream.

The streaming file was freshly recorded on November 2, 2026 through `scripts/recorder_proxy.py` against the official OpenAI Responses API, outside pytest. The developer approved changing only `model` from retired `gpt-4.1-nano` to `gpt-4.1-mini`; both prompts, `max_output_tokens: 20`, `stream: true`, and all other request-body fields remain unchanged. The sanitized recording was copied without further edits. Its terminal response includes the SDK-required `cache_write_tokens: 0` directly from the provider; no usage fields were synthesized.

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

### Comparison checks

Request expectations are compared independently of Molto's translator. For encrypted Messages-to-Responses requests, the comparison allows only the documented missing output-item IDs/phase, `include=["reasoning.encrypted_content"]`, `store=false`, and SSE padding option `stream_options.include_obfuscation=false`. It treats omitted reasoning `content` as an empty list and an implicit message `type` as `message`. Scalar assistant text is equivalent to one completed `output_text` part with empty annotations and log probabilities; it cannot stand in for multiple parts or messages. Output-only reasoning `status` is removed from the expected history, and reconstructed input must omit it. Call IDs, opaque state, input/part order, token limits, and cache settings remain compared.

Responses stream checks validate each output index and item ID before generated IDs are removed from cross-protocol comparisons. Every item update must occur between that item's start and completion. Completed and incomplete terminal responses must contain the completed items in output-index order. Translated terminal reasoning and search sources retain their `done` snapshots; native Responses may expose separate terminal snapshots for these fields and remains covered by exact SDK parity. Search identities, statuses, and queries remain compared across snapshots.

Non-streaming comparisons retain ordered supported output items and part boundaries, all reasoning items and opaque snapshots, terminal state, and semantic token usage. The incomplete-stream helper tests use an SDK-parsed recording to check `incomplete_details`, partial output, and final usage. The fresh streaming recording supplies the SDK-required cache-write count, allowing the incomplete-result comparison and its five negative cases to reach their assertions. A separate regression verifies that the helper rejects a missing count without inventing a value. Negative checks cover altered requests, item identities, output order, incomplete reasons, and usage. These checks do not add incomplete-route coverage beyond the routes described above or establish arbitrary interleaved-item support.

### Actual-output continuation

Matched public-route conversations start with the first saved request and carry its input forward cumulatively. Later calls append the actual returned assistant output and only the next saved user input or tool results, retaining that turn's settings. Before sending, the harness compares the complete carried history against the next cassette; the replay transport independently checks the outgoing upstream request. This applies to the existing streaming text, reasoning, image, and tool conversations and the existing non-streaming text/encrypted cases. Cache scenarios remain independent prefix-changing requests.

Collection is independent of Molto's adapters and collectors: Responses uses the terminal SDK response output, Chat uses the SDK's `ChatCompletionStreamState`, and Messages uses the pinned SDK's `accumulate_event`. Complete responses use the same test-side input conversion. The conversion removes reasoning's output-only `status`, streaming tool indexes, SDK parser properties, empty Chat annotations, the default Messages direct caller, and empty unsigned-thinking signatures. It does not recursively discard unknown fields. Unknown or unsupported meaningful fields must still fail comparison or public request validation.

Responses history comparison permits optional output-item ID omissions, omitted/empty reasoning content, implicit message types, and scalar assistant text equivalent to one structured output part. Empty text annotations/log probabilities are equivalent to their omission. Native fixtures may omit the optional `final_answer` phase; native commentary remains exact, while translated comparison omits phase because other protocols cannot represent it. Actual returned IDs and phase remain in the carried request wherever supplied. Portable call IDs, arguments, text/reasoning order, part boundaries, and opaque snapshots remain exact. First requests retain the original request assertions; these continuation equivalences apply only to later requests.

Negative public-replay checks change returned call IDs and opaque snapshots and require rejection before the next call. A three-turn check verifies that earlier actual output, including generated IDs absent from the fixture, remains in later requests.

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
| `simple_text` | Two plain-text turns over every protocol pair, carrying actual returned history. |
| `reasoning_raw` | Raw reasoning projected through Chat and Messages as well as Responses. Uses `vllm-*` filenames (`provider="vllm"`); see below. |
| `reasoning_encrypted` | A summarized reasoning item whose continuation is opaque encrypted state, exercising the encrypted-snapshot mapping. |
| `input_image_base64` | Base64-data-URL image user input. |
| `input_image_remote_url` | Two remote-URL image turns over all nine protocol pairs, preserving the URL and text/image order in actual continued history. |
| `input_image_with_reasoning_encrypted` | An image user turn followed by encrypted reasoning continuation. |
| `tool_call` | Three turns: a function call, then a result plus another call, then a result plus final text; exercises tool-call history over translated routes. |
| `tool_call_choice` | Named tool selection (`tool_choice: {name: read_file}`) followed by a disabled tool use (`tool_choice: none`). |
| `tool_call_parallel` | Two function calls and results with `parallel_tool_calls` enabled. |
| `tool_call_image_result_parallel` | Two parallel calls followed by image results and a final answer, replayed over Responses/Messages routes and Responses or Messages callers using Chat upstream. All results precede relocated images, with linked call IDs and preserved image order. |
| `tool_call_with_reasoning_encrypted` | A tool call whose conversation carries encrypted reasoning continuation across three turns. |
| `prompt_cache_implicit` | Two requests with automatic caching of a user prefix. |
| `prompt_cache_explicit` | Two requests with an explicit breakpoint on a system prefix and a changing user suffix. |
| `error-invalid_request_error` | A native invalid-request error scenario (`error-invalid_request_error`) used to exercise native error passthrough on the relevant route. |
| `web_search` | Search and URL attribution between Responses and Messages, native Chat parity, and rejection of signed reasoning on translated Chat routes in both modes; covered by `tests/core/test_web_search.py`. |

The `reasoning_raw` scenario is named with `vllm-*` filenames and its test derives `provider="vllm"` from that filename prefix, selecting the raw reasoning profile across all three protocols. A base URL never selects either behavior.

### Image and request projection checks

The Chat streaming matcher converts image-bearing Responses tool results into text-only tool messages followed by labeled user image messages. All parallel tool results stay contiguous before any relocated image, and images retain their source order within and across results. Constructed matcher tests assert contiguous results with multiple images and verify repeat matching on temporary files without rewriting recordings.

The two Chat companions in `matched/tool_call_image_result_parallel/` are edited replay fixtures aligned with Responses. Request-only matching aligns the model, output-token limit, streaming options, and history; event matching aligns reasoning effort, portable call IDs and their history/labels, response IDs, creation times, response model, target-default service tier, text/tool-argument deltas, finish state, and semantic usage with Responses. The first matched stream has 7 chunks and the second has 23. Matching preserves transport headers, recording timestamps, and source identifiers. The matched model is a replay value, not evidence that these edited requests were sent live. Matcher tests rematch both saved companions into temporary files, validate native SDK schemas, and verify byte idempotence while preserving both input fixtures. These stability checks do not require separate original recordings; transformation behavior is covered by the synthetic cases described above.

Focused public request tests check Responses and Messages callers using Chat upstream with images in the first, last, or both parallel results. They compare complete expected requests, including linked call IDs, result order, and image order. These tests reuse SDK-parsed text responses to check output parity; they do not establish live acceptance or replace full Chat image-result conversation replay.

Additional mocked request checks cover `required`/Messages `any` tool selection with parallel calls enabled and disabled, and nonempty Responses top-level `instructions` translated ahead of existing instructions. Replayed responses in these checks do not establish provider acceptance of the modified requests.

### Hosted search

The Responses recordings are unchanged canonical source data. The matched Messages fixtures project source URLs and citation ranges with the documented empty-title and `"N/A"` placeholders; these fixtures do not establish valid Anthropic continuation state. The Responses and Messages streaming fixtures retain opaque reasoning state, so their translation to Chat must raise `ValueError` rather than erase that state. Both streaming and non-streaming translated calls consume these upstream streams. The Chat streaming fixture remains a native SDK parity reference, not a faithful projection of the signed-reasoning streams. The non-streaming Responses recording contains no reasoning and remains representable by the Chat response matcher. Chat translation is also covered separately with unsigned raw reasoning. Native parity still compares each fixture's complete parsed SDK representation.

For independent provider evidence, see [Anthropic hosted-search recordings](#anthropic-hosted-search-recordings).

Both Messages cassette matchers reject request history containing multi-query searches, matching runtime translation. A pooled source result cannot complete each expanded server call in replayed history. Response output still expands query batches into separate calls with one pooled result linked to the first call.

The Responses stream exposes 13 sources at item completion and a fourteenth URL in the terminal snapshot. Messages stream matching uses the item-completion snapshot, as does incremental translation. Its reasoning signature likewise follows the item-completion snapshot rather than a later terminal ciphertext. Translated comparisons reduce only those unavailable differences and separately assert text/citation association, sources, placeholders, linkage, usage, and lifecycle. Do not erase these fields globally to force equality.

The SDK parses an actionless search item at `response.output_item.added`; strict model reconstruction would require the action that only appears at item completion. Retain the real event and use SDK-parsed native parity. Existing recordings have ASCII citations and successful searches. In-memory Unicode, failed/paused-search, browsing, query-cardinality, and history/error variants are synthetic regression cases, not new provider observations or edited recordings.

Responses recordings can explicitly report zero searches even outside hosted-search scenarios, while the matched Messages fixture omits search usage. Translated Messages parity takes this count from the canonical Responses terminal snapshot when token usage is reported. It compares the projected count, including zero, without rewriting the counterpart fixture or dropping search usage from comparisons.

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

The non-streaming Messages matcher maps cache writes using the matched request's automatic and block cache controls. Null controls are absent, and an omitted control lifetime defaults to `5m`. With mixed lifetimes, an existing observed breakdown is retained when its sum matches the canonical write total; a different positive total is rejected because Responses supplies no lifetime split. Zero writes clear both lifetime counts.

Relevant test entry points and canonicalization helpers:

- `python/molto/utils/test.py` defines the deterministic comparison helpers `reconstructable_responses_event`, `reconstructable_chat_completions_chunks`, and `reconstructable_anthropic_event`;
- `tests/test_match_openai_cassette_events.py`, `tests/test_match_anthropic_cassette_events.py`, `tests/test_match_openai_cassette_responses.py`, and `tests/test_match_anthropic_cassette_responses.py` verify the matcher scripts against committed cassettes; and
- `tests/utils/test_test_utils.py` covers the canonicalization helpers themselves.

When a matcher changes an SSE payload discriminator, update the corresponding `event:` header as part of the same change. Keep JSON key order and whitespace stable unless the changed value requires reserialization; this keeps fixture diffs focused on intentional semantic alignment.
