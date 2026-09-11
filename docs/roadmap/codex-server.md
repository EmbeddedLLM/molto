# Codex server roadmap

Status: proposed future work. Molto currently provides a Python translation library; it does not yet ship the HTTP server, model catalog, or launch command described here.

The goal is to let Codex CLI use Molto as an HTTP Responses provider, with requests routed to OpenAI Responses, Chat Completions, or Anthropic Messages upstreams. The first release should be stateless and streaming-only, reusing the library's translation and native passthrough paths.

## Existing foundations

The library already provides:

- `Molto.responses()` with upstream protocol, provider, base URL, and API-key selection.
- Streaming text, function calls and results, supported images, and reasoning history under the [Responses translation contract](../protocols/openai-responses.md).
- Prompt-cache options and supported content breakpoints under the [shared prompt-cache policy](../README.md#prompt-caching).
- Acceptance of `store=false` and `include=["reasoning.encrypted_content"]` as documented no-ops on translated routes, with supported reasoning signatures preserved independently of the include hint.
- Request-scoped `extra_body` and `extra_headers` arguments on every coordinator, with credential and transport-header filtering before upstream forwarding.
- First-event prefetching, so setup failures surface when awaiting the coordinator, and a `MoltoStream` that exposes successful upstream `response_headers` before consumption and can be closed explicitly.

The remaining work is the server boundary: Codex compatibility validation, model routing, request policy, downstream header filtering, HTTP streaming, and lifecycle handling. The transport APIs above are available library foundations; the server interfaces below remain proposed.

Current behavior remains documented in the [protocol guides](../README.md#choose-a-protocol), [Python architecture](../../python/ARCHITECTURE.md), and [Python conventions](../../python/CONVENTIONS.md).

## First-release scope

The proposed server entry point is `python -m molto.server`, implemented with FastAPI and an explicit optional `server` dependency extra. It should expose:

| Endpoint                            | Planned behavior                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| `POST /v1/responses`                | Validate a streaming Responses request, resolve its model route, and return Responses SSE events. |
| `GET /v1/models?client_version=...` | Return a Codex model catalog with public metadata only.                                           |

Codex should retain conversation history locally and resend the required history on each turn. Molto should retain startup configuration and shared HTTP client pools, without storing conversations. Local compaction must be verified with the supported Codex version before this scope is considered sufficient.

The first release should cover ordinary text turns, function-tool execution and continuation, supported image input, reasoning continuation, and local compaction. WebSockets, remote compaction, response retrieval, `previous_response_id` lookup, hosted search, and server-side conversation storage are deferred.

## Milestone 1: establish the Codex compatibility baseline

Select and record a tested Codex version. Verify its request bodies, model-catalog schema, headers, and tool definitions against the pinned SDKs and Molto's translated request contract.

Use this proposed configuration in the user's Codex configuration file when testing the server:

```toml
model_provider = "molto"
web_search = "disabled"

[model_providers.molto]
name = "Molto"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
supports_standalone_web_search = false
```

These transport and search settings are described in the [official Codex configuration reference](https://developers.openai.com/codex/config-reference). The example assumes a local server without inbound authentication; add a separate server credential when authentication is enabled.

Keep the provider name distinct from `OpenAI` and verify that the chosen Codex release performs compaction locally. For translated routes, start with catalog settings `apply_patch_tool_type: null`, `shell_type: unified_exec`, and `use_responses_lite: false`. Confirm that Codex can edit files through function tools without introducing unsupported custom patch tools.

Disable hosted web search in Codex configuration; model-catalog search metadata alone is insufficient. Reject unexpected hosted tools explicitly. Native Responses acceptance of a tool schema does not establish that Molto can translate it: custom tools, namespaces, deferred tool search, MCP, and apps each need compatibility coverage before being advertised.

Completion criteria: capture the baseline request shapes and prove that text, function-tool continuation, and local compaction require only the planned HTTP surface. Record any extra fields or endpoints as blockers for the next milestones.

## Milestone 2: package and validate model routing

Define a small `models.yaml` schema with separate public metadata and private routing:

| Area | Required information |
| --- | --- |
| Public catalog | Model slug, display name, reasoning presets, context and truncation limits, visibility, modalities, and supported tool settings. |
| Private route | Upstream endpoint, provider capability profile, base URL, optional upstream model override, API-key environment-variable name, and cache policy. |

The public slug should select one immutable route. Substitute the upstream model ID before the SDK call. Never return routing details or credentials from the model endpoint.

Ship a packaged catalog at `python/molto/models.yaml`, declared as setuptools package data and loaded through `importlib.resources`. Support an external replacement, such as `--models /etc/molto/models.yaml` or `MOLTO_MODELS_FILE`, without relying on the working directory. Keep resolved secrets in memory and the packaged resource read-only.

The initial catalog should exercise a native Responses route and a translated Messages route. The original candidates were `gpt-5.6-luna` and `claude-sonnet-5`; verify availability, model IDs, reasoning settings, context limits, and service tiers when implementing the catalog. Fill required Codex metadata with validated defaults instead of maintaining an unverified model snapshot here.

At startup, reject duplicate slugs, invalid route settings, missing required credentials, invalid upstream URLs, and metadata advertising unsupported features. At request time, reject unknown models before contacting a provider.

Serve the Codex-specific `{ "models": [...] }` representation, accept `client_version`, and optionally return an `ETag`. Validate the response against the supported Codex release's `ModelInfo` and `ModelsResponse` schemas. A static Codex `model_catalog_json` file is an alternative for early testing; that interface expects JSON, even if Molto's internal catalog uses YAML.

Completion criteria: catalog validation, model alias routing, public/private separation, external overrides, and installed-wheel resource loading all have tests.

## Milestone 3: implement server request and transport policies

### Responses fields and extensions

Reuse the library's acceptance of `store=false` and `include=["reasoning.encrypted_content"]` on translated routes. The stateless server should reject requests that require stored Responses state on every route, including native Responses routes that the library otherwise passes through.

Extract Codex `client_metadata` before strict request validation. Use the coordinator's `extra_body` argument to forward it only to a native Responses upstream known to accept it. On translated routes, discard it or retain bounded tracing fields under a documented policy. Avoid logging the full metadata because it can contain workspace and tool context. The server must restrict extension fields so they cannot override validated routing or request policy; `extra_body` bypasses normalization and is merged by the upstream SDK.

Define mapping or rejection rules for optional `service_tier` and `text` settings before exposing related catalog capabilities. Unsupported fields should produce actionable errors.

### Prompt-cache route policy

Reuse Molto's existing mapping; the server's remaining responsibility is choosing and enforcing a route policy.

| Route policy | Planned request handling |
| --- | --- |
| OpenAI Responses or Chat Completions | Preserve caller cache keys, options, retention, and supported breakpoints through Molto. |
| Messages with caller-controlled caching | Preserve supported options and markers. Molto drops the caller key; a key alone does not enable caching. |
| Messages with automatic caching by default | Supply `prompt_cache_options: {"mode": "implicit"}` when the caller has not specified options or markers. Without a TTL or markers, this uses Anthropic's five-minute default. |
| Messages with a one-hour default | Supply implicit mode and `ttl: "30m"`; Molto maps that minimum lifetime to Anthropic `1h`. |
| Messages with caching disabled | Reject options or markers that request caching. A key alone may remain because Molto drops it without enabling caching. Explicit mode is not an off switch for marked blocks. |

Use route defaults only when the caller has not supplied cache intent. Reject conflicts with a mandatory route policy instead of silently changing the caller's requested boundaries or lifetime. Document the cache-write pricing implications when enabling an Anthropic default.

Supported content-part `prompt_cache_breakpoint` values already map to Anthropic block-level `cache_control`. With OpenAI markers, omitted mode and TTL resolve to implicit mode and `30m`, which maps to one-hour Messages caching. Preserve the markers and let Molto enforce slot limits, TTL ordering, and prefix constraints. Messages still rejects `prompt_cache_retention` on translated routes. The [shared policy](../README.md#prompt-caching) is the authoritative list of mappings and limitations.

### Headers and credentials

Use the coordinator's `extra_headers` argument and existing filtering of client credentials, account selectors, hop-by-hop fields, request framing, and proxy topology headers. Define which Codex operational headers the server forwards. Upstream credentials must come from the selected route; optional inbound bearer authentication belongs at the server boundary.

Read successful upstream headers from `MoltoStream.response_headers` before the first downstream event is sent, and failure headers from the SDK exception's response. The server must filter request IDs, `x-codex-turn-state`, model and catalog metadata, supported rate-limit and safety-buffering headers, and retry information for forwarding. Do not copy `content-length`, `content-encoding`, `transfer-encoding`, or hop-by-hop headers after re-streaming the body; Molto preserves response headers without this downstream filtering. Passing turn-state headers in both directions should not require Molto-owned conversation state.

Completion criteria: server tests build on the library's SDK-backed transport coverage to verify accepted Responses fields, native-only extension policy, credential isolation, operational-header forwarding, and downstream response-header filtering. Cache-policy tests cover caller intent, route defaults, forbidden caching, key-only requests, and explicit markers.

## Milestone 4: implement HTTP streaming and lifecycle handling

Create and close shared Molto/HTTP clients with the application lifecycle. Resolve the model route and await `Molto.responses()` before starting the downstream streaming response so validation and setup errors can still produce HTTP errors.

Serialize typed events as `text/event-stream`, with the event type in `event:` and the serialized SDK event in `data:`, followed by a blank line. Preserve the complete Responses lifecycle: output items, function calls and argument deltas, reasoning, usage, and completed or incomplete status. Text deltas alone are insufficient for Codex continuation.

Return an OpenAI-compatible JSON error with an appropriate HTTP status for pre-stream failures. After headers are sent, emit a valid `response.failed` event when the failure can be represented faithfully; otherwise terminate the stream. Never synthesize successful completion after an upstream failure. Preserve upstream request IDs and retry information where available.

Close the upstream iterator on client disconnect, generator cancellation, failure, or normal completion, and configure deployment proxies to avoid SSE buffering. Reject non-streaming requests until collection is supported.

Completion criteria: HTTP integration tests verify catalog responses, model routing, authentication, exact SSE framing and event order, setup and mid-stream errors, cancellation, and resource cleanup.

## Milestone 5: validate and document the first release

Run a Codex smoke test against both native Responses and translated Messages routes, and against Chat Completions if included in the published catalog. Cover a text turn, `exec_command`, `function_call_output` continuation, image input where advertised, reasoning continuation, and long-history local compaction. Test any additional MCP or app behavior before claiming support.

Keep provider behavior covered by cassette-backed SDK parity tests under the [cassette policy](../../tests/cassettes/README.md). Normal test runs must replay committed fixtures. Add server-specific transport tests separately, including installed-wheel startup and catalog loading.

Publish the supported Codex version, install and launch commands, catalog overrides, authentication setup, cache defaults, and known limitations only after those paths work. Move delivered behavior into a server usage guide and update this roadmap as milestones land.

## Later work

Evaluate these independently after the stateless baseline is reliable:

- Translation of custom patch tools, namespaces, deferred tool search, and additional MCP/app request shapes.
- Hosted web search and the standalone search endpoint.
- Responses WebSockets and remote compaction.
- Non-streaming responses, stored response retrieval, and conversation state.
- Additional model catalog capabilities, including service tiers and structured output, once their mappings are implemented and tested.

Each expansion should start from a concrete Codex workflow and a verified protocol contract.
