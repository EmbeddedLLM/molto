# Codex server handoff

This is the remaining work to make `python -m molto.server` a viable HTTP upstream for Codex CLI. The recommended first version is a stateless, streaming-only Responses server backed by Molto.

## Compatibility baseline

Configure Codex to use HTTP Responses and to keep optional hosted features out of the request:

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

Keep the provider name different from the literal `OpenAI`. Codex uses that name to recognize the first-party provider; a non-OpenAI custom provider uses local compaction, so the initial Molto server does not need `/responses/compact`. Disabling WebSockets means it needs no WebSocket implementation, and disabling standalone search means it needs no `/alpha/search` endpoint.

Every model returned to Codex should include at least these compatibility choices:

```yaml
apply_patch_tool_type: null
shell_type: unified_exec
use_responses_lite: false
```

`apply_patch_tool_type: null` removes the dedicated custom patch tool. Codex can still edit through the `exec_command` function tool. The model catalog cannot disable hosted web search: `web_search_tool_type` selects a result representation and has no disabled value. The top-level `web_search = "disabled"` setting prevents that tool from entering Responses requests, while `supports_standalone_web_search = false` prevents out-of-band search calls. The server should reject an unexpected `web_search` tool with a clear unsupported-input error rather than silently dropping it.

The previous blanket SDK-compatibility report is no longer accurate. OpenAI SDK 2.53 accepts the current Codex `custom`, `namespace`, and `tool_search` definitions on native Responses requests. It rejects Codex's extended hosted `web_search` shape. Molto's cross-protocol Responses adapter is narrower and accepts only client `function` tools, so disabling custom patch remains necessary for translated routes; namespace, deferred tool search, MCP, and apps also require further translation work if those features must work through Chat Completions or Messages.

Relevant Codex implementation references are [model provider configuration](./__ref__/codex/codex-rs/model-provider-info/src/lib.rs), [tool construction](./__ref__/codex/codex-rs/core/src/tools/spec_plan.rs), and [hosted web-search selection](./__ref__/codex/codex-rs/core/src/tools/hosted_spec.rs).

## HTTP surface

Implement `python/molto/server.py` with a FastAPI application and a module entry point so this works:

```bash
python -m molto.server
```

The recommended configuration requires two endpoints:

- `POST /v1/responses`: accept a streaming Responses request and return Responses SSE.
- `GET /v1/models?client_version=...`: return the configured Codex model catalog.

Codex appends `client_version` when loading models. Its expected payload is the Codex-specific `{ "models": [...] }` schema, not the standard OpenAI `{ "object": "list", "data": [...] }` schema. The server may return an `ETag`, which Codex understands. The schema is defined by [`ModelInfo` and `ModelsResponse`](./__ref__/codex/codex-rs/protocol/src/openai_models.rs), and the HTTP behavior is in [the Codex models endpoint](./__ref__/codex/codex-rs/codex-api/src/endpoint/models.rs).

An alternative is to configure Codex with a static `model_catalog_json`, but Codex accepts JSON there, not YAML. Serving `/v1/models` lets Molto own a friendlier `models.yaml` and translate it to Codex's JSON representation.

## Model configuration

Define and validate a small `models.yaml` schema at startup. Each entry needs two kinds of data:

- Codex catalog metadata: slug, display name, reasoning presets, context and truncation limits, visibility, tool settings, and optional service tiers.
- Molto routing: upstream endpoint, provider capability profile, upstream base URL, optional upstream model-name override, and the name of the environment variable containing its API key.

A concrete initial catalog should live at `python/molto/models.yaml` and contain `gpt-5.6-luna` and `claude-sonnet-5`. One workable YAML shape keeps the public Codex metadata separate from private routing configuration:

```yaml
models:
  - slug: gpt-5.6-luna
    codex:
      display_name: GPT-5.6-Luna
      description: Fast and affordable agentic coding model.
      default_reasoning_level: medium
      supported_reasoning_levels:
        - effort: low
          description: Fast responses with lighter reasoning
        - effort: medium
          description: Balanced speed and reasoning depth
        - effort: high
          description: Greater reasoning depth for complex problems
        - effort: xhigh
          description: Extra-high reasoning depth for complex problems
        - effort: max
          description: Maximum reasoning depth for the hardest problems
      context_window: 272000
      max_context_window: 872000
      effective_context_window_percent: 95
      truncation_policy:
        mode: tokens
        limit: 10000
      input_modalities: [text, image]
      visibility: list
      supported_in_api: true
      priority: 0
      default_reasoning_summary: none
      support_verbosity: true
      default_verbosity: low
      shell_type: unified_exec
      apply_patch_tool_type: null
      supports_search_tool: false
      use_responses_lite: false
    upstream:
      endpoint: responses
      provider: openai
      model: gpt-5.6-luna
      base_url: https://api.openai.com/v1
      api_key_env: OPENAI_API_KEY

  - slug: claude-sonnet-5
    codex:
      display_name: Claude Sonnet 5
      description: Anthropic Sonnet model exposed through Molto translation.
      default_reasoning_level: high
      supported_reasoning_levels:
        - effort: low
          description: Lower latency and thinking depth
        - effort: medium
          description: Balanced latency and thinking depth
        - effort: high
          description: Default reasoning depth
        - effort: xhigh
          description: Extra-high reasoning depth
        - effort: max
          description: Maximum reasoning depth
      context_window: 1000000
      max_context_window: 1000000
      effective_context_window_percent: 95
      truncation_policy:
        mode: tokens
        limit: 10000
      input_modalities: [text, image]
      visibility: list
      supported_in_api: true
      priority: 1
      default_reasoning_summary: auto
      support_verbosity: false
      default_verbosity: null
      shell_type: unified_exec
      apply_patch_tool_type: null
      supports_search_tool: false
      use_responses_lite: false
    upstream:
      endpoint: messages
      provider: anthropic
      model: claude-sonnet-5
      base_url: https://api.anthropic.com
      api_key_env: ANTHROPIC_API_KEY
      prompt_cache:
        mode: automatic
        ttl: 5m
```

The Luna catalog values above follow Codex's checked-in model definition, except for the deliberate compatibility overrides `apply_patch_tool_type: null`, `supports_search_tool: false`, and `use_responses_lite: false`. Claude Sonnet 5 uses its stable API model ID, a 1M-token context window, adaptive reasoning with a default `high` effort, and no Priority Tier; verify these provider values when implementing the loader against the current [Anthropic model documentation](https://docs.anthropic.com/en/docs/about-claude/models/migrating-to-claude-4).

The loader should supply the less interesting required `ModelInfo` fields with conservative defaults, including `availability_nux: null`, `upgrade: null`, `experimental_supported_tools: []`, and empty service-tier lists. Lower numeric `priority` values appear first. `GET /v1/models` flattens each entry's `slug` and `codex` mapping into a complete Codex `ModelInfo` object and never serializes `upstream`.

The two namespaces must remain distinct: `slug` is the client-visible value accepted in `POST /v1/responses`, while `upstream.model` is substituted immediately before the SDK call. This permits aliases and provider-specific model IDs without leaking base URLs, secret names, or routing choices through the model endpoint.

Do not put API keys directly in YAML or expose routing secrets from `/v1/models`. Reject duplicate slugs, invalid endpoints/providers, missing secret environment variables, unsafe upstream URLs, and catalog settings that advertise unsupported server features. Resolve the request's public `model` slug to one immutable route, replace it with the configured upstream model ID if necessary, and reject unknown models before contacting an upstream.

The `claude-sonnet-5` entry depends on the server-side prompt-cache policy described below. The native `gpt-5.6-luna` Responses route can pass `prompt_cache_key` through unchanged.

### Packaging and loading the catalog

`models.yaml` must be Molto package data, not a repository-root runtime file. The current setuptools build needs an explicit package-data declaration:

```toml
[tool.setuptools.package-data]
molto = ["models.yaml"]
```

Load the packaged default with `importlib.resources` so it works from an installed wheel as well as an editable checkout:

```python
from importlib.resources import files

catalog_resource = files("molto").joinpath("models.yaml")
with catalog_resource.open("rb") as catalog_file:
    catalog = yaml.safe_load(catalog_file)
```

Do not derive the default path from the current working directory or assume the source tree is present. The CLI should optionally accept an external catalog path, for example `--models /etc/molto/models.yaml` or `MOLTO_MODELS_FILE`, which replaces the packaged default for that process. Treat the packaged resource as read-only and never write resolved secrets or runtime changes back into it.

Packaging tests should build a wheel, inspect or install it, and assert that `importlib.resources.files("molto").joinpath("models.yaml")` exists and contains both required slugs. Loader tests should cover the packaged default and an external override independently.

The FastAPI and YAML dependencies must become an explicit install contract, preferably a `server` extra. Keep the test extra pinned according to repository policy.

## Responses request path

Codex currently sends `stream=true`, `store=false`, `include=["reasoning.encrypted_content"]`, `prompt_cache_key`, and `client_metadata` on ordinary HTTP turns. The request assembly is in [Codex's client](./__ref__/codex/codex-rs/core/src/client.rs), and metadata contents are defined in [responses metadata](./__ref__/codex/codex-rs/core/src/responses_metadata.rs).

Molto now handles two behavior-neutral settings on translated Responses routes:

- `include=["reasoning.encrypted_content"]` is accepted and ignored. Molto already preserves encrypted reasoning that the provider returns.
- `store=false` is accepted and ignored. `store=true` remains an error because translated endpoints cannot promise provider-side Responses state.

For `client_metadata`, the server should remove it from the body before `OpenAIResponsesCreateRequest` validation and pass it separately:

```python
client_metadata = body.pop("client_metadata", None)
extra_body = {"client_metadata": client_metadata} if client_metadata is not None else None
stream = await molto.responses(body=body, extra_body=extra_body, ...)
```

Only inject that field into a native Responses upstream known to accept it. Chat Completions and Anthropic Messages do not define `client_metadata`; on translated routes, retain it only for bounded tracing or discard it. Do not log it wholesale because its serialized turn metadata can contain workspace and tool context.

### Prompt caching

OpenAI Responses and Chat Completions both accept the same top-level `prompt_cache_key`. Chat Completions also accepts `prompt_cache_options`, so a Responses-to-Chat route can forward the Codex key directly rather than dropping it. See the current [OpenAI Chat Completions request fields](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions/methods/create).

Anthropic Messages offers prompt caching with different semantics. It has no caller-defined cache key. Instead, `cache_control` enables content-derived prefix caching:

- Top-level `cache_control: {"type": "ephemeral"}` enables automatic caching and advances the breakpoint as the conversation grows.
- Block-level `cache_control` marks explicit breakpoints.
- Cache identity derives from the matching `tools`, `system`, and `messages` prefix rather than from the value of Codex's key.
- The default lifetime is five minutes; a one-hour TTL is also available.

See [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) for the current behavior and supported models.

Molto's native Chat Completions and Messages request models already contain `prompt_cache_key` and `cache_control`, respectively. Cross-protocol normalization does not currently preserve either setting. The server should therefore resolve the model route before strict Responses validation and apply this policy:

| Selected upstream | Server behavior |
| --- | --- |
| OpenAI Responses | Leave `prompt_cache_key` in the request and forward it normally. |
| OpenAI Chat Completions | Remove it from the Responses body and add the same key to `extra_body`. |
| Anthropic Messages with automatic caching | Remove it from the Responses body, discard the key value, and add the route-configured `cache_control` to `extra_body`. |
| Anthropic Messages with caching disabled | Remove and ignore it. |

Mapping `prompt_cache_key` to Anthropic automatic caching is intentionally not a semantic 1:1 translation: the Codex-provided key is unused. Ignoring it is correctness-preserving because prompt caching affects cost and latency, not conversation history or model inputs. Make Anthropic caching an explicit route policy, as in the `claude-sonnet-5` example, because cache writes have their own pricing and TTL behavior.

When constructing `extra_body`, merge this cache setting with any compatible extension fields instead of allowing one to overwrite another. Never forward `client_metadata` to Chat Completions or Messages merely because both use the same SDK escape hatch.

Optional request fields such as `service_tier` and `text` still need a mapping or rejection policy before the catalog advertises them.

## Request headers

Pass the inbound header collection to the selected coordinator as `extra_headers`. Molto now removes credentials, account selectors, connection-specific fields, request framing, and proxy topology headers before invoking the upstream SDK. This prevents the client-facing `Authorization` value from replacing the configured provider key.

Safe operational headers that remain after filtering include Codex session, thread, window, parent, subagent, originator, beta-feature, routing, and turn-state headers. The relevant construction is in [Codex's response metadata](./__ref__/codex/codex-rs/core/src/responses_metadata.rs) and [request header definitions](./__ref__/codex/codex-rs/codex-api/src/requests/headers.rs).

If the server itself requires authentication, validate the inbound bearer token at the FastAPI boundary before passing headers to Molto. Upstream credentials must come only from the selected model route.

## Response headers

The returned `MoltoStream.response_headers` now retains successful upstream headers before the first event is returned. The server should copy useful end-to-end headers onto the initial FastAPI streaming response, including when present:

- `x-codex-turn-state`
- `x-request-id`
- `openai-model`
- `x-models-etag`
- `x-reasoning-included`
- Codex rate-limit and credit headers
- Codex safety-buffering headers
- `retry-after` on applicable failures

Codex consumes these in [Responses SSE handling](./__ref__/codex/codex-rs/codex-api/src/sse/responses.rs), [rate-limit parsing](./__ref__/codex/codex-rs/codex-api/src/rate_limits.rs), and [safety buffering](./__ref__/codex/codex-rs/codex-api/src/safety_buffering.rs).

Do not copy response `content-length`, `content-encoding`, `transfer-encoding`, or hop-by-hop headers after converting or re-streaming the body. Preserve upstream request IDs and retry information on non-2xx responses as well.

`x-codex-turn-state` is sticky routing state that Codex reads and sends back. Passing it in both directions is sufficient; Molto does not need to store it.

## SSE and lifecycle

Serialize each SDK event as a Responses SSE record, using its event type:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta",...}

```

Use `text/event-stream`, avoid proxy buffering, and close the upstream iterator when the client disconnects or the downstream generator is cancelled. Molto prefetches the first event, so request validation, connection failures, and upstream non-2xx errors normally occur before FastAPI starts the streaming response.

Map pre-stream failures to an OpenAI-compatible JSON error with an appropriate HTTP status. After headers have been sent, emit a valid `response.failed` event when the failure can be represented faithfully; otherwise terminate the stream cleanly. Never report an HTTP success with a fabricated completed event after an upstream failure.

Codex uses complete streamed lifecycle events, not just text deltas. Preserve event ordering, function calls, function argument deltas, usage, completion/incomplete status, and encrypted reasoning history. Continue to require `stream=true` until Molto implements non-streaming collection.

## State model

The server can remain stateless:

```text
Codex local history
    -> complete POST /v1/responses
    -> Molto server and selected upstream
    -> Responses SSE events
    -> Codex appends output to local history
```

Codex owns ordinary conversation history and sends tool calls, tool outputs, assistant messages, and reasoning continuation items again on later requests. With WebSockets disabled, it does not require a server-side WebSocket session. With the custom provider treated as non-OpenAI, compaction remains local. The server retains only immutable startup configuration, the model catalog, and shared HTTP client pools.

Do not implement conversation storage, `previous_response_id` lookup, or response retrieval for this baseline. Prompt caching and `x-codex-turn-state` are provider hints passed through each request, not Molto-owned history.

## Implementation checklist

1. Add the server dependency extra, package `python/molto/models.yaml` through setuptools, and implement packaged-default plus external-override loading with startup validation.
2. Add model slug lookup and upstream route selection without exposing secrets.
3. Implement optional inbound server authentication.
4. Implement `GET /v1/models`, its Codex-specific JSON schema, `client_version` acceptance, and optional `ETag`.
5. Implement streaming-only `POST /v1/responses` using `Molto.responses()`.
6. Extract `client_metadata` and inject it through `extra_body` only for compatible Responses upstreams.
7. Pass inbound headers through `extra_headers`; use the retained `response_headers` for filtered downstream headers.
8. Serialize typed Responses events to correct SSE frames and guarantee disconnect cleanup.
9. Translate pre-stream errors and define the post-start stream-failure behavior.
10. Reject `store=true`, unexpected hosted web search, unsupported tools, non-streaming requests, and unknown models explicitly.
11. Implement the route-specific `prompt_cache_key` policy: native Responses pass-through, direct Chat Completions forwarding, and configurable Anthropic `cache_control` or behavior-neutral removal.
12. Test catalog validation, installed-wheel package data, external catalog override, model routing, auth isolation, `client_metadata`, header filtering, response headers, SSE framing, upstream errors, cancellation, and every rejection above.
13. Run an end-to-end Codex smoke test covering a text turn, `exec_command`, `function_call_output` continuation, long local history/compaction, and any MCP or app behavior claimed by the catalog.

No `/alpha/search`, Responses WebSocket, remote compaction, or server-side conversation database is required for the recommended first release.
