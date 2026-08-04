# Molto server extension

This directory contains a separate, internal-use HTTP extension for Molto. It is intended for local development, integration environments, and testing. It is not hardened for untrusted public internet deployment.

The extension is not a package and has no installation metadata. It directly uses the Molto environment and does not add anything to the core library's installed package or commands.

## Requirements

Molto already supplies `httpx` and Pydantic. The server additionally imports FastAPI, Uvicorn, and Loguru. The repository's existing `test` extra supplies FastAPI and Uvicorn via its exact-pinned `fastapi[standard]` dependency:

```bash
python -m pip install -e '.[test]'
```

Loguru is intentionally not added to Molto's dependencies. Install it directly in the internal environment:

```bash
python -m pip install loguru
```

No separate server package installation is required.

## Start

The server binds to `127.0.0.1:8017` by default:

```bash
python extensions/server/run.py
```

Host and port can be overridden:

```bash
python extensions/server/run.py --host 127.0.0.1 --port 8017
```

Use `-v` (or `--verbose`) to log sanitized request and response payloads:

```bash
python extensions/server/run.py -v
```

Set the upstream bearer credential through the OpenAI SDK's standard environment variable:

```bash
export OPENAI_API_KEY="..."
```

Clients may instead pass a per-request API key, which the server forwards to the upstream provider. The key is accepted in the standard `Authorization: Bearer <key>` header or as an `api_key` field in the JSON body:

```bash
curl http://127.0.0.1:8017/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <key>' \
  -d '{"model":"gpt-5-mini","messages":[{"role":"user","content":"Hello"}]}'
```

Precedence: a body-level `api_key` wins over the `Authorization` header, which wins over the server's `OPENAI_API_KEY`. A request with no client-supplied key and no server-configured key fails with `401 authentication_error` (`missing_api_key`).

For vLLM or another operator-controlled OpenAI-compatible endpoint:

```bash
export MOLTO_SERVER_BASE_URL="http://127.0.0.1:8000/v1"
```

Unsupported request parameters are dropped and ignored by default. Enable strict mode to reject them instead:

```bash
python extensions/server/run.py --strict
# or, for an operator-controlled default:
export MOLTO_SERVER_STRICT="true"
```

The server does not require caller authentication, but callers may supply a per-request API key (see Start). The extension has no TLS, rate limiting, multi-tenancy, public audit controls, or configurable CORS.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /healthz`

The internal server intentionally has no tests. Keep test coverage at the core request, IR, provider, and output boundaries; do not add server tests.

The request path selects the OpenAI request and output protocol. The optional `X-Molto-Endpoint` header selects only the execution endpoint and accepts:

- `chat_completions`
- `responses`

Model names may be sent as either `openai/<model>` or a bare `<model>` name. The server treats a bare name as an OpenAI model and adds the `openai/` prefix.

The server forwards the optional `X-Molto-Endpoint` header as Molto's keyword-only `endpoint` argument. It is consumed before `InputIR` conversion. When omitted, the endpoint normally matches the request path; Anthropic models always default to their OpenAI-compatible Chat endpoint. GPT-5.6 Chat requests containing function tools automatically execute through Responses unless `reasoning_effort` is explicitly `none`. The HTTP response remains in Chat Completions format. An explicit `X-Molto-Endpoint` header overrides this automatic selection.

When the endpoint header is omitted for an OpenAI GPT-5.6 request with tools and active reasoning, the server selects `responses`. For automatically routed streams, Chat `stream_options.include_usage` is handled as an output-format option rather than forwarded to Responses. Setting it to `true` still produces the final Chat usage-only chunk. The Responses request builder omits neutral Chat defaults `frequency_penalty=0` and `presence_penalty=0`; non-zero penalties are dropped unless the server runs in strict mode, in which case they fail capability validation.

When a Responses request executes through Chat Completions, Molto removes Responses-only `include` and reasoning `summary` controls before capability validation. `store` remains representable on Chat; Anthropic's OpenAI compatibility endpoint accepts and ignores it. Responses hosted tools are not part of the Chat Completions contract and remain unsupported on this route.

For compatibility with clients that serialize generic harness request objects, the Chat endpoint removes null `reasoning_content`, `reasoning`, `tool_calls`, and `tool_call_id` message placeholders before role-specific validation. It also ignores the top-level `logit_bias`, `thinking_budget`, and `chat_template_kwargs` fields. Non-null unsupported message fields remain invalid, and the core Molto request and IR models remain strict.

Both endpoints ignore the top-level `client_metadata` field. Harnesses inject it as client transport metadata; it is not a provider parameter, and the server discards it before validation.

Chat Completions example:

```bash
curl http://127.0.0.1:8017/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Molto-Endpoint: chat_completions' \
  -d '{"model":"gpt-5-mini","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

The example executes through Chat Completions and returns a Chat Completions payload. Unsupported provider, endpoint, or feature combinations continue to fail through Molto's capability validation.

Chat Completions web-search example:

```bash
curl http://127.0.0.1:8017/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gpt-5-search-api",
    "messages":[
      {
        "role":"user",
        "content":"What are the latest OpenAI API announcements? Include source links."
      }
    ]
  }'
```

`gpt-5-search-api` is OpenAI's Chat Completions web-search model and always searches the web before responding. The response remains a Chat Completions payload; render any citation annotations returned with the assistant message as clickable links. This legacy Chat integration does not accept the Responses `tools: [{"type":"web_search"}]` shape or its web-search controls.

Chat Completions custom-function example:

```bash
curl http://127.0.0.1:8017/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gpt-5-mini",
    "messages":[
      {"role":"user","content":"What is the weather in Paris?"}
    ],
    "tools":[
      {
        "type":"function",
        "function":{
          "name":"get_weather",
          "description":"Get the current weather for a city.",
          "parameters":{
            "type":"object",
            "properties":{"city":{"type":"string"}},
            "required":["city"],
            "additionalProperties":false
          }
        }
      }
    ],
    "tool_choice":{"type":"function","function":{"name":"get_weather"}}
  }'
```

The server returns an assistant message with `finish_reason: "tool_calls"` and a `tool_calls` entry. Validate its `function.arguments`, execute `get_weather` in your application, then send both the assistant tool call and your result back to the server. Replace the example call ID and result with the values from your application:

```bash
curl http://127.0.0.1:8017/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gpt-5-mini",
    "messages":[
      {"role":"user","content":"What is the weather in Paris?"},
      {
        "role":"assistant",
        "tool_calls":[
          {
            "id":"call_example",
            "type":"function",
            "function":{"name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}
          }
        ]
      },
      {
        "role":"tool",
        "tool_call_id":"call_example",
        "content":"{\\"temperature_c\\":18,\\"conditions\\":\\"clear\\"}"
      }
    ]
  }'
```

Molto transports function calls but does not execute application-defined tools. Treat model-supplied arguments as untrusted input and validate them before calling your own service or code.

Responses example:

```bash
curl http://192.168.80.76:8017/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'X-Molto-Upstream: chat_completions' \
  -d '{"model":"gpt-5-mini","input":"Hello","stream":true}'
```

The example executes through Responses and returns a Responses event stream.

FastAPI documentation routes, CORS, proxy-header trust, and access logs are disabled. Put request-size and network controls in front of the extension if it is used anywhere beyond a trusted local environment.

## Diagnostic logging

The server writes colorized Loguru error entries by default. With `-v`, it also logs request and response payloads. Error entries include:

- request ID;
- public HTTP endpoint;
- requested and resolved `X-Molto-Upstream` value;
- the server error type, code, and offending parameter for rejected requests;
- pretty-printed request parameters when verbose logging is enabled;
- the actual exception type, message, and traceback for unexpected or streaming failures.

Authorization headers and provider credentials are never included. A body-level `api_key` is consumed before validation and never appears in logs; the `Authorization` header is never logged. Verbose request logs show the validated request parameters, which may contain prompts, tool definitions, or other sensitive content placed in the request body. Structured logs redact secret-like payload keys and bound very large strings such as base64 payloads. Raw errors are logged only on the server; client-facing error payloads remain sanitized.
