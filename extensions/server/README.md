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

For vLLM or another operator-controlled OpenAI-compatible endpoint:

```bash
export MOLTO_SERVER_BASE_URL="http://127.0.0.1:8000/v1"
```

The server does not require caller authentication. The extension has no TLS, rate limiting, multi-tenancy, public audit controls, or configurable CORS.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /healthz`

The request path selects the OpenAI request and output protocol. The optional `X-Molto-Upstream` header selects only the execution endpoint and accepts:

- `chat_completions`
- `responses`

Model names may be sent as either `openai/<model>` or a bare `<model>` name. The server treats a bare name as an OpenAI model and adds the `openai/` prefix.

When omitted, the upstream normally matches the request path. GPT-5.6 Chat requests containing function tools automatically execute through Responses unless `reasoning_effort` is explicitly `none`. The HTTP response remains in Chat Completions format. An explicit `X-Molto-Upstream` header always overrides this automatic selection.

When the header is omitted for an OpenAI GPT-5.6 request with tools and active reasoning, the server injects `X-Molto-Upstream: responses` before using the normal header-routing path. For automatically routed streams, Chat `stream_options.include_usage` is handled as an output-format option rather than forwarded to Responses. Setting it to `true` still produces the final Chat usage-only chunk. The Responses request builder omits neutral Chat defaults `frequency_penalty=0` and `presence_penalty=0`; non-zero penalties are rejected because the Responses SDK cannot represent them.

For compatibility with clients that serialize generic harness request objects, the Chat endpoint removes null `reasoning_content`, `reasoning`, `tool_calls`, and `tool_call_id` message placeholders before role-specific validation. It also ignores the top-level `logit_bias`, `thinking_budget`, and `chat_template_kwargs` fields. Non-null unsupported message fields remain invalid, and the core Molto request and IR models remain strict.

Chat Completions example:

```bash
curl http://127.0.0.1:8017/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Molto-Upstream: chat_completions' \
  -d '{"model":"gpt-5-mini","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

The example executes through Chat Completions and returns a Chat Completions payload. Unsupported provider, upstream, or feature combinations continue to fail through Molto's capability validation.

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
- pretty-printed request parameters when verbose logging is enabled;
- the actual exception type, message, and traceback for unexpected or streaming failures.

Authorization headers and provider credentials are never included. The verbose Chat request log contains the exact raw JSON body and is intentionally not redacted; it may therefore contain prompts, tool definitions, or credentials placed in the request body. Other structured logs redact secret-like payload keys and bound very large strings such as base64 payloads. Raw errors are logged only on the server; client-facing error payloads remain sanitized.
