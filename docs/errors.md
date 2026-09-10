# Errors

Upstream clients normalize recognized SDK failures as `APIException`; downstream adapters project them into their SDK namespace. Native and translated routes use the same cause chain:

```text
downstream SDK exception
    caused by normalized Molto APIException
    caused by original upstream SDK exception
```

Inspect `error.__cause__` for the portable error category. Unknown errors pass through unchanged.

## Recognition rules

Recognition uses `isinstance`, including subclasses. Model-specific `404` rules take precedence over generic resource-not-found errors.

| Normalized exception | Upstream protocol | Upstream SDK exception | Recognition rule |
| --- | --- | --- | --- |
| `AuthenticationError` | all | protocol `APIStatusError` | status `401` |
| `AuthenticationError` | OpenAI Responses and Chat Completions | `openai.OpenAIError`, excluding `openai.APIError` | message starts with `Missing credentials.` |
| `AuthenticationError` | Anthropic Messages | `TypeError`, including subclasses | message starts with `"Could not resolve authentication method.` |
| `ContextLengthExceededError` | OpenAI Responses and Chat Completions | `openai.APIStatusError` | status `400` and `code == "context_length_exceeded"` |
| `ContextLengthExceededError` | OpenAI Responses | `openai.APIError` excluding `APIStatusError` | stream response status `200` and `code == "context_length_exceeded"` |
| `ContextLengthExceededError` | Anthropic Messages | `anthropic.APIStatusError` | status `400` and an SDK error message containing `prompt is too long:` |
| `ModelNotFoundError` | OpenAI Responses | `openai.APIStatusError` | status `400` and `code == "model_not_found"` |
| `ModelNotFoundError` | OpenAI Chat Completions | `openai.APIStatusError` | status `404` and `code == "model_not_found"` |
| `ModelNotFoundError` | Anthropic Messages | `anthropic.APIStatusError` | status `404` and an error message containing `model:` |
| `ResourceNotFoundError` | all | protocol `APIStatusError` | status `404` |

## HTTP errors

Molto preserves the upstream status, request, response, message, and body. Translated routes select the downstream SDK class from that status, not the normalized category. For example, an Anthropic model-not-found `404` becomes an OpenAI `NotFoundError` with the original Anthropic body.

For recognized HTTP errors on translated routes, select the outward class from the preserved upstream status:

| Downstream SDK | Status `400`      | Status `401`          | Status `404`    | Other statuses   |
| -------------- | ----------------- | --------------------- | --------------- | ---------------- |
| OpenAI         | `BadRequestError` | `AuthenticationError` | `NotFoundError` | `APIStatusError` |
| Anthropic      | `APIStatusError`  | `AuthenticationError` | `NotFoundError` | `APIStatusError` |

The normalized category carries portable meaning; the outward class reflects status. Native routes construct a fresh exception of the original SDK class, except for [missing credentials](#missing-sdk-credentials).

## Responses failures during streaming

A Responses request can succeed with HTTP `200` and then fail during stream iteration. In the recorded context-length case, the SDK raises plain `openai.APIError` on the SSE `error` event with `code="context_length_exceeded"`, before yielding `response.failed`. Watching only for `response.failed` misses this failure.

Molto exposes the downstream SDK's plain `APIError`, preserving request, message, and body. Its `ContextLengthExceededError` cause retains the real HTTP `200` response at `error.__cause__.response`; the outward error has no `status_code` or `response`. No failure status is synthesized. Catch errors while opening and consuming streams, and use the normalized category to decide whether to shorten context.

To retain the real response, `native_events()` passes `stream.response` to `_handle_error()` even though the SDK's plain `APIError` has no response.

## Missing SDK credentials

Molto deliberately raises the downstream SDK's `AuthenticationError` instead of the native SDK's local `OpenAIError` or `TypeError`, on every route. The SDK constructor requires HTTP metadata, so the error carries a synthetic `401` response and an unsent `POST` request targeting the effective upstream URL. The original message is preserved; the body and normalized cause's response remain `None`. Only this credential discrepancy synthesizes request/response metadata; no HTTP call occurs.

The upstream client constructs the intended request using its effective base URL and endpoint, preserving SDK path-prefix and trailing-slash behavior. The adapter wraps it in the synthetic response.

## Implementation and verification

See [error translation conventions](../python/CONVENTIONS.md#error-translation) for implementation and [error cassettes](../tests/cassettes/README.md#error-cassettes) for replay requirements. Non-streaming error-handler coverage is deferred.
