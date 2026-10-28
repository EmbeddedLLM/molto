# Hosted web search

Molto preserves provider-hosted web search on native routes and translates basic search through Responses and Messages upstreams. Searches run at the provider, separately from client functions.

| Caller protocol  | Responses upstream                          | Messages upstream                   | Chat upstream             |
| ---------------- | ------------------------------------------- | ----------------------------------- | ------------------------- |
| Responses        | Native passthrough                          | Search, answer, and URL attribution | Rejected before transport |
| Messages         | Search, answer, and reduced URL attribution | Native passthrough                  | Rejected before transport |
| Chat Completions | Mandatory search, answer text               | Mandatory search, answer text       | Native passthrough        |

Both streaming and non-streaming calls use this contract. Native routes preserve provider fields and events. Translated non-streaming results collect the same projected events as streaming calls.

Translation supports Responses `web_search` and Messages `web_search_20250305`. Other hosted tools, dynamic filtering, and fetch remain unsupported for translation.

## Requests and selection

Responses declares `{"type":"web_search"}` in `tools`. Messages declares `{"type":"web_search_20250305","name":"web_search"}`. Chat enables search with `web_search_options: {}`; translated Chat requests require hosted search. Choose an upstream model that supports search; Molto keeps the requested model.

| Setting | Translation |
| --- | --- |
| Allowed domains | Responses `filters.allowed_domains` <-> Messages `allowed_domains` |
| Approximate location | Country, city, region, and timezone; Chat unwraps `user_location.approximate` |
| Search budget | Responses `max_tool_calls` <-> Messages search `max_uses`, copied numerically |
| Context size | Preserved for Responses; explicit Responses/Chat `search_context_size` rejects when targeting Messages, including `medium` |
| Blocked domains | Native Messages preserves `blocked_domains`; translation to Responses rejects an explicit blocklist |
| Source inclusion | Translated searches to Responses request `web_search_call.action.sources`, preserving existing include values without adding duplicates |

Omitted options remain omitted; native options pass through unchanged. Native Chat forwards `web_search_options` supplied in either the request or `extra_body`. Translated routes targeting Chat reject hosted search before transport, including search enabled through `extra_body.web_search_options`.

On translated routes, hosted search supplied through `extra_body.tools` uses the same output policy as search declared in the downstream request. When `extra_body.tools` replaces the projected tools, the effective upstream declarations determine whether Chat omits signature-only reasoning.

Responses and Messages preserve automatic, disabled, required, and supported named selection. Translated Chat requests require hosted search specifically, using Responses required `allowed_tools` restricted to search or Messages named `web_search` selection. Conflicting explicit Chat tool choices are rejected on translated routes. A client function named `web_search` remains distinct from hosted search, but Messages cannot select between them by name; rename the function when using named selection.

A Messages upstream requires an output-token limit: supply `max_output_tokens` from Responses, `max_completion_tokens` from Chat, or `max_tokens` from Messages. Search budgets do not replace that limit.

Budget exhaustion remains provider-specific. A Messages `max_uses_exceeded` result follows the failed-search mapping below.

## Answers and attribution

Responses URL citations identify a range within a text part. Translated Messages splits that part at citation boundaries and attaches each citation to the corresponding blocks. Concatenating those blocks preserves the answer exactly, including whitespace. Overlapping ranges can associate one source with several blocks. Messages citations map back to ranges spanning their associated text blocks, with offsets local to each projected Responses part.

Messages rejects empty and whitespace-only text blocks in conversation history. Citation projection folds blank fragments into the preceding block, or into the first following block for leading whitespace. Merging can widen a citation's associated block, while its original excerpt remains unchanged. Zero-length annotations remain representable in the normalized output and attach to the block containing their offset; a zero-length annotation at the end of the answer attaches to the final block. Their fallback `cited_text` remains empty because they cover no answer characters. An entirely blank text part has no valid Messages block that can preserve its content and attribution, so projection raises `ValueError`; assistant-history requests reject it with `UnsupportedInput` before transport.

For example, this upstream Responses text part contains answer `"Alpha. Beta."` with a citation on `[7, 12)`:

```json
{
  "type": "output_text",
  "text": "Alpha. Beta.",
  "annotations": [
    {
      "type": "url_citation",
      "start_index": 7,
      "end_index": 12,
      "url": "https://example.com/b",
      "title": "Source B"
    }
  ]
}
```

It becomes these illustrative Messages content blocks:

```json
[
  { "type": "text", "text": "Alpha. " },
  {
    "type": "text",
    "text": "Beta.",
    "citations": [
      {
        "type": "web_search_result_location",
        "url": "https://example.com/b",
        "title": "Source B",
        "cited_text": "Beta.",
        "encrypted_index": "N/A"
      }
    ]
  }
]
```

`cited_text` on this translated route contains the associated **answer range**, which may paraphrase sources. It is not a verified source quotation. `encrypted_index: "N/A"` means citation state is unavailable; it cannot restore provider continuation state. The Anthropic Messages-to-Responses route retains the original excerpt and encrypted reference in the [provider metadata extension](#preserving-anthropic-search-history). Native Messages preserves both directly. Missing or null titles become `""` when the translated target requires a string.

Offsets use Python string characters without normalization. Recordings verify ASCII ranges; synthetic Unicode tests do not establish the provider's Unicode indexing convention.

Messages buffers text parts until their citations are complete, preserving part order. This can increase text latency and change chunk boundaries.

Translated Chat omits structured annotations in both modes. It preserves actual answer text and existing inline links, without inserting replacement citations or narration. Signature-only reasoning with no representable text is omitted on the hosted-search Chat route; other reasoning follows the existing protocol restrictions.

## Search activity and results

In translated output, Responses search actions become Messages `server_tool_use` blocks followed by linked `web_search_tool_result` blocks. Emission waits until the action and arguments are known.

- A non-empty `queries` list produces one `server_tool_use` block per entry, in order. Repeated query strings remain separate invocations. Otherwise a present singular `query` supplies `input.query`.
- Missing or null query metadata, including an empty query list without a singular query, produces `input: {}`.
- An explicitly empty query remains `input: {"query":""}`.
- The first invocation retains the original search ID; additional invocations receive distinct generated IDs. The upstream result contains pooled sources without per-query attribution, so it appears once, linked to the original ID. Additional invocations have no separately attributable result.

Observed Responses sources become result entries. A URL-only source produces this illustrative result:

```json
{
  "type": "web_search_tool_result",
  "tool_use_id": "ws_1",
  "content": [
    {
      "type": "web_search_result",
      "url": "https://example.com/b",
      "title": "",
      "encrypted_content": "N/A"
    }
  ]
}
```

Available titles and `page_age` are preserved; unavailable age is omitted or null. The result placeholder supplies neither retrieved content nor valid continuation state. Native Messages retains original encrypted content. An explicitly empty source list becomes an empty result list. Missing sources remain unknown and are never reconstructed from answer citations. An unfinished invocation gets no invented successful result.

Translated Messages omits Responses `open_page` and `find_in_page` operations, retaining the answer and citations.

## Failures, pauses, and usage

An individual search failure is separate from an API request failure and from overall answer completion:

| Direction               | Failed-search representation                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| Messages -> Responses   | The linked `web_search_call` has `status: "failed"`; the Messages error code has no equivalent |
| Responses -> Messages   | A linked result error uses `error_code: "unavailable"` when no cause is available              |
| Either upstream -> Chat | Search-error blocks are omitted; actual model content and overall finish state are retained    |

`unavailable` is a fallback, not a provider diagnosis. Chat returns the actual answer, or empty content if none is representable, with the mapped finish reason. Search-result errors remain content; HTTP failures follow the [request error contract](errors.md).

A Messages pause has these terminal values:

| Downstream | Terminal state                                                                                |
| ---------- | --------------------------------------------------------------------------------------------- |
| Messages   | `stop_reason: "pause_turn"`                                                                   |
| Responses  | `status: "incomplete"`, `incomplete_details: null`; streaming ends with `response.incomplete` |
| Chat       | `finish_reason: "length"`                                                                     |

A paused search can remain `in_progress` without a result. Chat's `length` approximates unfinished generation; increasing the token limit may not resolve the pause. Continuation is caller-managed.

A Responses-source incomplete response without a reason normalizes to `unknown` and arrives as Messages `end_turn` or Chat `stop`. A hosted-search pause in that form is indistinguishable from any other reason-less incomplete response, so callers lose the `pause_turn` continuation hint and must manage continuation themselves.

Native usage is preserved. Token translation is unchanged. Messages search counts have no public equivalent on translated Responses or Chat and are omitted. When Responses explicitly reports `tool_usage.web_search.num_requests`, translated Messages copies that number to `usage.server_tool_use.web_search_requests` and supplies `web_fetch_requests: 0` for this search-only scope. Explicit zero remains zero; absent/null counts remain unknown. Counts are not derived from operations, queries, sources, citations, or replayed history, and repeated usage snapshots are updates rather than additions. This is a numeric compatibility approximation, not equivalent billing or accounting.

## Preserving Anthropic search history

`Molto.responses(endpoint="messages", provider="anthropic")` retains observed Anthropic search state in a typed `provider_metadata.anthropic` extension. Omitting `provider` selects Anthropic by default for this endpoint. This is a Molto extension to Responses, not a standard OpenAI field.

| Responses location                 | Fields in `provider_metadata.anthropic`                                 |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `web_search_call.action.sources[]` | Original `encrypted_content`, plus `title` and `page_age` when supplied |
| Text `annotations[]` URL citations | Original `encrypted_index` and `cited_text`                             |

For example, an observed search source retains its opaque state alongside the standard URL:

```json
{
  "type": "url",
  "url": "https://example.com/page",
  "provider_metadata": {
    "anthropic": {
      "encrypted_content": "<original encrypted content>",
      "title": "Original source title"
    }
  }
}
```

Streaming annotation events, completed search items, and the terminal response preserve the extension. Non-streaming responses use that same projection. Clients must retain the complete output items through serialization and replay them in `body.input`:

```python
history.extend(item.model_dump(exclude_none=True) for item in response.output)
```

On the next request, Molto restores the original Messages source and citation fields without forwarding `provider_metadata` itself. Keep the upstream destination and provider fixed. History containing this extension is accepted only with `endpoint="messages"` and `provider="anthropic"`; other routes raise `UnsupportedInput` before transport. Unknown metadata keys and malformed values fail request validation. The extension does not recover state discarded by a client or create Anthropic state for a search originally served by OpenAI.

The extension is supported in the typed Responses request body. `extra_body` remains a caller-owned upstream transport override and bypasses this normalization, as it does for other fields. Normal projection to an OpenAI upstream continues to use standard Responses fields only.

An [independent continuation recording](../tests/cassettes/README.md#anthropic-hosted-search-recordings) confirms HTTP `200` with `end_turn` after replaying the original search through this extension and restoring its encrypted source content and citation references.

## Follow-ups and recovery

Keep the upstream and provider fixed across a conversation. Native routes preserve search history. The Anthropic Responses route preserves opaque search state when clients retain its metadata extension; other translated Responses/Messages history follows the reductions above. Translated Chat history retains only visible text and supported function calls. History can carry citations without redeclaring search. When targeting Responses, citation buffering follows the effective upstream history, including replacement history supplied through `extra_body.input`.

Projection to a Messages upstream rejects search-history calls containing more than one query with `UnsupportedInput`. Responses pools results across those queries; expanding that history into separate Messages calls would leave calls without matching results. Single-query history preserves the call/result link.

When projecting Messages history to Responses, a search result updates the preceding call with the same ID even when they occupy separate assistant messages. The call preserves its query metadata while applying the result's status and supplied sources. Assistant text messages and text runs around search blocks retain their boundaries.

For a Messages response named `response`, preserve supported assistant blocks with:

```python
assistant_content = [block.model_dump(exclude_none=True) for block in response.content]
```

The SDK/API decides whether reduced history, including `"N/A"` placeholders, is usable. If it rejects replay, the caller can retry with text-only history:

```python
assistant_content = "".join(block.text for block in response.content if block.type == "text")
```

Text-only history preserves the answer but loses retrieved context; another search may be needed. Molto does not automatically resume paused turns, strip history, or retry rejected continuation.
