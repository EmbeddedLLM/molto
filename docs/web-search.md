# Hosted web search

Molto supports basic provider-hosted web search through Responses and Messages upstreams. Searches run at the provider, separately from client functions.

| Caller protocol  | Responses upstream                          | Messages upstream                   | Chat upstream             |
| ---------------- | ------------------------------------------- | ----------------------------------- | ------------------------- |
| Responses        | Native passthrough                          | Search, answer, and URL attribution | Rejected before transport |
| Messages         | Search, answer, and reduced URL attribution | Native passthrough                  | Rejected before transport |
| Chat Completions | Mandatory search, answer text               | Mandatory search, answer text       | Rejected before transport |

Both streaming and non-streaming calls use this contract. Native Responses and Messages preserve provider fields and events. Translated non-streaming results collect the same projected events as streaming calls.

Translation supports Responses `web_search` and Messages `web_search_20250305`. Other hosted tools, dynamic filtering, and fetch remain unsupported for translation.

## Requests and selection

Responses declares `{"type":"web_search"}` in `tools`. Messages declares `{"type":"web_search_20250305","name":"web_search"}`. Chat requests mandatory hosted search with `web_search_options: {}`. Choose an upstream model that supports search; Molto keeps the requested model.

| Setting | Translation |
| --- | --- |
| Allowed domains | Responses `filters.allowed_domains` <-> Messages `allowed_domains` |
| Approximate location | Country, city, region, and timezone; Chat unwraps `user_location.approximate` |
| Search budget | Responses `max_tool_calls` <-> Messages search `max_uses`, copied numerically |
| Context size | Preserved for Responses; explicit Responses/Chat `search_context_size` rejects when targeting Messages, including `medium` |
| Blocked domains | Native Messages preserves `blocked_domains`; translation to Responses rejects an explicit blocklist |
| Source inclusion | Translated searches to Responses request `web_search_call.action.sources`, preserving existing include values without adding duplicates |

Omitted options remain omitted; native Responses and Messages options pass through unchanged.

Responses and Messages preserve automatic, disabled, required, and supported named selection. Chat requires hosted search specifically, using Responses required `allowed_tools` restricted to search or Messages named `web_search` selection. Conflicting explicit Chat tool choices are rejected. A client function named `web_search` remains distinct from hosted search, but Messages cannot select between them by name; rename the function when using named selection.

A Messages upstream requires an output-token limit: supply `max_output_tokens` from Responses, `max_completion_tokens` from Chat, or `max_tokens` from Messages. Search budgets do not replace that limit.

Budget exhaustion remains provider-specific. A Messages `max_uses_exceeded` result follows the failed-search mapping below.

## Answers and attribution

Responses URL citations identify a range within a text part. Translated Messages splits that part at citation boundaries and attaches each citation to the corresponding blocks. Concatenating those blocks preserves the answer exactly, including whitespace. Overlapping ranges can associate one source with several blocks. Messages citations map back to ranges spanning their associated text blocks, with offsets local to each projected Responses part.

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

`cited_text` on this translated route contains the associated **answer range**, which may paraphrase sources. It is not a verified source quotation. `encrypted_index: "N/A"` means citation state is unavailable; it cannot restore provider continuation state. Messages-to-Responses translation loses the original excerpt and encrypted reference. Native Messages preserves both. Missing or null titles become `""` when the translated target requires a string.

Offsets use Python string characters without normalization. Recordings verify ASCII ranges; synthetic Unicode tests do not establish the provider's Unicode indexing convention.

Messages buffers text parts until their citations are complete, preserving part order. This can increase text latency and change chunk boundaries.

Translated Chat omits structured annotations in both modes. It preserves actual answer text and existing inline links, without inserting replacement citations or narration. Signature-only reasoning with no representable text is omitted on the hosted-search Chat route; other reasoning follows the existing protocol restrictions.

## Search activity and results

Responses search actions become Messages `server_tool_use` blocks followed by linked `web_search_tool_result` blocks. Emission waits until the action and arguments are known.

- A non-empty `queries` list produces one `server_tool_use` block per entry, in order. Repeated query strings remain separate invocations. Otherwise a present singular `query` supplies `input.query`.
- Missing or null query metadata, including an empty query list without a singular query, produces `input: {}`.
- An explicitly empty query remains `input: {"query":""}`.
- The first invocation retains the original search ID; additional invocations receive distinct generated IDs. The upstream result contains pooled sources without per-query attribution, so it appears once, linked to the original ID. Additional invocations have no separately attributable result. The same mapping applies when projecting search history to a Messages upstream.

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

Native usage is preserved. Token translation is unchanged. Messages search counts have no public equivalent on translated Responses or Chat and are omitted. When Responses explicitly reports `tool_usage.web_search.num_requests`, translated Messages copies that number to `usage.server_tool_use.web_search_requests` and supplies `web_fetch_requests: 0` for this search-only scope. Explicit zero remains zero; absent/null counts remain unknown. Counts are not derived from operations, queries, sources, citations, or replayed history, and repeated usage snapshots are updates rather than additions. This is a numeric compatibility approximation, not equivalent billing or accounting.

## Follow-ups and recovery

Keep the upstream and provider fixed across a conversation. Native routes preserve search history. Translated Responses/Messages routes map supported calls, results, and citations with the reductions above; Chat history retains only visible text and supported function calls. History can carry citations without redeclaring search.

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
