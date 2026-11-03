# Chat image tool-result recording

These two files are faithful, sanitized recordings from the official OpenAI Chat Completions API at `https://api.openai.com/v1/chat/completions`. They were recorded on November 3, 2026, outside pytest, using `scripts/recorder_proxy.py` with an in-process `httpx.ASGITransport` driver. The recorder's HTTP client made the live requests. The developer authorized the recording and use of the repository's `.env` credentials; credentials are not included in these files.

The requested model was `gpt-4.1-mini`; the provider returned `gpt-4.1-mini-2025-04-14`. Both requests stream with `max_completion_tokens: 4096`, usage reporting, and parallel tool calls enabled. OpenAI SDK 2.53.0 parsed the streams. No response fields, tool IDs, timestamps, model names, usage counts, or chunk boundaries were matched to another protocol.

- `openai-chat_completions-stream-00.yaml` asks for `read_image('a.png')` and `read_image('b.png')` in parallel. It returns two calls, 15 chunks, and `finish_reason: tool_calls`.
- `openai-chat_completions-stream-01.yaml` carries the actual SDK-collected assistant calls. Both text-only tool messages precede the labeled image user messages, which reuse the image data from the existing Responses scenario. The final user asks what the logos are. The response has 36 chunks, no further calls, and `finish_reason: stop`.

Chat tool messages cannot contain images. This source demonstrates the supported relocation shape: `assistant, tool A, tool B, user image A, user image B, user question`. Portable call IDs remain linked to their tool results and image labels. It establishes acceptance of these exact requests by the recorded OpenAI model; it does not establish acceptance by other providers or models.

Both streams include one usage chunk and a final `[DONE]` marker. The recorder masks authorization and retains only its configured header whitelist. Recorded `content-length` values describe the original transport and are not recalculated after sanitization.

The originals are retained separately from deliberately matched fixtures. Matching must operate on companions after the originals have been reviewed and committed.

| Original                                 | SHA-256                                                            |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `openai-chat_completions-stream-00.yaml` | `1776fa3d366845fa964be447c8d13d1d63d2d67f79be990a581a50b369457e89` |
| `openai-chat_completions-stream-01.yaml` | `9a0ff0d4af27e3d6f7c18f8c3516dcabca10bd94f8cd3e67ab08bbf12692f2a6` |
