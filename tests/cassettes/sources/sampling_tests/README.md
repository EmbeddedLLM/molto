# Sampling source recordings

These 12 unchanged sanitized recordings were captured on November 4, 2026 from the official OpenAI and Anthropic APIs. They replace the archived model-settings references with focused sampling evidence compatible with the repository's pinned SDKs.

| Provider  | Protocol                       | Requested model             | SDK / API version                                           |
| --------- | ------------------------------ | --------------------------- | ----------------------------------------------------------- |
| OpenAI    | Responses and Chat Completions | `gpt-4.1-mini`              | OpenAI Python `2.53.0`                                      |
| Anthropic | Messages                       | `claude-haiku-4-5-20251001` | Anthropic Python `0.121.0`, `anthropic-version: 2023-06-01` |

Source hosts are `https://api.openai.com` and `https://api.anthropic.com`. The Anthropic API version is explicit in the recording driver and matches the pinned SDK's default. The recorder's existing header whitelist does not retain that header. Sampling parameters are removed in Anthropic API version `2026-10-21`; these recordings establish acceptance on the documented older version with Haiku 4.5, not on every model or API version. See [Anthropic's parameter deprecation guidance](https://platform.claude.com/docs/en/about-claude/model-deprecations#api-parameter-deprecations).

## Scenario

Each protocol has independent SSE and JSON conversations, each with two genuine turns:

| Turn | Prompt                                           | Sampling controls                   |
| ---- | ------------------------------------------------ | ----------------------------------- |
| 00   | `Reply with one short sentence about the ocean.` | `temperature: 0.45`; `top_p` absent |
| 01   | `Reply with one short sentence about a forest.`  | `top_p: 0.55`; `temperature` absent |

Turn 01 carries the actual assistant output returned in turn 00. Responses retains native output items; Chat uses the SDK message without local parser properties or empty annotations; Messages retains the returned text blocks. Neither sampling control is inherited from a previous request. The generated answers can differ between protocols and wire modes.

All requests use an output-token limit of 128, expressed as `max_output_tokens`, `max_completion_tokens`, or `max_tokens`. There are no tools, reasoning options, or structured-output settings. Responses sets `store: false` and disables streaming obfuscation. Chat streaming requests enable usage reporting. Anthropic thinking is omitted so the selected model uses ordinary text generation.

## Recording and validation

A standalone SDK driver loaded the two credentials explicitly from the repository root `.env` and called `scripts/recorder_proxy.py` through `httpx.ASGITransport`. The proxy's HTTP client made the live requests to the official hosts. This ran outside pytest. Credentials were never printed; authorization was masked and the existing recorder whitelist excluded Anthropic's API-key header.

Every request returned HTTP 200 and completed normally. Validation checked:

- Exact recorded request bodies, sampling values, and absence of the unused control.
- All raw SSE events and JSON responses against the pinned native SDK schemas, without added fields or coercions.
- Normal termination, positive input/output token usage, and actual assistant-history continuation.
- Credential masking and absence of both configured credentials from every recording.
- Native Molto output against SDK-parsed events or responses for all six conversations, using offline mock transports and exact outgoing request assertions.
- SHA-256 hashes before and after replay, confirming that validation did not change any original.

The recorder-generated files were renamed without changing their bytes. No provider response field, usage count, text, identifier, or chunk boundary was synthesized. These originals are source-only and do not automatically join generic route discovery. Review and commit them before preparing separate matched companions under the [cassette workflow](../../README.md#specifications-and-tooling).

## Original hashes

| File                                         | SHA-256                                                            |
| -------------------------------------------- | ------------------------------------------------------------------ |
| `openai-responses-stream-00.yaml`            | `8a8f054c9635a555427233b9f9b1d4369782509d2ed6a56af4536764727e3ee7` |
| `openai-responses-stream-01.yaml`            | `5a3b8fb67ae4718b629b6e7928056a412bfe1dd52e3aae5feb78ca195ab9076f` |
| `openai-responses-non_stream-00.yaml`        | `6e472bcafd07d7c81f4d80d92652f5c009c5745d5053a6ed8589ec6d6c0f610b` |
| `openai-responses-non_stream-01.yaml`        | `7f331d43dbed310e23595180aec59b5edf7fdad2790270258699719143fcf0c3` |
| `openai-chat_completions-stream-00.yaml`     | `ce772c820f17e93e2742fd9073ad84834b4a67c07936e6b4619d7c0deaa96383` |
| `openai-chat_completions-stream-01.yaml`     | `d3ebf013922f125bee0a059392fc6f3bda104a0967d41cfeed1cc889bdaa55cc` |
| `openai-chat_completions-non_stream-00.yaml` | `603eeb95ef052fd9e8dbb12ec6e37d376bbe057eb3a839279292fedcafc2fb6f` |
| `openai-chat_completions-non_stream-01.yaml` | `91bfc7ec8e14c3d2eae02d2da8c8f018107452ccd2df429028fe576c14e568b4` |
| `anthropic-messages-stream-00.yaml`          | `76c1fdc8851108132cff36fdd8379f9843e1c3e7d5bfec6a4e4ffcca0d496fd9` |
| `anthropic-messages-stream-01.yaml`          | `1521cfaf5ce2560b69a590ac9d9f15ef15f13027f6b94b4e04f3337c7e2f3198` |
| `anthropic-messages-non_stream-00.yaml`      | `dd154ee49b9aa880c625cecc07a2a64a33f460e67989d848797aee6ac2793544` |
| `anthropic-messages-non_stream-01.yaml`      | `4fc570765f2fd633ab48d90f7708b0e6fbdb102879fa38e073b9f1a9605a5b29` |
