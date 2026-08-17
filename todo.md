# Responses Parameter to Handle

- Map role `developer` to `system`
- `stream_options.include_obfuscation`
- `ToolSearchCall`
- `logprobs` and `top_logprobs`

# Cross-Endpoint Stream Translation

- Handle or reject Responses reasoning-summary events before projecting to Chat; Molto currently requests `summary="detailed"` and then fails on `response.reasoning_summary_part.added`.
- Preserve Chat `finish_reason="length"` and project it as Responses `response.incomplete` with `incomplete_details.reason="max_output_tokens"`.
- Map Responses `response.incomplete` with reason `max_output_tokens` to Chat `finish_reason="length"` instead of rejecting the terminal event.