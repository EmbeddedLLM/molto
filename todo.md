# Responses Parameter to Handle

- Map role `developer` to `system`
- `stream_options.include_obfuscation`
- `ToolSearchCall`

# SDK Parity

- Add non-streaming support, the primary missing execution mode.
- Close the remaining literal SDK parity gaps:
  - Per-request `timeout`, `extra_headers`, `extra_query`, and `extra_body` arguments.
  - Raw-response and streaming-response SDK helpers.
  - Synchronous SDK interfaces.
