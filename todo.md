# Responses Parameter to Handle

- Map role `developer` to `system`
- `stream_options.include_obfuscation`
- `ToolSearchCall`

# SDK Parity

- Add non-streaming support, the primary missing execution mode.
- Close the remaining literal SDK parity gaps:
  - Per-request `timeout`, `extra_headers`, `extra_query`, and `extra_body` arguments.
  - The native `AsyncStream` object; Molto currently returns its own async iterator yielding native event objects.
  - Raw-response and streaming-response SDK helpers.
  - Other SDK resource operations such as retrieve, cancel, delete, and conversation management.
  - Synchronous SDK interfaces.
