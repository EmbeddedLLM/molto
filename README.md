# Molto

A unified LLM-provider translation library for Python. Call any LLM through a single, consistent interface.

## Core concepts

| Concept    | Purpose                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `endpoint` | Selects the wire protocol: Responses, Chat Completions, or Messages.                                                              |
| `provider` | Identifies the implementation serving that protocol: OpenAI, Anthropic, or vLLM. It also selects the provider capability profile. |
| `base_url` | Selects only where HTTP requests are sent.                                                                                        |

These settings are independent. In particular, Molto never infers provider behavior from a URL hostname. The explicit provider determines how ambiguous reasoning fields in Chat Completions and Messages are interpreted.
