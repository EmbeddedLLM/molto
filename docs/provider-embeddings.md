# Embeddings

`await Molto.embeddings(...)` accepts an OpenAI-shaped embedding request and returns an OpenAI SDK `CreateEmbeddingResponse`. It supports dense text embeddings through five explicit provider integrations. It does not stream, split batches, tokenize, truncate, pool vectors, or submit asynchronous jobs. An ordinary asynchronous Python call waits for one upstream HTTP response.

The implementation and offline coverage extend beyond the available live evidence. OpenAI and Cohere have successful live recordings. VoyageAI has authentication-error recordings and synthetic success coverage. Infinity and vLLM have pinned schema references and synthetic coverage, but **no live-tested server version or model**. These gaps remain part of [issue #93](https://github.com/EmbeddedLLM/molto/issues/93).

## Public contract

```python
from molto import Molto

async with Molto() as molto:
    result = await molto.embeddings(
        {
            "model": "text-embedding-3-small",
            "input": ["A document about planets", "A document about stars"],
            "dimensions": 256,
        }
    )
    vectors_by_input = {item.index: item.embedding for item in result.data}
```

The request body has the same body fields as the pinned OpenAI SDK: required `model` and `input`, and optional `dimensions`, `encoding_format`, and `user`. `OpenAIEmbeddingsCreateRequest` is available from `molto.types.openai_embeddings`. Inputs may be a string, a list of strings, one list of token IDs, or a batch of token-ID lists. Token IDs must be integers; Molto does not coerce strings, floats, or booleans into IDs. Cohere and VoyageAI support only the text forms in this coordinator.

`endpoint`, `provider`, and `base_url` have separate responsibilities:

| Argument | Meaning |
| --- | --- |
| `endpoint` | Selects `embeddings`, `cohere/embed`, or `voyage/embeddings`; defaults to `embeddings`. These are protocol identifiers, not literal URL paths. |
| `provider` | Selects `openai`, `cohere`, `voyageai`, `infinity`, or `vllm`. When omitted, defaults to the canonical provider for the endpoint. |
| `base_url` | Overrides only the destination and may include a deployment path prefix. Infinity and vLLM require it explicitly. |
| `api_key` | Overrides the configured provider key. Self-hosted routes use the non-secret `EMPTY` credential when omitted. |
| `options` | Accepts the matching typed `CohereEmbeddingOptions` or `VoyageEmbeddingOptions` on the corresponding native endpoint. |
| `extra_body` | Applies upstream body extensions after request projection. The effective body is checked for supported embedding semantics. |
| `extra_headers` | Adds request-scoped upstream headers after Molto removes credentials, account selectors, connection headers, and ingress transport headers. |

Changing a provider never silently changes the endpoint. For example, `provider="cohere"` requires `endpoint="cohere/embed"`. A custom URL does not imply compatibility with a different provider or protocol. OpenAI account environment headers are suppressed on compatible routes, and explicit credentials cannot be replaced through `extra_headers`.

## Evidence and support matrix

Evidence was collected on 2026-11-03. A schema reference is not a claim that its server release or any model was executed. Synthetic vectors are small hand-authored values that test transport and parsing; they are not embedding-quality evidence.

| Provider | Endpoint and route | Authentication | Model and version evidence | Covered boundary |
| --- | --- | --- | --- | --- |
| OpenAI | `embeddings`; `https://api.openai.com/v1/embeddings` | Bearer; `OPENAI_API_KEY` | Live `text-embedding-3-small`; parsed SDK 2.53.0 | Scalar/batched text, single/batched token IDs, omitted/float/base64 encoding, default 1536 and explicit 8 dimensions, `user`, native usage, 400/404 errors. |
| Cohere | `cohere/embed`; `https://api.cohere.com/v2/embed` | Bearer; `COHERE_API_KEY` | Live `embed-v4.0`, API v2; official native response schema | Text query/document, classification/clustering, `truncate` values `NONE`/`START`/`END`, 256 dimensions, float/base64, billed-unit metadata, model error. |
| VoyageAI | `voyage/embeddings`; `https://api.voyageai.com/v1/embeddings` | Bearer; `VOYAGE_API_KEY` | Live attempts returned 401; successful response behavior is synthetic, schema-based | Text query/document intent, dimensions, float/base64, truncation, source-backed token totals and native metadata. No successful live model verification. |
| Infinity | `embeddings`; caller-supplied base ending in `/v1`, then `/embeddings` | Configured bearer or `EMPTY` | Schema pinned to **3.0.0**, commit `09acbe5`; no live server/model test | Synthetic dense text and token-ID forms, dimensions, encoding defaults/overrides, `user`, usage, and native errors. |
| vLLM | `embeddings`; caller-supplied base ending in `/v1`, then `/embeddings` | Configured bearer or `EMPTY` | Schema pinned to **0.33.0**, commit `4ba787e`; configured server connection timed out | Synthetic dense text and token-ID forms, dimensions, encoding defaults/overrides, special-token/truncation controls, native indices, usage, IDs, timestamps, and errors. |

The live captures establish only the listed model/options, not every value accepted by the schema. Cohere accepted all three truncation options, but these short inputs do not demonstrate actual truncation outcomes, other dimensions, or model limits.

The self-hosted fixtures use `schema-test-model` deliberately. This is not a downloadable model or a server configuration example. Choose a deployed dense embedding model and verify its dimensionality, tokenizer, input limits, and task preparation against that deployment before relying on it.

## Encodings and response fidelity

On OpenAI-compatible routes, the SDK owns response parsing and Molto returns its response object directly. Omitted `encoding_format` follows the parsed SDK contract: the SDK requests base64 on the wire and decodes float32 data into Python floats. Explicit `float` preserves float vectors; explicit `base64` preserves the encoded strings. An `extra_body` encoding override changes the wire request but preserves SDK parsing behavior: when the typed `encoding_format` argument was omitted, the SDK still decodes base64 to floats; an explicit typed encoding disables that automatic decoding. For example, omitted typed encoding plus `extra_body={"encoding_format": "base64"}` still returns floats, while typed `encoding_format="float"` with that same override returns strings, just like the SDK.

Cohere and VoyageAI use their native wire formats. Molto maps only dense vectors into the OpenAI-shaped result. Omitted or explicit `float` returns float vectors; explicit `base64` returns base64 strings. On these translated routes, the effective encoding after `extra_body` overrides determines the output representation. Integer, binary, sparse, tensor, and multiple-vector results are unsupported, rather than coerced into dense float vectors. Non-float32 encodings and incompatible byte order are rejected on compatible routes.

Each vector retains its input index. Native indexed responses retain their response order and original indices, even if the upstream returns indices out of order; use `item.index` to associate vectors with inputs. Cohere's ordered vector list receives corresponding zero-based indices. Molto does not automatically split requests or combine inputs.

Native responses preserve the returned model, usage, and SDK extension fields. Translated results retain native response metadata under `provider_metadata`, including Cohere's ID, echoed texts, and `meta`, and VoyageAI's unmapped fields and vector metadata. When a native response omits the model name, as Cohere does, its result identifies the effective requested model; Molto does not invent a resolved model revision.

Missing token usage remains unknown. In particular, Cohere's `meta.billed_units` is preserved as billing metadata and is not relabeled as actual token usage; the OpenAI-shaped `usage` is `None`. VoyageAI's reported `total_tokens` is preserved, while `prompt_tokens` remains `None` when unavailable. These `None` values are intentional extensions to the SDK's static annotations. Do not replace them with zero or infer that vectors from different models inhabit the same space.

## Native task and truncation controls

```python
from molto import Molto
from molto.types.embeddings import CohereEmbeddingOptions, VoyageEmbeddingOptions

async with Molto() as molto:
    cohere_result = await molto.embeddings(
        {"model": "embed-v4.0", "input": ["A document about planets"], "dimensions": 256},
        endpoint="cohere/embed",
        options=CohereEmbeddingOptions(input_type="search_document", truncate="NONE"),
    )
    voyage_result = await molto.embeddings(
        {"model": "voyage-3.5", "input": ["A document about planets"]},
        endpoint="voyage/embeddings",
        options=VoyageEmbeddingOptions(input_type="document", truncation=False),
    )
```

The Voyage example describes the implemented API; its success has not been verified with the available credentials.

Cohere options use its native `input_type`: `search_query`, `search_document`, `classification`, or `clustering`. An input type is required. `truncate` accepts `NONE`, `START`, or `END`; omission leaves the upstream default. The input string or list becomes `texts`, `dimensions` becomes `output_dimension`, and the requested encoding selects one `embedding_types` entry.

Voyage options accept optional `input_type="query"` or `"document"` and optional boolean `truncation`. Molto preserves the input string/list shape, maps `dimensions` to `output_dimension`, and requests float output data. It does not treat Voyage's boolean truncation as equivalent to Cohere's directional truncation. Omitted options leave the provider defaults intact. Both translated routes reject `user` and token-ID input because no supported semantic mapping exists.

On vLLM, compatible native extensions such as `add_special_tokens`, `truncate_prompt_tokens`, and `truncation_side` may be supplied in `extra_body`; they are forwarded unchanged. This is provider-specific transport, not a portable query/document abstraction. Molto never inserts textual query prefixes, prepends titles, or changes token sequences to imitate an option. Unsupported typed fields fail validation; unknown native extensions remain the upstream service's responsibility.

## Modality and contextual boundaries

This coordinator supports one dense vector per text or supported token-ID input. It does not claim complete modality coverage for any provider.

| Provider | Modes outside this coordinator |
| --- | --- |
| OpenAI | Image, audio, file, conversation, and contextual-group input are not part of the embedding request contract. |
| Cohere | Embed v4 image-only and mixed text/image `inputs`, including multiple content parts combined into one vector, are unsupported here. Binary/integer outputs and multiple simultaneous representations are unsupported. |
| VoyageAI | Multimodal and contextual embedding endpoints, document/chunk grouping, file metadata, and integer/binary output types are unsupported. |
| Infinity | Conversation `messages`, multimodal content, tensor and multiple-vector outputs, and separate sparse/token-level endpoints are unsupported. |
| vLLM | Conversation/multimodal embeddings, token embeddings, late chunking, binary-response modes, and alternate embedding data types are unsupported. |

The request schema rejects non-text content objects and nested contextual string groups. Known unsupported modes in `extra_body` also fail explicitly. Output validation rejects vector matrices, tensor objects, and sparse structures without flattening or pooling them. An unrecognized provider extension is not a support claim for a new modality.

## Limits, errors, and lifecycle

Providers and models own batch-size, token, dimension, and truncation limits. Molto forwards supported requests as one call and preserves upstream errors; it does not enforce a stale local model catalog or silently split an oversized batch. OpenAI currently documents 8192 tokens per input, at most 2048 inputs, and 300000 tokens across a request. Cohere documents at most 96 texts per call. Voyage limits depend on the model and request token total. Self-hosted limits depend on the selected model and server configuration. Consult the linked API references for current values.

OpenAI-compatible HTTP errors retain the SDK error contract and pass through Molto's existing recognized-error translation. Native Cohere/Voyage HTTP failures surface in the OpenAI error namespace with the real upstream request, response, status, and body retained. Missing credentials fail locally before HTTP. Unsupported translation raises `UnsupportedInput`; malformed output raises an output validation error.

All routes borrow Molto's shared HTTP client and respect its lifecycle. `Molto.aclose()` closes that client. Configure HTTP timeouts on the injected `httpx.AsyncClient`. There is no embedding stream to close and no background job to poll.

## Sources and replay

- [OpenAI embedding API](https://developers.openai.com/api/reference/resources/embeddings/methods/create) and the repository-pinned OpenAI SDK define the public baseline.
- [Cohere Embed API v2](https://docs.cohere.com/reference/embed) defines its native task, encoding, and billing fields.
- [Voyage embeddings API](https://docs.voyageai.com/reference/embeddings-api) defines text embedding options and usage.
- [Infinity 3.0.0 OpenAPI schema](https://github.com/michaelfeil/infinity/blob/09acbe5cc421b2981d4385f0dbacc868bccd3ae7/libs/infinity-v3/docs/openapi.json) pins the self-hosted protocol reference.
- [vLLM 0.33.0 embedding protocol](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/entrypoints/pooling/embed/protocol.py), [shared request options](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/entrypoints/pooling/base/protocol.py), and [token embedding protocol](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/entrypoints/pooling/token_embed/protocol.py) distinguish supported dense requests from excluded modes.

Fixtures live under `tests/cassettes/embeddings/recorded/` and `tests/cassettes/embeddings/synthetic/`. Manifests state provenance, options, versions, and limitations. Recorded requests were made outside pytest, independently of Molto, and reviewed for secrets. Tests only replay fixtures, assert exact upstream bodies, and compare compatible results against parsed OpenAI SDK responses using the same fixture. They never compare independently generated vectors across providers or replace expected output with Molto's own output.
