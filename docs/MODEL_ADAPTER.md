# ModelAdapter contract, offline fake, and pinned GPT-2 adapter

Issue 7 introduces a small runtime-checkable `ModelAdapter` protocol. It keeps
the audit layer independent of a specific model library or editor and requires:

- immutable model, tokenizer, logical-state, device, and dtype metadata;
- deterministic tokenization;
- normalized target-sequence scoring;
- a tokenizer-derived, unambiguous subject token span;
- a module root for the separately tested path resolver.

`validate_adapter_pair` fails closed unless the baseline and edited adapters
declare different logical state IDs, the expected baseline/edited roles, and
the same model and tokenizer identities. This prevents two references to one
mutated state from being accepted as a comparison.

`FakeModelAdapter` is an offline test double. Callers must explicitly provide
every text-to-token mapping and every prompt/target logits fixture. The adapter
copies these inputs, rejects unknown text or score pairs, and delegates score
normalization to KEditAudit's existing target-sequence log-probability reducer.
It performs no model loading, network access, generation, or editing.

```python
from kedit_audit.adapters import FakeModelAdapter, ModelMetadata

adapter = FakeModelAdapter(
    metadata=ModelMetadata(
        model_id="synthetic/tiny-causal-lm",
        model_revision="revision-1",
        tokenizer_id="synthetic/tiny-tokenizer",
        tokenizer_revision="revision-1",
        state_id="baseline-state",
        state_kind="baseline",
        device="cpu",
        dtype="float32",
    ),
    token_ids_by_text={"prompt": [0], " target": [1]},
    logits_by_prompt_target={("prompt", " target"): [[0.0, 1.0]]},
    module_root=object(),
)
```

The protocol alone does not claim compatibility with Transformers, ROME,
EasyEdit, or any production model. Compatibility requires a pinned adapter and
an integration test.

## Pinned Transformers GPT-2 slice

Issue 11 adds `GPT2CausalLMAdapter` for one deliberately narrow integration:

- `transformers==5.16.1`;
- `torch==2.13.0`;
- exact `GPT2LMHeadModel` class with `config.model_type == "gpt2"`;
- CPU float32 parameters;
- one fast tokenizer whose vocabulary size matches the model;
- one Torch intra-op thread, set explicitly with `torch.set_num_threads(1)`.

The first slice also fails before model execution when text exceeds 4096
characters, the combined sequence exceeds `config.n_positions`, a contextual
target exceeds 64 tokens, or a custom GPT-2 vocabulary exceeds 65,536 entries.
These are explicit resource limits for the Python-list logits reduction path,
not claims about the maximum capacity of GPT-2 itself.

The adapter accepts an already-created model and tokenizer. It has no
`from_pretrained` helper, performs no Hub request, enables no remote code, and
does not infer model or tokenizer revisions. The caller must supply immutable
revision and logical-state provenance through `ModelMetadata`.

Target scoring tokenizes `prompt` and `prompt + target`, requires the prompt
tokens to remain an exact prefix, selects the causal logits aligned to the
contextual target suffix, and delegates log-probability normalization to the
existing offline reducer. An ambiguous tokenization boundary fails closed.
Subject spans come from fast-tokenizer character offsets rather than searching
for a separately tokenized subject subsequence.

The integration fixture creates a two-layer random GPT-2 and word-level fast
tokenizer entirely in memory. It verifies CPU scoring against an independent
Torch log-softmax calculation, offset-derived subject alignment, and the
`transformer.h.0.mlp` numeric module path without downloading a checkpoint.
Issue 12 adds a separate `GPT2CausalTraceAdapter` subclass for activation
experiments. Its exact corruption and restoration contract is documented in
[`CAUSAL_TRACING.md`](CAUSAL_TRACING.md); the base scorer remains usable without
requesting activation tracing.

```powershell
python -m pip install -e ".[dev,ml]"
python -m pytest -m integration tests/integration/test_transformers_gpt2_adapter.py
```

This slice follows the official
[Transformers GPT-2 model contract](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2)
and the pinned
[Transformers v5.16.1 release](https://github.com/huggingface/transformers/releases/tag/v5.16.1).
It does not load public GPT-2 weights, support GPU or reduced precision, audit
an edited checkpoint, or generalize activation tracing beyond verified GPT-2
block outputs.

## Safe module-path resolution

`resolve_module_path(root, path)` resolves validated dotted paths across
registered `_modules` mappings, ordinary mappings, plain instance fields, and
numeric sequence indices. This supports paths such as
`transformer.h.0.mlp` without importing Torch or hard-coding a layer number.

The resolver rejects private names, empty segments, negative indices,
ambiguous leading-zero indices, null children, missing modules, and out-of-range
sequence indices. It reads an instance's existing namespace directly and does
not execute property descriptors while traversing untrusted objects.
