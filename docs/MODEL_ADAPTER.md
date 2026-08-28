# ModelAdapter contract and offline fake

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

This contract does not claim compatibility with Transformers, ROME, EasyEdit,
or any production model. Those require pinned adapters and integration tests.
