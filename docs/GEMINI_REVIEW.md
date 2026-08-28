# Review of the Gemini draft

## Capture context

The user asked Gemini to design an open-source project combining knowledge editing, mechanistic interpretability, and safety auditing. The conversation proposed **EditScope**, drafted an architecture, an application narrative, PyTorch hook/causal-tracing code, tests, and a script intended to generate repository files.

This repository records the useful knowledge but does not copy the generated prototype as trusted production code.

## Ideas retained

- Compare edit efficacy with generality, locality, ripple, and structural drift.
- Keep editing, forensic/audit metrics, schemas, visualization, and CLI concerns modular.
- Use clean/corrupted/restored runs for causal tracing.
- Export structured JSON plus human-readable reports.
- Develop through small TDD milestones.
- Provide CPU-friendly toy-model tests before large-model integration.
- Use Apache-2.0 as the proposed public license.

## Important corrections

### 1. Name collision

“EditScope” is already used in current knowledge-editing research. The project
uses **KEditAudit** to avoid misleading association. A public availability check
was completed on 2026-08-16 and is recorded in `NAME_AND_CITATION.md`; it must be
repeated before publication because it does not reserve a namespace.

### 2. Product scope

The draft attempted to implement ROME, MEMIT, LoRA, causal tracing, auditing, visualization, and multi-model support at once. Mature projects such as EasyEdit already cover broad editor integration. The revised scope is an audit-first layer with adapters.

### 3. ROME formula

The generated `compute_rome_delta_weight` function is an illustrative rank-one update, not the full ROME algorithm. It omits target-value optimization, context construction, covariance regularization details, layer selection, token alignment, and model-specific conventions.

### 4. Unverified layer tables

The Gemini draft listed fixed “typical injection layers” for GPT-2 XL, LLaMA-3, and Qwen. These values were not backed by pinned model revisions and integration tests. They must not appear as supported defaults.

### 5. Causal-tracing randomness

The generated tracer sampled new random noise in every restoration run. That confounds restoration effects with different corruptions. A valid paired experiment must reuse the exact same corruption tensor.

### 6. Module path resolution

The generated `_get_module()` used repeated `getattr()`. Paths such as `layers.0` require numeric indexing or a tested framework resolver; plain `getattr(module_list, "0")` is not reliable.

### 7. Hugging Face output assumptions

The prototype treated model results as either tensors or tuples. Transformers commonly returns structured `ModelOutput` objects, and internal modules can also return tuples. Adapters must normalize outputs without changing their types accidentally.

### 8. Baseline contamination

The README example passed `baseline_model=model` while applying an in-place edit to the same object. This can make “baseline” and “edited” references identical. Baseline scores or state must be captured independently.

### 9. API and file-tree inconsistencies

- Several tree entries used `init.py` instead of `__init__.py`.
- The quick start called methods that were not implemented, such as report export.
- The packaging text mixed Poetry-specific configuration with a claim of generic PEP 621 compliance.
- The test asserted cleanup only after leaving the context, but did not verify cleanup on exceptions.

### 10. Overstated safety and forensic claims

Locality, perplexity, KL divergence, or weight norms do not prove that a model is safe, aligned, or free of backdoors. “Forensic” should be presented as an audit metaphor, not a legal or certification claim.

### 11. Token and scale claims

The application draft proposed a 10,000-entity synthetic benchmark and fixed token-budget ratios before an MVP, dataset license review, or cost measurement existed. The revised application plan asks for resources tied to measured maintainer workflows and reproducible experiments.

## Decision

Do not run the Gemini one-file generator. Use the reviewed architecture and implement from Issue 1 in `ROADMAP.md`, preserving citations, deterministic tests, and explicit evidence boundaries.
