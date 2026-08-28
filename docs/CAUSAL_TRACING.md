# Causal-tracing primitives

KEditAudit's causal-analysis layer is intentionally separated from model and
editor implementations. These primitives manage experimental mechanics; they
do not interpret a restored score as proof of semantic causality or model
safety.

## HookManager

`HookManager` owns forward-hook handles and removes them in reverse
registration order. It supports explicit `close()` and context-manager use.
Cleanup is attempted on normal exit and on exceptions.

- Observer hooks that return `None` leave tuple and model-output-like values
  unchanged.
- A hook replacement is returned exactly as supplied; the manager does not
  coerce tensor, tuple, or model-output types.
- If one handle removal fails, every remaining handle is still attempted and a
  `HookCleanupError` records all removal errors.
- If model execution and cleanup both fail, the original model exception is
  preserved and the cleanup failure is attached as an exception note.
- A closed manager cannot register or own additional hooks.

The current tests use offline fake modules. No Transformers hook integration or
model-specific activation shape is claimed yet.

## Clean/corrupt/restore coordinator

`run_causal_trace` coordinates an adapter-defined experiment in a fixed order:

1. resolve every requested module path before model execution;
2. tokenize the prompt and target and derive the subject span through the
   adapter tokenizer;
3. run clean scoring and capture exactly one activation per requested module;
4. call `create_corruption` once with the recorded seed and subject span;
5. pass that same corruption object to the corrupted run and every restoration
   run;
6. retain raw clean, corrupted, and restored target scores for each module.

For module \(m\), the JSON-ready evidence records:

\[
\Delta_m = S_{restored,m} - S_{corrupted}
\]

and, when the denominator is non-zero:

\[
R_m = \frac{S_{restored,m} - S_{corrupted}}
{S_{clean} - S_{corrupted}}.
\]

If clean and corrupted scores are equal, `recovery_fraction` is `null` and an
explicit warning is retained. Recovery values may be below zero or above one;
the coordinator does not clip or automatically interpret them.

`CausalTraceResult.as_dict()` emits ordered heatmap data with schema version,
model/tokenizer revisions, logical state ID, seed, tokenizer-derived subject
span, raw scores, reductions, and warnings. It deliberately excludes prompt
text, activation tensors, and the corruption tensor. The primary experimental
reference is [Locating and Editing Factual Associations in GPT / ROME](https://rome.baulab.info/).

This implementation is an offline orchestration contract tested with a fake
adapter. It does not yet run a Transformers model, establish that a restored
module semantically caused an answer, or support a production architecture.
