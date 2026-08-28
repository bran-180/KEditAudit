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

The hook-manager tests use offline fake modules. A separate marked integration
test exercises the same lifecycle against pinned GPT-2 embedding and block
modules, including cleanup after a deliberately invalid restoration tensor.

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

## Pinned GPT-2 activation experiment

`GPT2CausalTraceAdapter` connects the coordinator to the exact model and
dependency versions documented in [`MODEL_ADAPTER.md`](MODEL_ADAPTER.md). Its
supported intervention is deliberately narrow:

- trace paths must resolve to exact GPT-2 blocks and have the form
  `transformer.h.<index>`;
- the clean run captures the block's primary hidden-state tensor;
- a local CPU generator produces standard-normal float32 noise with fixed
  standard deviation `1.0`, so corruption creation does not advance Torch's
  global RNG state;
- the same noise tensor is added only to the tokenizer-derived subject token
  embeddings in the corrupted and every restoration run;
- one restoration run replaces the corresponding subject positions at one
  block with their detached clean hidden states;
- target scores are target-sequence summed natural log-probabilities from the
  existing authoritative reducer.

The emitted heatmap artifact is the ordered `modules` array from
`CausalTraceResult.as_dict()`. Each row retains the module path and raw clean,
corrupted, and restored scores plus restoration delta and recovery fraction.
It contains no prompt, target, subject text, activation, or corruption tensor.

The integration fixture uses random tiny weights and an in-memory tokenizer;
it is a mechanics and reproducibility test, not evidence about factual
knowledge in public GPT-2 weights. Restoration scores can be negative, exceed
the clean score, or fail to recover behavior. They must not be interpreted as
proof that a block semantically caused an answer, that an edit is safe, or that
the model has no hidden harmful behavior.
