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
