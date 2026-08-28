# Audit runner state contract

Issue 22 implements a manifest-first execution primitive for caller-owned audit
operations. It deliberately accepts an already constructed `running`
RunManifest and a callback; it does not load checkpoints, invoke ROME/EasyEdit,
or infer module paths.

`execute_audit` performs these transitions:

```text
validate running manifest
    -> atomically persist run-manifest.json
    -> execute caller-owned operation
    -> completed manifest on success
    -> failed manifest on an Exception
```

The output filename is fixed to `run-manifest.json`. Every state validates
against the versioned RunManifest schema before an atomic replace. The runner
rejects symbolic-link output targets, uses a same-directory temporary file,
flushes it before replacement, and removes a leftover temporary file when a
write fails.

Failure evidence records the configured public stage and exception type, but
not `str(exception)`, a traceback, prompt, output, or model value. The original
exception remains chained to `AuditExecutionError` for local handling. The
runner catches `Exception`, not `BaseException`, so process cancellation and
keyboard interrupts are not converted into ordinary audit failures.

The caller owns baseline/edited separation, evaluation, and the initial
manifest contents. A persisted completed state means that the callback
returned; it does not certify metric completeness or model safety. Issue 23
adds validated report writing, and Issue 25 will connect these primitives into
the end-to-end CLI.
