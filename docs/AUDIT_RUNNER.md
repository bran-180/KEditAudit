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
    -> construct completed manifest
    -> optionally finalize report with that terminal manifest
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

The optional `finalize(value, completed_manifest)` callback runs while the
standalone manifest still says `running`. This lets the report writer embed the
exact completed manifest before the terminal manifest is persisted. A
finalizer exception follows the same failed-manifest path as an evaluation
exception. Cross-file writes are not a filesystem transaction: if a later
write fails, an already written authoritative JSON report can remain beside a
failed standalone manifest and must not be presented as a completed run.

The caller owns baseline/edited separation, evaluation, and the initial
manifest contents. The data-only Issue 25 pipeline supplies those pieces from
validated AuditSnapshots. A persisted completed state means that computation,
report validation, and requested report writes returned; it does not certify
metric meaning, upstream model-state isolation, or model safety.
