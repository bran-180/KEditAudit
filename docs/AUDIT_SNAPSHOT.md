# AuditSnapshot data-only input contract

`AuditSnapshot` version `1.0.0` is the bounded local input used by the
Milestone 5 `audit` command. It describes measurements from exactly one logical
model state. It is not a checkpoint format and is never executed.

The authoritative Draft 2020-12 schema is
[`audit_snapshot.schema.json`](../src/kedit_audit/artifacts/audit_snapshot.schema.json).
Each snapshot records:

- `state`, either `baseline` or `edited`, plus a distinct `snapshot_id`;
- exact model, tokenizer, artifact, KEditAudit, editor, environment,
  generation, quantization, and seed provenance;
- one finite, non-positive mean target log-probability for every exact,
  paraphrase, locality, and portability probe in the AuditCase;
- aligned finite logits for every control probe and one positive temperature.

The JSON loader caps each file at 10 MiB and rejects duplicate keys,
non-UTF-8 input, non-standard numeric constants, and excessive nesting.
Schema limits additionally bound probe counts, positions, and vocabulary rows.
Probe IDs must be globally unique within a snapshot.

## Pair and AuditCase compatibility

The pipeline fails before writing a manifest unless:

- the two inputs declare the expected baseline and edited roles;
- snapshot IDs and model artifact IDs differ;
- comparable artifact hashes do not identify the same content;
- model/tokenizer identity, KEditAudit provenance, editor metadata, runtime
  environment, generation configuration, seeds, and control temperature match;
- target-score and control-logit IDs exactly cover the corresponding AuditCase
  groups in both snapshots.

These checks establish artifact-level logical separation. They cannot prove
that the upstream producer isolated two live model objects correctly. Snapshot
provenance is caller-supplied and must be reviewed like any other experimental
record.

Valid synthetic fixtures are
[`baseline.json`](../tests/fixtures/audit_snapshots/valid/baseline.json) and
[`edited.json`](../tests/fixtures/audit_snapshots/valid/edited.json). They contain
fictional measurements and no model weights, prompts, targets, or production
results.

## Security and privacy boundary

The command performs local JSON parsing and deterministic numeric reductions.
It does not import an external editor, deserialize a checkpoint, enable remote
code, contact a model hub, or upload data. Reports retain probe IDs and numeric
evidence but deliberately omit AuditCase prompts, subjects, and targets.

Input validation errors occur before a trustworthy run manifest can be frozen,
so no manifest is written for malformed or incompatible preflight inputs.
Once the validated running manifest is persisted, metric and report failures
replace it with a validated `failed` manifest whose public message excludes
exception text, tracebacks, prompts, and model outputs.
