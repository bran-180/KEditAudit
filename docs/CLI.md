# Command-line interface

Status: `validate-case`, data-only `audit`, and `compare` are implemented.

```powershell
kedit-audit --help
kedit-audit validate-case path\to\case.json
python -m kedit_audit validate-case path\to\case.json
kedit-audit audit --baseline baseline.json --edited edited.json `
  --case case.json --out reports\run-id
kedit-audit compare report-a.json report-b.json
```

The CLI uses `argparse` from the Python standard library, so help and case
validation do not import Torch, Transformers, NumPy, or a model adapter. JSON
inputs are capped at 10 MiB, must be UTF-8, reject duplicate keys and non-finite
numeric constants, and are validated with the packaged AuditCase contract.

A successful command prints only the schema version, public case ID, and
`status: valid`. Validation failures identify JSONPath-like locations while
avoiding the input value in the error output; prompts and targets are not
echoed. Exit code `0` means valid and exit code `2` means the file could not be
read or did not satisfy the contract.

`compare` validates both reports, requires matching case, baseline, model,
tokenizer, device, dtype, quantization, generation, and seed context, then
prints a versioned comparison JSON artifact to stdout. Aggregate deltas are
always `report_b - report_a`; no automatic improvement, regression, harm, or
safety conclusion is emitted.

## Data-only audit command

`audit` consumes two versioned [`AuditSnapshot`](AUDIT_SNAPSHOT.md) JSON files
and one AuditCase. Snapshots contain already-computed target scores and control
logits plus complete provenance; they are not checkpoints. The command never
loads a model, imports an external editor, enables remote code, or performs a
network request.

The pipeline validates both snapshots and exact AuditCase probe coverage before
creating output. It then persists `run-manifest.json` as `running`, computes
efficacy, generality, locality, portability, and directed control KL metrics,
validates the complete AuditReport, writes authoritative `audit-report.json`,
derives escaped `audit-report.md`, and finally persists the completed manifest.
Reports contain probe IDs and numeric evidence but do not copy case prompts,
subjects, or targets.

Input paths may not alias `run-manifest.json`, `audit-report.json`, or
`audit-report.md` in the output directory. Malformed, incompatible, or
path-overlapping inputs fail during preflight with exit code `2` and no
manifest, because trustworthy provenance could not be frozen. Failures after
the running manifest exists return exit code `1` and replace it with a
validated failed manifest that omits exception text and private input values.
Successful audit and comparison commands return exit code `0`.
