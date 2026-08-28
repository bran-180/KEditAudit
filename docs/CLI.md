# Command-line interface

Status: `validate-case` and `compare` are implemented. The `audit` name is
reserved by the parser but remains unavailable until Issue 25 connects the
runner, adapters, and report writer.

```powershell
kedit-audit --help
kedit-audit validate-case path\to\case.json
python -m kedit_audit validate-case path\to\case.json
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
