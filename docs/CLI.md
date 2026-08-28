# Command-line interface

Status: Issue 21 implements the dependency-light command parser and
`validate-case`. The `audit` and `compare` names are reserved by the parser but
remain unavailable until their issue-scoped implementations land.

```powershell
kedit-audit --help
kedit-audit validate-case path\to\case.json
python -m kedit_audit validate-case path\to\case.json
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
