# AuditCase contract

`AuditCase` is the versioned, editor-independent input that describes one
intended knowledge edit and the prompts used to audit its effects. The
authoritative machine-readable contract is
[`audit_case.schema.json`](../src/kedit_audit/artifacts/audit_case.schema.json).

## Version 1.0.0

Every case contains five top-level fields:

- `schema_version`: exactly `1.0.0`;
- `case_id`: a stable identifier using letters, numbers, `.`, `_`, or `-`;
- `edit`: the subject, one `{subject}` prompt template, intended target, and an
  optional original target;
- `probes`: explicit `exact`, `paraphrase`, `locality`, `portability`, and
  `control` arrays;
- `provenance`: dataset name, dataset license, and a source identifier.

All five probe arrays must be present so an empty category is deliberate rather
than ambiguous. At least one exact probe is required. Each `probe_id` must be
unique across every category, which lets later reports retain a stable link to
the original evidence.

Portability probes additionally record an `expected_relationship` with a
relation and target. The allowed relation labels are `entails`, `contradicts`,
`supports`, and `is_related_to`. These labels document the test author's
expectation; they are not themselves metric results or safety conclusions.

`dataset_license` should use an SPDX identifier or expression when one exists.
`source` may be a URL, DOI, URN, or another durable source identifier. Synthetic
cases still need an explicit license and source.

## Minimal example

```json
{
  "schema_version": "1.0.0",
  "case_id": "capital-example",
  "edit": {
    "subject": "Freedonia",
    "prompt_template": "The capital of {subject} is",
    "target_new": " Sylvania",
    "target_original": " Bigtown"
  },
  "probes": {
    "exact": [
      {"probe_id": "exact-1", "prompt": "The capital of Freedonia is"}
    ],
    "paraphrase": [],
    "locality": [],
    "portability": [],
    "control": []
  },
  "provenance": {
    "dataset_name": "Synthetic example",
    "dataset_license": "CC0-1.0",
    "source": "urn:kedit-audit:example:capital"
  }
}
```

## Python validation

```python
import json
from pathlib import Path

from kedit_audit.artifacts import validate_audit_case

case = json.loads(Path("case.json").read_text(encoding="utf-8"))
validate_audit_case(case)
```

Invalid input raises `AuditCaseValidationError`. Its `issues` attribute retains
every known problem as a `ValidationIssue` with a JSONPath-like `path` and an
actionable `message`. Validation combines Draft 2020-12 structure checks with
semantic checks that provide clearer errors and cross-collection constraints:

- the prompt template contains exactly one `{subject}` field and no other
  format fields;
- probe IDs are unique across all five categories;
- when supplied, `target_original` differs from `target_new`.

## Versioning policy

The value in `schema_version` identifies this contract, independently of the
Python package version. A change that invalidates a previously valid case or
changes field meaning requires a new major schema version. Additive optional
fields may use a minor version. Clarifications that do not change validation
may use a patch version.

Consumers must fail closed on unsupported schema versions. Migration between
versions should be explicit; validators must not silently reinterpret a case.
