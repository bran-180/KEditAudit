# RunManifest and artifact hashing

`RunManifest` is the versioned provenance record for one KEditAudit execution.
It freezes the identities, revisions, configuration, seeds, timestamps, and
failure state needed to interpret later audit artifacts. The authoritative
machine-readable contract is
[`run_manifest.schema.json`](../src/kedit_audit/artifacts/run_manifest.schema.json).

## Version 1.0.0

Every manifest records:

- a stable `run_id`, schema version, lifecycle status, and UTC timestamps;
- the KEditAudit package version, exact Git commit, and dirty-worktree flag;
- Python and package versions, device, dtype, and quantization method/configuration;
- model and tokenizer IDs with immutable revisions;
- logically distinct baseline and edited artifact references;
- the AuditCase schema version and artifact reference;
- editor adapter revision and JSON-native hyperparameters;
- generation configuration and every random seed source used;
- either `null` or structured failure information.

The four lifecycle states are:

- `running`: `ended_at` and `failure` are `null`;
- `completed`: `ended_at` is present and `failure` is `null`;
- `failed`: the run terminated with an `ended_at` value and failure details;
- `incomplete`: the run ended but mandatory evidence is missing, with the
  reason recorded in `failure`.

The validator rejects an end time earlier than the start time. It also rejects
baseline and edited states with the same `artifact_id` or the same comparable
content hash. These checks protect the logical comparison boundary; they do not
prove that an adapter loaded or isolated model state correctly.

A complete valid example is available in
[`completed.json`](../tests/fixtures/run_manifests/valid/completed.json).

## Content-hash record

Every recorded hash contains:

```json
{
  "algorithm": "sha256",
  "encoding": "raw-bytes",
  "digest": "64 lowercase hexadecimal characters",
  "size_bytes": 123
}
```

SHA-256 follows the [NIST Secure Hash Standard, FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final).
The digest detects content differences; it is not a signature, proof of origin,
license grant, safety result, or authorization to distribute an artifact.

If an artifact cannot be hashed, `content_hash` is replaced with
`hash_unavailable`. Both a controlled reason (`not-permitted`,
`not-accessible`, or `not-applicable`) and explanatory details are mandatory.
The validator never treats an omitted hash as equivalent to a verified hash.

## Hash encodings

### `raw-bytes`

Use `raw-bytes` for checkpoints, deltas, datasets, and other files. SHA-256 is
computed over the exact byte stream. Line endings, archive metadata, file
format, and serialization order therefore affect the digest. `hash_file`
streams the file in fixed-size chunks while producing the same result as
`hash_bytes` over the complete contents.

### `kedit-audit-canonical-json-v1`

Use this encoding for parsed JSON artifacts such as AuditCase. It is a
KEditAudit profile, not a claim of RFC 8785 conformance. The algorithm is:

1. accept only JSON-native values: null, booleans, integers, finite numbers,
   strings, lists, and objects with string keys;
2. sort object keys by their Unicode code-point order;
3. preserve array order and JSON numeric values;
4. omit insignificant whitespace;
5. encode strings without ASCII escaping and encode the result as UTF-8;
6. hash those bytes with SHA-256.

Unicode strings are not normalized. Visually similar strings with different
Unicode code-point sequences therefore produce different digests. Non-finite
numbers, non-string object keys, tuples, custom objects, and other non-JSON
values fail closed.

## Python API

```python
import json
from pathlib import Path

from kedit_audit.artifacts import hash_json, validate_run_manifest

case = json.loads(Path("case.json").read_text(encoding="utf-8"))
case_hash = hash_json(case)

manifest = json.loads(Path("run-manifest.json").read_text(encoding="utf-8"))
validate_run_manifest(manifest)
print(case_hash.as_dict())
```

Invalid manifests raise `RunManifestValidationError`. Its `issues` attribute
contains all detected `ValidationIssue` entries with JSONPath-like locations.

## Versioning rule

The manifest schema version is independent of the Python package and AuditCase
schema versions. Breaking validation or meaning changes require a new major
version. Consumers must reject unsupported versions rather than silently
guessing how to interpret provenance.
