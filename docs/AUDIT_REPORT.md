# MetricResult and AuditReport contracts

KEditAudit publishes Draft 2020-12 JSON Schemas for `MetricResult` and
`AuditReport` version `1.0.0`:

- [`metric_result.schema.json`](../src/kedit_audit/artifacts/metric_result.schema.json)
- [`audit_report.schema.json`](../src/kedit_audit/artifacts/audit_report.schema.json)

The JSON report is the authoritative evidence artifact. A future Markdown
renderer must derive its content from a validated JSON report and must not
invent values or conclusions.

## MetricResult 1.0.0

Every metric result records:

- `metric_id`, `status`, direction, unit, aggregate, and reduction method;
- explicit `coverage` counts and fraction;
- every supplied probe with its stable `probe_id`, evaluation status, raw
  `values`, and a missing-data reason when applicable;
- warnings and at least one method or benchmark citation;
- an optional numeric threshold with its declared comparison result.

Statuses have narrow meanings:

- `complete`: every probe was evaluated and the aggregate is present;
- `incomplete`: at least one probe is missing or failed, while available
  evidence may still support a partial aggregate;
- `failed`: the metric did not produce an aggregate.

The validator checks that probe IDs are unique, coverage counts agree with the
raw probe array, the coverage fraction equals `evaluated / total`, aggregate
availability agrees with evaluated coverage, optional threshold comparisons
are internally consistent, and every numeric value is finite. Missing evidence
is never interpreted as zero.

## AuditReport 1.0.0

An audit report records:

- report identity, completion status, and generation timestamp;
- the full validated `RunManifest`;
- audit-case identity, content hash, and dataset/license metadata;
- versioned metric results and optional structural evidence;
- limitations and warnings.

The report intentionally carries audit-case metadata rather than copying case
prompts or target text. This reduces accidental duplication of potentially
sensitive inputs while the manifest hash still identifies the exact source
artifact used for the run.

Report status must match the manifest status. A `completed` report cannot
contain incomplete or failed metric or structural evidence. The audit-case
schema version, artifact ID, and comparable content hash must match the
reference frozen in the manifest. Metric IDs must be unique within a report.

These checks establish internal consistency and provenance only. They do not
certify model safety, prove the absence of harmful behavior, or determine
whether a chosen threshold is scientifically or operationally appropriate.

## Offline validation API

```python
from kedit_audit.artifacts import (
    validate_audit_report,
    validate_metric_result,
)

validate_metric_result(metric_result)
validate_audit_report(audit_report)
```

Validation raises `MetricResultValidationError` or
`AuditReportValidationError`. Each exception exposes an `issues` tuple with a
JSONPath-like `path` and an actionable `message`.

The deterministic completed fixture is
[`tests/fixtures/audit_reports/valid/completed.json`](../tests/fixtures/audit_reports/valid/completed.json).
Its KEditAudit canonical JSON v1 representation is 4,354 bytes with SHA-256
`67bb4e1adbc3e0e321e44fa25536c760549519ca6ad16efc68a54c52a778bf6b`.
