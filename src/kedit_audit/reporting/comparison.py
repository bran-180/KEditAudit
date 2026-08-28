"""Strictly comparable aggregate deltas between validated AuditReports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from kedit_audit.artifacts import (
    REPORT_COMPARISON_SCHEMA_VERSION,
    AuditReportValidationError,
    canonical_json_bytes,
    hash_json,
    validate_audit_report,
    validate_report_comparison,
)

Presence = Literal["both", "only-a", "only-b"]


class ReportComparisonError(ValueError):
    """Raised when reports are invalid or do not share required run inputs."""


@dataclass(frozen=True)
class MetricSnapshot:
    status: str
    aggregate: float | None
    coverage: float

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "aggregate": self.aggregate,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class MetricComparison:
    metric_id: str
    presence: Presence
    comparable: bool
    unit: str | None
    direction: str | None
    reduction: str | None
    report_a: MetricSnapshot | None
    report_b: MetricSnapshot | None
    aggregate_delta_b_minus_a: float | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "presence": self.presence,
            "comparable": self.comparable,
            "unit": self.unit,
            "direction": self.direction,
            "reduction": self.reduction,
            "report_a": self.report_a.as_dict() if self.report_a is not None else None,
            "report_b": self.report_b.as_dict() if self.report_b is not None else None,
            "aggregate_delta_b_minus_a": self.aggregate_delta_b_minus_a,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ReportComparison:
    report_a_id: str
    report_a_hash: Mapping[str, object]
    report_b_id: str
    report_b_hash: Mapping[str, object]
    audit_case_sha256: str
    baseline_sha256: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    device: str
    dtype: str
    metrics: tuple[MetricComparison, ...]
    warnings: tuple[str, ...]
    schema_version: str = REPORT_COMPARISON_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "report_a": {
                "report_id": self.report_a_id,
                "content_hash": dict(self.report_a_hash),
            },
            "report_b": {
                "report_id": self.report_b_id,
                "content_hash": dict(self.report_b_hash),
            },
            "comparability": {
                "audit_case_sha256": self.audit_case_sha256,
                "baseline_sha256": self.baseline_sha256,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer_revision": self.tokenizer_revision,
                "device": self.device,
                "dtype": self.dtype,
            },
            "metrics": [metric.as_dict() for metric in self.metrics],
            "warnings": list(self.warnings),
        }
        validate_report_comparison(result)
        return result


def compare_audit_reports(
    report_a: Mapping[str, object],
    report_b: Mapping[str, object],
) -> ReportComparison:
    """Compare compatible metric aggregates as descriptive ``B - A`` deltas."""

    normalized_a = _validated_report_copy(report_a, label="report_a")
    normalized_b = _validated_report_copy(report_b, label="report_b")
    context_a = _comparison_context(normalized_a, label="report_a")
    context_b = _comparison_context(normalized_b, label="report_b")
    _require_same_context(context_a, context_b)

    metrics_a = _metrics_by_id(normalized_a)
    metrics_b = _metrics_by_id(normalized_b)
    metric_rows = tuple(
        _compare_metric(metric_id, metrics_a.get(metric_id), metrics_b.get(metric_id))
        for metric_id in sorted(metrics_a.keys() | metrics_b.keys())
    )
    warnings = [
        (
            "aggregate deltas are descriptive report_b - report_a values; they do not "
            "automatically establish improvement, regression, semantic harm, or model safety"
        ),
        (
            "raw probe and structural-value deltas are not reduced here; inspect the two "
            "hash-linked source reports"
        ),
    ]
    commit_a = _nested(normalized_a, "manifest", "kedit_audit", "commit")
    commit_b = _nested(normalized_b, "manifest", "kedit_audit", "commit")
    if commit_a != commit_b:
        warnings.append(
            "KEditAudit commits differ; metric rows are compared only when their declared "
            "unit, direction, and reduction match"
        )

    comparison = ReportComparison(
        report_a_id=cast(str, normalized_a["report_id"]),
        report_a_hash=hash_json(normalized_a).as_dict(),
        report_b_id=cast(str, normalized_b["report_id"]),
        report_b_hash=hash_json(normalized_b).as_dict(),
        audit_case_sha256=context_a.audit_case_sha256,
        baseline_sha256=context_a.baseline_sha256,
        model_id=context_a.model_id,
        model_revision=context_a.model_revision,
        tokenizer_id=context_a.tokenizer_id,
        tokenizer_revision=context_a.tokenizer_revision,
        device=context_a.device,
        dtype=context_a.dtype,
        metrics=metric_rows,
        warnings=tuple(warnings),
    )
    comparison.as_dict()
    return comparison


@dataclass(frozen=True)
class _ComparisonContext:
    audit_case_reference: Mapping[str, object]
    audit_case_sha256: str
    baseline_reference: Mapping[str, object]
    baseline_sha256: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    device: str
    dtype: str
    quantization: Mapping[str, object]
    generation: Mapping[str, object]
    seeds: Mapping[str, object]


def _validated_report_copy(
    report: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(report, Mapping):
        raise ReportComparisonError(f"{label} must be a mapping")
    try:
        normalized = json.loads(canonical_json_bytes(report))
    except (TypeError, ValueError, RecursionError) as error:
        raise ReportComparisonError(f"{label} must contain finite JSON values") from error
    try:
        validate_audit_report(normalized)
    except AuditReportValidationError as error:
        raise ReportComparisonError(
            f"{label} must satisfy the AuditReport contract"
        ) from error
    return cast(dict[str, object], normalized)


def _comparison_context(
    report: Mapping[str, object],
    *,
    label: str,
) -> _ComparisonContext:
    case_reference = cast(
        Mapping[str, object],
        _nested(report, "audit_case", "artifact"),
    )
    baseline_reference = cast(
        Mapping[str, object],
        _nested(report, "manifest", "model", "baseline"),
    )
    case_hash = case_reference.get("content_hash")
    baseline_hash = baseline_reference.get("content_hash")
    if not isinstance(case_hash, Mapping):
        raise ReportComparisonError(
            f"{label}.audit_case.artifact requires a content hash for comparison"
        )
    if not isinstance(baseline_hash, Mapping):
        raise ReportComparisonError(
            f"{label}.manifest.model.baseline requires a content hash for comparison"
        )
    return _ComparisonContext(
        audit_case_reference=case_reference,
        audit_case_sha256=cast(str, case_hash["digest"]),
        baseline_reference=baseline_reference,
        baseline_sha256=cast(str, baseline_hash["digest"]),
        model_id=cast(str, _nested(report, "manifest", "model", "model_id")),
        model_revision=cast(str, _nested(report, "manifest", "model", "model_revision")),
        tokenizer_id=cast(str, _nested(report, "manifest", "model", "tokenizer_id")),
        tokenizer_revision=cast(
            str,
            _nested(report, "manifest", "model", "tokenizer_revision"),
        ),
        device=cast(str, _nested(report, "manifest", "environment", "device")),
        dtype=cast(str, _nested(report, "manifest", "environment", "dtype")),
        quantization=cast(
            Mapping[str, object],
            _nested(report, "manifest", "environment", "quantization"),
        ),
        generation=cast(
            Mapping[str, object],
            _nested(report, "manifest", "generation"),
        ),
        seeds=cast(Mapping[str, object], _nested(report, "manifest", "seeds")),
    )


def _require_same_context(a: _ComparisonContext, b: _ComparisonContext) -> None:
    comparisons = (
        ("$.audit_case.artifact", a.audit_case_reference, b.audit_case_reference),
        ("$.manifest.model.baseline", a.baseline_reference, b.baseline_reference),
        ("$.manifest.model.model_id", a.model_id, b.model_id),
        ("$.manifest.model.model_revision", a.model_revision, b.model_revision),
        ("$.manifest.model.tokenizer_id", a.tokenizer_id, b.tokenizer_id),
        (
            "$.manifest.model.tokenizer_revision",
            a.tokenizer_revision,
            b.tokenizer_revision,
        ),
        ("$.manifest.environment.device", a.device, b.device),
        ("$.manifest.environment.dtype", a.dtype, b.dtype),
        ("$.manifest.environment.quantization", a.quantization, b.quantization),
        ("$.manifest.generation", a.generation, b.generation),
        ("$.manifest.seeds", a.seeds, b.seeds),
    )
    mismatches = [path for path, value_a, value_b in comparisons if value_a != value_b]
    if mismatches:
        raise ReportComparisonError(
            "reports are not comparable; mismatched required context: " + ", ".join(mismatches)
        )


def _metrics_by_id(
    report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    metrics = cast(Sequence[Mapping[str, object]], report["metrics"])
    return {cast(str, metric["metric_id"]): metric for metric in metrics}


def _compare_metric(
    metric_id: str,
    metric_a: Mapping[str, object] | None,
    metric_b: Mapping[str, object] | None,
) -> MetricComparison:
    if metric_a is None:
        return MetricComparison(
            metric_id=metric_id,
            presence="only-b",
            comparable=False,
            unit=None,
            direction=None,
            reduction=None,
            report_a=None,
            report_b=_snapshot(cast(Mapping[str, object], metric_b)),
            aggregate_delta_b_minus_a=None,
            warnings=("metric is absent from report_a",),
        )
    if metric_b is None:
        return MetricComparison(
            metric_id=metric_id,
            presence="only-a",
            comparable=False,
            unit=None,
            direction=None,
            reduction=None,
            report_a=_snapshot(metric_a),
            report_b=None,
            aggregate_delta_b_minus_a=None,
            warnings=("metric is absent from report_b",),
        )

    contract_fields = ("unit", "direction", "reduction")
    mismatches = tuple(field for field in contract_fields if metric_a[field] != metric_b[field])
    if mismatches:
        return MetricComparison(
            metric_id=metric_id,
            presence="both",
            comparable=False,
            unit=None,
            direction=None,
            reduction=None,
            report_a=_snapshot(metric_a),
            report_b=_snapshot(metric_b),
            aggregate_delta_b_minus_a=None,
            warnings=(
                "metric contract differs across reports: " + ", ".join(mismatches),
            ),
        )

    snapshot_a = _snapshot(metric_a)
    snapshot_b = _snapshot(metric_b)
    warnings: list[str] = []
    delta: float | None = None
    if snapshot_a.aggregate is None or snapshot_b.aggregate is None:
        warnings.append("aggregate delta unavailable because at least one aggregate is null")
    else:
        candidate = snapshot_b.aggregate - snapshot_a.aggregate
        if math.isfinite(candidate):
            delta = candidate
        else:
            warnings.append("aggregate delta is outside the finite float range")
    if snapshot_a.status != "complete" or snapshot_b.status != "complete":
        warnings.append("at least one metric result is not complete")
    return MetricComparison(
        metric_id=metric_id,
        presence="both",
        comparable=True,
        unit=cast(str, metric_a["unit"]),
        direction=cast(str, metric_a["direction"]),
        reduction=cast(str, metric_a["reduction"]),
        report_a=snapshot_a,
        report_b=snapshot_b,
        aggregate_delta_b_minus_a=delta,
        warnings=tuple(warnings),
    )


def _snapshot(metric: Mapping[str, object]) -> MetricSnapshot:
    aggregate = metric["aggregate"]
    coverage = cast(Mapping[str, object], metric["coverage"])
    return MetricSnapshot(
        status=cast(str, metric["status"]),
        aggregate=float(aggregate) if isinstance(aggregate, (int, float)) else None,
        coverage=float(cast(float, coverage["fraction"])),
    )


def _nested(root: Mapping[str, object], *keys: str) -> object:
    value: object = root
    for key in keys:
        value = cast(Mapping[str, object], value)[key]
    return value
