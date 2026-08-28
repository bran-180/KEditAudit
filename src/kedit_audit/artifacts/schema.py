"""Load and validate versioned KEditAudit JSON schemas."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from string import Formatter
from typing import Any, TypeGuard, cast

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

AUDIT_CASE_SCHEMA_VERSION = "1.0.0"
RUN_MANIFEST_SCHEMA_VERSION = "1.0.0"
METRIC_RESULT_SCHEMA_VERSION = "1.0.0"
AUDIT_REPORT_SCHEMA_VERSION = "1.0.0"
EDITOR_ARTIFACT_SCHEMA_VERSION = "1.0.0"
RIPPLE_CASE_SCHEMA_VERSION = "1.0.0"
REPORT_COMPARISON_SCHEMA_VERSION = "1.0.0"
_PROBE_GROUPS = ("exact", "paraphrase", "locality", "portability", "control")
_RIPPLE_GROUPS = (
    "relation_specificity",
    "logical_generalization",
    "subject_aliasing",
    "compositionality_i",
    "compositionality_ii",
    "forgetfulness",
)


@dataclass(frozen=True)
class ValidationIssue:
    """One validation failure located by a JSONPath-like string."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class AuditCaseValidationError(ValueError):
    """Raised when an AuditCase violates its structural or semantic contract."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("AuditCaseValidationError requires at least one issue")
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"AuditCase validation failed:\n{details}")


class RunManifestValidationError(ValueError):
    """Raised when a RunManifest violates its structural or semantic contract."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("RunManifestValidationError requires at least one issue")
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"RunManifest validation failed:\n{details}")


class MetricResultValidationError(ValueError):
    """Raised when a MetricResult violates its structural or semantic contract."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("MetricResultValidationError requires at least one issue")
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"MetricResult validation failed:\n{details}")


class AuditReportValidationError(ValueError):
    """Raised when an AuditReport violates its structural or semantic contract."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("AuditReportValidationError requires at least one issue")
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"AuditReport validation failed:\n{details}")


class EditorArtifactManifestValidationError(ValueError):
    """Raised when an EditorArtifact manifest violates its data contract."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("EditorArtifactManifestValidationError requires at least one issue")
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"EditorArtifact manifest validation failed:\n{details}")


class RippleCaseValidationError(ValueError):
    """Raised when a RippleCase violates its structural or semantic contract."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("RippleCaseValidationError requires at least one issue")
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"RippleCase validation failed:\n{details}")


class ReportComparisonValidationError(ValueError):
    """Raised when a ReportComparison violates its versioned contract."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("ReportComparisonValidationError requires at least one issue")
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"ReportComparison validation failed:\n{details}")


def load_audit_case_schema() -> dict[str, Any]:
    """Return the packaged AuditCase schema as a new dictionary."""

    return _load_schema("audit_case.schema.json")


def load_run_manifest_schema() -> dict[str, Any]:
    """Return the packaged RunManifest schema as a new dictionary."""

    return _load_schema("run_manifest.schema.json")


def load_metric_result_schema() -> dict[str, Any]:
    """Return the packaged MetricResult schema as a new dictionary."""

    return _load_schema("metric_result.schema.json")


def load_audit_report_schema() -> dict[str, Any]:
    """Return the packaged AuditReport schema as a new dictionary."""

    return _load_schema("audit_report.schema.json")


def load_editor_artifact_schema() -> dict[str, Any]:
    """Return the packaged EditorArtifact schema as a new dictionary."""

    return _load_schema("editor_artifact.schema.json")


def load_ripple_case_schema() -> dict[str, Any]:
    """Return the packaged RippleCase schema as a new dictionary."""

    return _load_schema("ripple_case.schema.json")


def load_report_comparison_schema() -> dict[str, Any]:
    """Return the packaged ReportComparison schema as a new dictionary."""

    return _load_schema("report_comparison.schema.json")


def validate_audit_case(instance: object) -> None:
    """Validate an AuditCase and raise one error containing every known issue."""

    schema_issues = _schema_issues(load_audit_case_schema(), instance)
    issues = schema_issues + _semantic_issues(instance)
    if issues:
        raise AuditCaseValidationError(issues)


def validate_run_manifest(instance: object) -> None:
    """Validate a RunManifest and raise one error containing every known issue."""

    issues = _schema_issues(load_run_manifest_schema(), instance)
    issues.extend(_run_manifest_semantic_issues(instance))
    if issues:
        raise RunManifestValidationError(issues)


def validate_metric_result(instance: object) -> None:
    """Validate a MetricResult, including coverage and raw-evidence consistency."""

    issues = _schema_issues(load_metric_result_schema(), instance)
    issues.extend(_metric_result_semantic_issues(instance))
    if issues:
        raise MetricResultValidationError(issues)


def validate_audit_report(instance: object) -> None:
    """Validate an AuditReport and every nested versioned contract."""

    issues = _schema_issues(
        load_audit_report_schema(),
        instance,
        registry=_artifact_schema_registry(),
    )
    issues.extend(_audit_report_semantic_issues(instance))
    if issues:
        raise AuditReportValidationError(issues)


def validate_editor_artifact_manifest(instance: object) -> None:
    """Validate a bounded data-only external-editor manifest."""

    issues = _schema_issues(load_editor_artifact_schema(), instance)
    issues.extend(_editor_artifact_semantic_issues(instance))
    if issues:
        raise EditorArtifactManifestValidationError(issues)


def validate_ripple_case(instance: object) -> None:
    """Validate one versioned portability/ripple case and its provenance."""

    issues = _schema_issues(load_ripple_case_schema(), instance)
    issues.extend(_ripple_case_semantic_issues(instance))
    if issues:
        raise RippleCaseValidationError(issues)


def validate_report_comparison(instance: object) -> None:
    """Validate a report comparison and its metric-row consistency."""

    issues = _schema_issues(load_report_comparison_schema(), instance)
    issues.extend(_report_comparison_semantic_issues(instance))
    if issues:
        raise ReportComparisonValidationError(issues)


def _load_schema(resource_name: str) -> dict[str, Any]:
    schema_text = files("kedit_audit.artifacts").joinpath(resource_name).read_text(encoding="utf-8")
    return cast(dict[str, Any], json.loads(schema_text))


def _schema_issues(
    schema: Mapping[str, Any],
    instance: object,
    *,
    registry: Registry[Any] | None = None,
) -> list[ValidationIssue]:
    if registry is None:
        validator = Draft202012Validator(schema)
    else:
        validator = Draft202012Validator(schema, registry=registry)
    return [
        ValidationIssue(_format_path(error.absolute_path), error.message)
        for error in sorted(
            validator.iter_errors(instance),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )
    ]


def _artifact_schema_registry() -> Registry[Any]:
    registry: Registry[Any] = Registry()
    for schema in (load_run_manifest_schema(), load_metric_result_schema()):
        identifier = schema.get("$id")
        if not isinstance(identifier, str):
            raise TypeError("packaged artifact schemas must declare a string $id")
        registry = registry.with_resource(identifier, Resource.from_contents(schema))
    return registry


def _format_path(parts: Iterable[object]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _semantic_issues(instance: object) -> list[ValidationIssue]:
    if not isinstance(instance, Mapping):
        return []

    issues = _prompt_template_issues(instance)
    issues.extend(_probe_id_issues(instance))
    issues.extend(_target_issues(instance))
    return issues


def _editor_artifact_semantic_issues(instance: object) -> list[ValidationIssue]:
    if not isinstance(instance, Mapping):
        return []
    issues = _non_finite_number_issues(instance)
    model = instance.get("model")
    if isinstance(model, Mapping):
        baseline = model.get("baseline")
        edited = model.get("edited")
        if isinstance(baseline, Mapping) and isinstance(edited, Mapping):
            if baseline.get("state_id") == edited.get("state_id"):
                issues.append(
                    ValidationIssue(
                        "$.model.edited.state_id",
                        "must differ from $.model.baseline.state_id",
                    )
                )
            if baseline.get("artifact_sha256") == edited.get("artifact_sha256"):
                issues.append(
                    ValidationIssue(
                        "$.model.edited.artifact_sha256",
                        "must differ from $.model.baseline.artifact_sha256",
                    )
                )

    changed_tensors = instance.get("changed_tensors")
    if _is_array(changed_tensors):
        first_path_by_name: dict[str, str] = {}
        for index, tensor in enumerate(changed_tensors):
            if not isinstance(tensor, Mapping):
                continue
            name = tensor.get("name")
            if isinstance(name, str):
                path = f"$.changed_tensors[{index}].name"
                first_path = first_path_by_name.setdefault(name, path)
                if first_path != path:
                    issues.append(
                        ValidationIssue(
                            path,
                            f"duplicate changed tensor name {name!r}; first declared at {first_path}",
                        )
                    )
            if tensor.get("baseline_sha256") == tensor.get("edited_sha256"):
                issues.append(
                    ValidationIssue(
                        f"$.changed_tensors[{index}].edited_sha256",
                        "must differ from baseline_sha256 for a reported changed tensor",
                    )
                )
    return issues


def _ripple_case_semantic_issues(instance: object) -> list[ValidationIssue]:
    if not isinstance(instance, Mapping):
        return []
    issues: list[ValidationIssue] = []
    edit = instance.get("edit")
    if isinstance(edit, Mapping):
        target_new = edit.get("target_new")
        target_original = edit.get("target_original")
        if isinstance(target_new, str) and target_new == target_original:
            issues.append(
                ValidationIssue(
                    "$.edit.target_original",
                    "must differ from $.edit.target_new when supplied",
                )
            )

    probes = instance.get("probes")
    total_probe_count = 0
    first_path_by_id: dict[str, str] = {}
    if isinstance(probes, Mapping):
        for group_name in _RIPPLE_GROUPS:
            group = probes.get(group_name)
            if not _is_array(group):
                continue
            total_probe_count += len(group)
            for index, probe in enumerate(group):
                if not isinstance(probe, Mapping):
                    continue
                probe_id = probe.get("probe_id")
                if not isinstance(probe_id, str):
                    continue
                path = f"$.probes.{group_name}[{index}].probe_id"
                first_path = first_path_by_id.setdefault(probe_id, path)
                if first_path != path:
                    issues.append(
                        ValidationIssue(
                            path,
                            f"duplicate probe_id {probe_id!r}; first declared at {first_path}",
                        )
                    )
        if total_probe_count == 0:
            issues.append(
                ValidationIssue(
                    "$.probes",
                    "must contain at least one probe across the six ripple categories",
                )
            )

    provenance = instance.get("provenance")
    if isinstance(provenance, Mapping):
        upstream_case_id = provenance.get("upstream_case_id")
        artifact_kind = instance.get("artifact_kind")
        if artifact_kind == "external-benchmark-case" and not isinstance(
            upstream_case_id, str
        ):
            issues.append(
                ValidationIssue(
                    "$.provenance.upstream_case_id",
                    "is required for an external-benchmark-case",
                )
            )
        if artifact_kind == "synthetic-contract-fixture" and upstream_case_id is not None:
            issues.append(
                ValidationIssue(
                    "$.provenance.upstream_case_id",
                    "must be omitted for a synthetic-contract-fixture",
                )
            )
    return issues


def _prompt_template_issues(instance: Mapping[object, object]) -> list[ValidationIssue]:
    edit = instance.get("edit")
    if not isinstance(edit, Mapping):
        return []
    template = edit.get("prompt_template")
    if not isinstance(template, str):
        return []

    try:
        fields = [field for _, field, _, _ in Formatter().parse(template) if field is not None]
    except ValueError as error:
        return [ValidationIssue("$.edit.prompt_template", f"invalid format template: {error}")]

    if fields != ["subject"]:
        return [
            ValidationIssue(
                "$.edit.prompt_template",
                "must contain exactly one {subject} field and no other format fields",
            )
        ]
    return []


def _probe_id_issues(instance: Mapping[object, object]) -> list[ValidationIssue]:
    probes = instance.get("probes")
    if not isinstance(probes, Mapping):
        return []

    issues: list[ValidationIssue] = []
    first_path_by_id: dict[str, str] = {}
    for group_name in _PROBE_GROUPS:
        group = probes.get(group_name)
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            continue
        for index, probe in enumerate(group):
            if not isinstance(probe, Mapping):
                continue
            probe_id = probe.get("probe_id")
            if not isinstance(probe_id, str):
                continue
            path = f"$.probes.{group_name}[{index}].probe_id"
            first_path = first_path_by_id.setdefault(probe_id, path)
            if first_path != path:
                issues.append(
                    ValidationIssue(
                        path,
                        f"duplicate probe_id {probe_id!r}; first declared at {first_path}",
                    )
                )
    return issues


def _target_issues(instance: Mapping[object, object]) -> list[ValidationIssue]:
    edit = instance.get("edit")
    if not isinstance(edit, Mapping):
        return []
    target_new = edit.get("target_new")
    target_original = edit.get("target_original")
    if isinstance(target_new, str) and target_new == target_original:
        return [
            ValidationIssue(
                "$.edit.target_original",
                "must differ from $.edit.target_new when supplied",
            )
        ]
    return []


def _run_manifest_semantic_issues(
    instance: object,
    *,
    include_non_finite: bool = True,
) -> list[ValidationIssue]:
    if not isinstance(instance, Mapping):
        return []

    issues = _model_state_issues(instance)
    issues.extend(_timestamp_issues(instance))
    if include_non_finite:
        issues.extend(_non_finite_number_issues(instance))
    return issues


def _metric_result_semantic_issues(
    instance: object,
    *,
    include_non_finite: bool = True,
) -> list[ValidationIssue]:
    if not isinstance(instance, Mapping):
        return []

    issues = _non_finite_number_issues(instance) if include_non_finite else []
    probes = instance.get("probes")
    coverage = instance.get("coverage")
    if not _is_array(probes) or not isinstance(coverage, Mapping):
        return issues

    first_path_by_id: dict[str, str] = {}
    evaluated_count = 0
    for index, probe in enumerate(probes):
        if not isinstance(probe, Mapping):
            continue
        probe_id = probe.get("probe_id")
        if isinstance(probe_id, str):
            path = f"$.probes[{index}].probe_id"
            first_path = first_path_by_id.setdefault(probe_id, path)
            if first_path != path:
                issues.append(
                    ValidationIssue(
                        path,
                        f"duplicate probe_id {probe_id!r}; first declared at {first_path}",
                    )
                )
        if probe.get("status") == "evaluated":
            evaluated_count += 1

    total_count = len(probes)
    missing_count = total_count - evaluated_count
    issues.extend(
        _coverage_consistency_issues(
            coverage,
            total_count=total_count,
            evaluated_count=evaluated_count,
            missing_count=missing_count,
        )
    )

    status = instance.get("status")
    if status == "complete" and missing_count != 0:
        issues.append(
            ValidationIssue(
                "$.status",
                "complete requires every probe to have status 'evaluated'",
            )
        )
    if status == "incomplete" and missing_count == 0:
        issues.append(
            ValidationIssue(
                "$.status",
                "incomplete requires at least one missing or failed probe",
            )
        )

    aggregate = instance.get("aggregate")
    if evaluated_count == 0 and aggregate is not None:
        issues.append(
            ValidationIssue(
                "$.aggregate",
                "must be null when no probes were evaluated",
            )
        )
    if evaluated_count > 0 and status in {"complete", "incomplete"} and aggregate is None:
        issues.append(
            ValidationIssue(
                "$.aggregate",
                "must be present when at least one probe was evaluated",
            )
        )

    issues.extend(_threshold_issues(instance))
    return issues


def _coverage_consistency_issues(
    coverage: Mapping[object, object],
    *,
    total_count: int,
    evaluated_count: int,
    missing_count: int,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    expected_counts = {
        "total": total_count,
        "evaluated": evaluated_count,
        "missing": missing_count,
    }
    for field, expected in expected_counts.items():
        actual = coverage.get(field)
        if _is_integer(actual) and actual != expected:
            description = "raw probes" if field == "total" else f"{field} probes"
            issues.append(
                ValidationIssue(
                    f"$.coverage.{field}",
                    f"must equal {expected}, the number of {description}",
                )
            )

    fraction = coverage.get("fraction")
    expected_fraction = evaluated_count / total_count if total_count else 0.0
    if (
        _is_number(fraction)
        and math.isfinite(float(fraction))
        and not math.isclose(float(fraction), expected_fraction, rel_tol=0.0, abs_tol=1e-12)
    ):
        issues.append(
            ValidationIssue(
                "$.coverage.fraction",
                "must equal evaluated / total (or 0 when total is 0)",
            )
        )
    return issues


def _threshold_issues(instance: Mapping[object, object]) -> list[ValidationIssue]:
    threshold = instance.get("threshold")
    aggregate = instance.get("aggregate")
    if not isinstance(threshold, Mapping) or not _is_number(aggregate):
        return []

    operator = threshold.get("operator")
    value = threshold.get("value")
    passed = threshold.get("passed")
    if not isinstance(operator, str) or not _is_number(value) or not isinstance(passed, bool):
        return []
    if not math.isfinite(float(aggregate)) or not math.isfinite(float(value)):
        return []

    comparisons = {
        "greater-than": float(aggregate) > float(value),
        "greater-than-or-equal": float(aggregate) >= float(value),
        "less-than": float(aggregate) < float(value),
        "less-than-or-equal": float(aggregate) <= float(value),
    }
    expected = comparisons.get(operator)
    if expected is not None and passed != expected:
        return [
            ValidationIssue(
                "$.threshold.passed",
                "must equal the declared comparison of aggregate and threshold value",
            )
        ]
    return []


def _audit_report_semantic_issues(instance: object) -> list[ValidationIssue]:
    if not isinstance(instance, Mapping):
        return []

    issues = _non_finite_number_issues(instance)
    manifest = instance.get("manifest")
    if isinstance(manifest, Mapping):
        issues.extend(
            _prefix_issues(
                "$.manifest",
                _run_manifest_semantic_issues(manifest, include_non_finite=False),
            )
        )

    metrics = instance.get("metrics")
    indexed_metrics: list[tuple[int, Mapping[object, object]]] = []
    if _is_array(metrics):
        first_path_by_id: dict[str, str] = {}
        for index, metric in enumerate(metrics):
            if not isinstance(metric, Mapping):
                continue
            indexed_metrics.append((index, metric))
            issues.extend(
                _prefix_issues(
                    f"$.metrics[{index}]",
                    _metric_result_semantic_issues(metric, include_non_finite=False),
                )
            )
            metric_id = metric.get("metric_id")
            if isinstance(metric_id, str):
                path = f"$.metrics[{index}].metric_id"
                first_path = first_path_by_id.setdefault(metric_id, path)
                if first_path != path:
                    issues.append(
                        ValidationIssue(
                            path,
                            f"duplicate metric_id {metric_id!r}; first declared at {first_path}",
                        )
                    )

    issues.extend(_report_status_issues(instance, manifest, indexed_metrics))
    issues.extend(_report_generated_at_issues(instance, manifest))
    issues.extend(_structural_evidence_id_issues(instance))
    issues.extend(_report_case_reference_issues(instance, manifest))
    return issues


def _report_status_issues(
    report: Mapping[object, object],
    manifest: object,
    metrics: Sequence[tuple[int, Mapping[object, object]]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    report_status = report.get("status")
    manifest_status = manifest.get("status") if isinstance(manifest, Mapping) else None
    if (
        isinstance(report_status, str)
        and isinstance(manifest_status, str)
        and report_status != manifest_status
    ):
        issues.append(
            ValidationIssue(
                "$.status",
                f"must equal $.manifest.status ({manifest_status!r})",
            )
        )

    if report_status != "completed":
        return issues

    incomplete_evidence = False
    for index, metric in metrics:
        if metric.get("status") != "complete":
            incomplete_evidence = True
            issues.append(
                ValidationIssue(
                    f"$.metrics[{index}].status",
                    "must be 'complete' when the report status is 'completed'",
                )
            )

    structural = report.get("structural_evidence")
    if _is_array(structural):
        for index, evidence in enumerate(structural):
            if isinstance(evidence, Mapping) and evidence.get("status") != "complete":
                incomplete_evidence = True
                issues.append(
                    ValidationIssue(
                        f"$.structural_evidence[{index}].status",
                        "must be 'complete' when the report status is 'completed'",
                    )
                )
    if incomplete_evidence:
        issues.append(
            ValidationIssue(
                "$.status",
                "cannot be 'completed' while included evidence is incomplete or failed",
            )
        )
    return issues


def _report_generated_at_issues(
    report: Mapping[object, object],
    manifest: object,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    generated_at = _parse_utc_timestamp(report.get("generated_at"), "$.generated_at", issues)
    if not isinstance(manifest, Mapping):
        return issues
    timestamps = manifest.get("timestamps")
    if not isinstance(timestamps, Mapping):
        return issues

    ignored_manifest_issues: list[ValidationIssue] = []
    ended_at = _parse_utc_timestamp(
        timestamps.get("ended_at"),
        "$.manifest.timestamps.ended_at",
        ignored_manifest_issues,
    )
    if generated_at is not None and ended_at is not None and generated_at < ended_at:
        issues.append(
            ValidationIssue(
                "$.generated_at",
                "must not precede $.manifest.timestamps.ended_at",
            )
        )
    return issues


def _structural_evidence_id_issues(
    report: Mapping[object, object],
) -> list[ValidationIssue]:
    structural = report.get("structural_evidence")
    if not _is_array(structural):
        return []

    issues: list[ValidationIssue] = []
    first_path_by_id: dict[str, str] = {}
    for index, evidence in enumerate(structural):
        if not isinstance(evidence, Mapping):
            continue
        evidence_id = evidence.get("evidence_id")
        if not isinstance(evidence_id, str):
            continue
        path = f"$.structural_evidence[{index}].evidence_id"
        first_path = first_path_by_id.setdefault(evidence_id, path)
        if first_path != path:
            issues.append(
                ValidationIssue(
                    path,
                    f"duplicate evidence_id {evidence_id!r}; first declared at {first_path}",
                )
            )
    return issues


def _report_case_reference_issues(
    report: Mapping[object, object],
    manifest: object,
) -> list[ValidationIssue]:
    report_case = report.get("audit_case")
    if not isinstance(report_case, Mapping) or not isinstance(manifest, Mapping):
        return []
    manifest_case = manifest.get("audit_case")
    if not isinstance(manifest_case, Mapping):
        return []

    issues: list[ValidationIssue] = []
    if report_case.get("schema_version") != manifest_case.get("schema_version"):
        issues.append(
            ValidationIssue(
                "$.audit_case.schema_version",
                "must equal $.manifest.audit_case.schema_version",
            )
        )

    report_artifact = report_case.get("artifact")
    manifest_artifact = manifest_case.get("artifact")
    if not isinstance(report_artifact, Mapping) or not isinstance(manifest_artifact, Mapping):
        return issues
    if report_case.get("case_id") != report_artifact.get("artifact_id"):
        issues.append(
            ValidationIssue(
                "$.audit_case.case_id",
                "must equal $.audit_case.artifact.artifact_id",
            )
        )
    if report_artifact.get("artifact_id") != manifest_artifact.get("artifact_id"):
        issues.append(
            ValidationIssue(
                "$.audit_case.artifact.artifact_id",
                "must equal the audit-case artifact ID in the manifest",
            )
        )

    report_hash = _comparable_hash(report_artifact.get("content_hash"))
    manifest_hash = _comparable_hash(manifest_artifact.get("content_hash"))
    if report_hash != manifest_hash:
        issues.append(
            ValidationIssue(
                "$.audit_case.artifact.content_hash.digest",
                "must equal the comparable audit-case content hash in the manifest",
            )
        )
    if (
        report_hash is None
        and manifest_hash is None
        and report_artifact.get("hash_unavailable")
        != manifest_artifact.get("hash_unavailable")
    ):
        issues.append(
            ValidationIssue(
                "$.audit_case.artifact.hash_unavailable",
                "must equal the audit-case hash-unavailable provenance in the manifest",
            )
        )
    return issues


def _prefix_issues(prefix: str, issues: Sequence[ValidationIssue]) -> list[ValidationIssue]:
    return [
        ValidationIssue(prefix + issue.path.removeprefix("$"), issue.message)
        for issue in issues
    ]


def _is_array(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _report_comparison_semantic_issues(instance: object) -> list[ValidationIssue]:
    if not isinstance(instance, Mapping):
        return []
    issues = _non_finite_number_issues(instance)
    metrics = instance.get("metrics")
    if not _is_array(metrics):
        return issues

    first_index_by_id: dict[str, int] = {}
    for index, metric in enumerate(metrics):
        if not isinstance(metric, Mapping):
            continue
        metric_id = metric.get("metric_id")
        if isinstance(metric_id, str):
            first_index = first_index_by_id.setdefault(metric_id, index)
            if first_index != index:
                issues.append(
                    ValidationIssue(
                        f"$.metrics[{index}].metric_id",
                        f"duplicates metrics[{first_index}].metric_id",
                    )
                )

        presence = metric.get("presence")
        snapshot_a = metric.get("report_a")
        snapshot_b = metric.get("report_b")
        expected_presence = (
            {
                "both": (True, True),
                "only-a": (True, False),
                "only-b": (False, True),
            }.get(presence)
            if isinstance(presence, str)
            else None
        )
        if expected_presence is not None:
            actual_presence = (
                isinstance(snapshot_a, Mapping),
                isinstance(snapshot_b, Mapping),
            )
            if actual_presence != expected_presence:
                issues.append(
                    ValidationIssue(
                        f"$.metrics[{index}].presence",
                        "must agree with report_a and report_b snapshot availability",
                    )
                )

        comparable = metric.get("comparable")
        delta = metric.get("aggregate_delta_b_minus_a")
        if comparable is False and delta is not None:
            issues.append(
                ValidationIssue(
                    f"$.metrics[{index}].aggregate_delta_b_minus_a",
                    "must be null when comparable is false",
                )
            )
        if comparable is True:
            if presence != "both":
                issues.append(
                    ValidationIssue(
                        f"$.metrics[{index}].comparable",
                        "can be true only when presence is 'both'",
                    )
                )
            aggregate_a = (
                snapshot_a.get("aggregate") if isinstance(snapshot_a, Mapping) else None
            )
            aggregate_b = (
                snapshot_b.get("aggregate") if isinstance(snapshot_b, Mapping) else None
            )
            if _is_number(aggregate_a) and _is_number(aggregate_b):
                expected_delta = float(aggregate_b) - float(aggregate_a)
                if not math.isfinite(expected_delta):
                    if delta is not None:
                        issues.append(
                            ValidationIssue(
                                f"$.metrics[{index}].aggregate_delta_b_minus_a",
                                "must be null when the subtraction is outside the finite range",
                            )
                        )
                elif not _is_number(delta) or not math.isclose(
                    float(delta),
                    expected_delta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    issues.append(
                        ValidationIssue(
                            f"$.metrics[{index}].aggregate_delta_b_minus_a",
                            "must equal report_b.aggregate - report_a.aggregate",
                        )
                    )
            elif delta is not None:
                issues.append(
                    ValidationIssue(
                        f"$.metrics[{index}].aggregate_delta_b_minus_a",
                        "must be null when either aggregate is unavailable",
                    )
                )
    return issues


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _model_state_issues(instance: Mapping[object, object]) -> list[ValidationIssue]:
    model = instance.get("model")
    if not isinstance(model, Mapping):
        return []
    baseline = model.get("baseline")
    edited = model.get("edited")
    if not isinstance(baseline, Mapping) or not isinstance(edited, Mapping):
        return []

    issues: list[ValidationIssue] = []
    baseline_id = baseline.get("artifact_id")
    edited_id = edited.get("artifact_id")
    if isinstance(baseline_id, str) and baseline_id == edited_id:
        issues.append(
            ValidationIssue(
                "$.model.edited.artifact_id",
                "must differ from $.model.baseline.artifact_id",
            )
        )

    baseline_hash = _comparable_hash(baseline.get("content_hash"))
    edited_hash = _comparable_hash(edited.get("content_hash"))
    if baseline_hash is not None and baseline_hash == edited_hash:
        issues.append(
            ValidationIssue(
                "$.model.edited.content_hash.digest",
                "must differ from the baseline content hash",
            )
        )
    return issues


def _comparable_hash(value: object) -> tuple[object, object, object] | None:
    if not isinstance(value, Mapping):
        return None
    algorithm = value.get("algorithm")
    encoding = value.get("encoding")
    digest = value.get("digest")
    if not all(isinstance(part, str) for part in (algorithm, encoding, digest)):
        return None
    return algorithm, encoding, digest


def _timestamp_issues(instance: Mapping[object, object]) -> list[ValidationIssue]:
    timestamps = instance.get("timestamps")
    if not isinstance(timestamps, Mapping):
        return []

    issues: list[ValidationIssue] = []
    started = _parse_utc_timestamp(timestamps.get("started_at"), "$.timestamps.started_at", issues)
    ended = _parse_utc_timestamp(timestamps.get("ended_at"), "$.timestamps.ended_at", issues)
    if started is not None and ended is not None and ended < started:
        issues.append(
            ValidationIssue(
                "$.timestamps.ended_at",
                "must not precede $.timestamps.started_at",
            )
        )
    return issues


def _parse_utc_timestamp(
    value: object,
    path: str,
    issues: list[ValidationIssue],
) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        issues.append(ValidationIssue(path, "must be a valid UTC date-time"))
        return None


def _non_finite_number_issues(value: object, path: str = "$") -> list[ValidationIssue]:
    if isinstance(value, float) and not math.isfinite(value):
        return [ValidationIssue(path, "must be a finite JSON number")]
    if isinstance(value, Mapping):
        issues: list[ValidationIssue] = []
        for key, item in value.items():
            if isinstance(key, str):
                issues.extend(_non_finite_number_issues(item, f"{path}.{key}"))
        return issues
    if isinstance(value, list):
        issues = []
        for index, item in enumerate(value):
            issues.extend(_non_finite_number_issues(item, f"{path}[{index}]"))
        return issues
    return []
