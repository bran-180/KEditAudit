"""Load and validate versioned KEditAudit JSON schemas."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from string import Formatter
from typing import Any, cast

from jsonschema import Draft202012Validator

AUDIT_CASE_SCHEMA_VERSION = "1.0.0"
RUN_MANIFEST_SCHEMA_VERSION = "1.0.0"
_PROBE_GROUPS = ("exact", "paraphrase", "locality", "portability", "control")


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


def load_audit_case_schema() -> dict[str, Any]:
    """Return the packaged AuditCase schema as a new dictionary."""

    return _load_schema("audit_case.schema.json")


def load_run_manifest_schema() -> dict[str, Any]:
    """Return the packaged RunManifest schema as a new dictionary."""

    return _load_schema("run_manifest.schema.json")


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


def _load_schema(resource_name: str) -> dict[str, Any]:
    schema_text = files("kedit_audit.artifacts").joinpath(resource_name).read_text(encoding="utf-8")
    return cast(dict[str, Any], json.loads(schema_text))


def _schema_issues(schema: Mapping[str, Any], instance: object) -> list[ValidationIssue]:
    validator = Draft202012Validator(schema)
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


def _run_manifest_semantic_issues(instance: object) -> list[ValidationIssue]:
    if not isinstance(instance, Mapping):
        return []

    issues = _model_state_issues(instance)
    issues.extend(_timestamp_issues(instance))
    issues.extend(_non_finite_number_issues(instance))
    return issues


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
