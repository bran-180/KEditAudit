"""Data-only end-to-end audit assembly for the dependency-light CLI."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from kedit_audit.artifacts import (
    AUDIT_REPORT_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    AuditCaseValidationError,
    AuditSnapshotValidationError,
    canonical_json_bytes,
    hash_json,
    validate_audit_case,
    validate_audit_snapshot,
)
from kedit_audit.audit.runner import AuditExecutionResult, execute_audit
from kedit_audit.metrics import (
    ControlDivergenceReduction,
    PairedControlLogits,
    PairedProbeScore,
    ProbeScoreReduction,
    reduce_control_kl_divergence,
    reduce_efficacy_log_probability_deltas,
    reduce_generality_log_probability_deltas,
    reduce_locality_log_probability_drift,
    reduce_portability_log_probability_deltas,
)
from kedit_audit.reporting import AuditReportWriteResult, write_audit_report

_BEHAVIORAL_CITATIONS: tuple[dict[str, str], ...] = (
    {
        "title": "EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models",
        "source": "https://aclanthology.org/2024.acl-demos.9/",
    },
    {
        "title": "Locating and Editing Factual Associations in GPT",
        "source": "https://rome.baulab.info/",
    },
)
_PORTABILITY_CITATIONS: tuple[dict[str, str], ...] = (
    _BEHAVIORAL_CITATIONS[0],
    {
        "title": "Ripple Effects in Knowledge Editing",
        "source": "https://aclanthology.org/2024.tacl-1.16/",
    },
)
_CONTROL_CITATIONS: tuple[dict[str, str], ...] = (
    {
        "title": "On Information and Sufficiency",
        "source": "https://doi.org/10.1214/aoms/1177729694",
        "identifier": "10.1214/aoms/1177729694",
    },
)
_TARGET_GROUPS = ("exact", "paraphrase", "locality", "portability")


class AuditPipelineInputError(ValueError):
    """Raised before a run starts when input provenance cannot be frozen safely."""


@dataclass(frozen=True)
class AuditPipelineResult:
    """Paths and public identity for one successfully assembled audit."""

    run_id: str
    manifest_path: Path
    report_json_path: Path
    report_markdown_path: Path


def run_audit_pipeline(
    *,
    audit_case: object,
    baseline_snapshot: object,
    edited_snapshot: object,
    output_directory: str | Path,
    clock: Callable[[], datetime] | None = None,
) -> AuditPipelineResult:
    """Validate data-only snapshots, reduce evidence, and publish one audit report."""

    case = _validated_copy(audit_case, kind="case")
    baseline = _validated_copy(baseline_snapshot, kind="snapshot")
    edited = _validated_copy(edited_snapshot, kind="snapshot")
    _validate_pair_and_coverage(case, baseline, edited)

    read_clock = clock if clock is not None else _utc_now
    started_at = _read_utc_clock(read_clock)
    initial_manifest = _build_running_manifest(
        case=case,
        baseline=baseline,
        edited=edited,
        started_at=started_at,
    )
    published: list[AuditReportWriteResult] = []

    def operation() -> list[dict[str, object]]:
        return _build_metric_results(case, baseline, edited)

    def finalize(
        metrics: list[dict[str, object]],
        completed_manifest: Mapping[str, object],
    ) -> None:
        report = _build_report(
            case=case,
            manifest=completed_manifest,
            metrics=metrics,
        )
        published.append(
            write_audit_report(report, output_directory=output_directory)
        )

    execution: AuditExecutionResult[list[dict[str, object]]] = execute_audit(
        initial_manifest=initial_manifest,
        output_directory=output_directory,
        operation=operation,
        finalize=finalize,
        failure_stage="audit-pipeline",
        clock=read_clock,
    )
    if len(published) != 1:
        raise RuntimeError("audit report finalizer did not publish exactly one report")
    report_paths = published[0]
    return AuditPipelineResult(
        run_id=cast(str, execution.manifest["run_id"]),
        manifest_path=execution.manifest_path,
        report_json_path=report_paths.json_path,
        report_markdown_path=report_paths.markdown_path,
    )


def _validated_copy(value: object, *, kind: str) -> dict[str, object]:
    try:
        encoded = canonical_json_bytes(value)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as error:
        raise AuditPipelineInputError(f"{kind} input must contain finite JSON values") from error
    if not isinstance(normalized, dict):
        raise AuditPipelineInputError(f"{kind} input must be a JSON object")
    result = cast(dict[str, object], normalized)
    try:
        if kind == "case":
            validate_audit_case(result)
        else:
            validate_audit_snapshot(result)
    except (AuditCaseValidationError, AuditSnapshotValidationError) as error:
        raise AuditPipelineInputError(f"{kind} input does not satisfy its contract") from error
    return result


def _validate_pair_and_coverage(
    case: Mapping[str, object],
    baseline: Mapping[str, object],
    edited: Mapping[str, object],
) -> None:
    if baseline["state"] != "baseline" or edited["state"] != "edited":
        raise AuditPipelineInputError(
            "baseline and edited snapshots must declare their corresponding states"
        )
    if baseline["snapshot_id"] == edited["snapshot_id"]:
        raise AuditPipelineInputError("baseline and edited snapshot IDs must differ")

    for field_name in (
        "kedit_audit",
        "environment",
        "editor",
        "generation",
        "seeds",
    ):
        if baseline[field_name] != edited[field_name]:
            raise AuditPipelineInputError(
                f"baseline and edited snapshots have incompatible {field_name} metadata"
            )

    baseline_model = cast(Mapping[str, object], baseline["model"])
    edited_model = cast(Mapping[str, object], edited["model"])
    for field_name in (
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
    ):
        if baseline_model[field_name] != edited_model[field_name]:
            raise AuditPipelineInputError(
                f"baseline and edited snapshots have incompatible model field {field_name}"
            )

    baseline_artifact = cast(Mapping[str, object], baseline_model["artifact"])
    edited_artifact = cast(Mapping[str, object], edited_model["artifact"])
    if baseline_artifact["kind"] != "model-checkpoint":
        raise AuditPipelineInputError("baseline artifact kind must be model-checkpoint")
    if edited_artifact["kind"] not in {"model-checkpoint", "model-delta"}:
        raise AuditPipelineInputError(
            "edited artifact kind must be model-checkpoint or model-delta"
        )
    if baseline_artifact["artifact_id"] == edited_artifact["artifact_id"]:
        raise AuditPipelineInputError("baseline and edited artifact IDs must differ")
    baseline_hash = _content_hash_identity(baseline_artifact)
    edited_hash = _content_hash_identity(edited_artifact)
    if baseline_hash is not None and baseline_hash == edited_hash:
        raise AuditPipelineInputError("baseline and edited artifact hashes must differ")

    expected_target_ids, expected_control_ids = _case_probe_ids(case)
    baseline_target_ids, baseline_control_ids = _snapshot_probe_ids(baseline)
    edited_target_ids, edited_control_ids = _snapshot_probe_ids(edited)
    if baseline_target_ids != expected_target_ids or edited_target_ids != expected_target_ids:
        raise AuditPipelineInputError(
            "target-score probe IDs must exactly cover the AuditCase target probe groups"
        )
    if baseline_control_ids != expected_control_ids or edited_control_ids != expected_control_ids:
        raise AuditPipelineInputError(
            "control-logit probe IDs must exactly cover the AuditCase control group"
        )

    baseline_measurements = cast(Mapping[str, object], baseline["measurements"])
    edited_measurements = cast(Mapping[str, object], edited["measurements"])
    if (
        baseline_measurements["control_temperature"]
        != edited_measurements["control_temperature"]
    ):
        raise AuditPipelineInputError(
            "baseline and edited control temperatures must match"
        )


def _build_running_manifest(
    *,
    case: Mapping[str, object],
    baseline: Mapping[str, object],
    edited: Mapping[str, object],
    started_at: str,
) -> dict[str, object]:
    case_hash = hash_json(case).as_dict()
    baseline_hash = hash_json(baseline).digest
    edited_hash = hash_json(edited).digest
    run_digest = hash_json(
        {
            "audit_case": case_hash,
            "baseline_snapshot": baseline_hash,
            "edited_snapshot": edited_hash,
        }
    ).digest
    run_id = f"run-{run_digest[:24]}"
    case_id = cast(str, case["case_id"])
    case_reference = {
        "schema_version": case["schema_version"],
        "artifact": {
            "artifact_id": case_id,
            "kind": "audit-case",
            "content_hash": case_hash,
        },
    }
    baseline_model = cast(Mapping[str, object], baseline["model"])
    edited_model = cast(Mapping[str, object], edited["model"])
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "timestamps": {"started_at": started_at, "ended_at": None},
        "kedit_audit": baseline["kedit_audit"],
        "environment": baseline["environment"],
        "model": {
            "model_id": baseline_model["model_id"],
            "model_revision": baseline_model["model_revision"],
            "tokenizer_id": baseline_model["tokenizer_id"],
            "tokenizer_revision": baseline_model["tokenizer_revision"],
            "baseline": baseline_model["artifact"],
            "edited": edited_model["artifact"],
        },
        "audit_case": case_reference,
        "editor": baseline["editor"],
        "generation": baseline["generation"],
        "seeds": baseline["seeds"],
        "failure": None,
    }


def _build_metric_results(
    case: Mapping[str, object],
    baseline: Mapping[str, object],
    edited: Mapping[str, object],
) -> list[dict[str, object]]:
    baseline_scores = _target_score_index(baseline)
    edited_scores = _target_score_index(edited)
    probes = cast(Mapping[str, Sequence[Mapping[str, object]]], case["probes"])
    results: list[dict[str, object]] = []
    reductions = (
        (
            "exact",
            reduce_efficacy_log_probability_deltas,
            _BEHAVIORAL_CITATIONS,
        ),
        (
            "paraphrase",
            reduce_generality_log_probability_deltas,
            _BEHAVIORAL_CITATIONS,
        ),
        (
            "locality",
            reduce_locality_log_probability_drift,
            _BEHAVIORAL_CITATIONS,
        ),
        (
            "portability",
            reduce_portability_log_probability_deltas,
            _PORTABILITY_CITATIONS,
        ),
    )
    for group_name, reducer, citations in reductions:
        group = probes[group_name]
        if not group:
            continue
        pairs = [
            PairedProbeScore(
                probe_id=cast(str, probe["probe_id"]),
                baseline_mean_log_probability=baseline_scores[
                    cast(str, probe["probe_id"])
                ],
                edited_mean_log_probability=edited_scores[cast(str, probe["probe_id"])],
            )
            for probe in group
        ]
        results.append(_probe_metric_result(reducer(pairs), citations=citations))

    control_group = probes["control"]
    if control_group:
        baseline_logits = _control_logits_index(baseline)
        edited_logits = _control_logits_index(edited)
        baseline_measurements = cast(Mapping[str, object], baseline["measurements"])
        temperature = cast(float, baseline_measurements["control_temperature"])
        paired_controls = [
            PairedControlLogits(
                probe_id=cast(str, probe["probe_id"]),
                baseline_logits=baseline_logits[cast(str, probe["probe_id"])],
                edited_logits=edited_logits[cast(str, probe["probe_id"])],
            )
            for probe in control_group
        ]
        results.append(
            _control_metric_result(
                reduce_control_kl_divergence(
                    paired_controls,
                    temperature=temperature,
                )
            )
        )
    return results


def _probe_metric_result(
    reduction: ProbeScoreReduction,
    *,
    citations: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "metric_id": reduction.metric_id,
        "status": "complete",
        "direction": reduction.direction,
        "unit": reduction.unit,
        "aggregate": reduction.aggregate,
        "reduction": reduction.reduction,
        "coverage": {
            "total": reduction.total_probe_count,
            "evaluated": reduction.evaluated_probe_count,
            "missing": reduction.missing_probe_count,
            "fraction": reduction.coverage,
        },
        "probes": [
            {
                "probe_id": probe.probe_id,
                "status": "evaluated",
                "values": {
                    "baseline_mean_log_probability": probe.baseline_mean_log_probability,
                    "edited_mean_log_probability": probe.edited_mean_log_probability,
                    "signed_delta": probe.signed_delta,
                    "absolute_delta": probe.absolute_delta,
                    "contribution": probe.contribution,
                },
                "missing_reason": None,
            }
            for probe in reduction.probes
        ],
        "warnings": list(reduction.warnings),
        "citations": [dict(citation) for citation in citations],
    }


def _control_metric_result(reduction: ControlDivergenceReduction) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "metric_id": reduction.metric_id,
        "status": "complete",
        "direction": reduction.direction,
        "unit": reduction.unit,
        "aggregate": reduction.aggregate,
        "reduction": reduction.reduction,
        "coverage": {
            "total": reduction.total_probe_count,
            "evaluated": reduction.evaluated_probe_count,
            "missing": reduction.missing_probe_count,
            "fraction": reduction.coverage,
        },
        "probes": [
            {
                "probe_id": probe.probe_id,
                "status": "evaluated",
                "values": {
                    "position_kl_divergences": list(
                        cast(tuple[float, ...], probe.position_kl_divergences)
                    ),
                    "mean_kl_divergence": probe.mean_kl_divergence,
                    "position_count": probe.position_count,
                    "vocabulary_size": probe.vocabulary_size,
                    "temperature": reduction.temperature,
                    "divergence_direction": reduction.divergence_direction,
                },
                "missing_reason": None,
            }
            for probe in reduction.probes
        ],
        "warnings": list(reduction.warnings),
        "citations": [dict(citation) for citation in _CONTROL_CITATIONS],
    }


def _build_report(
    *,
    case: Mapping[str, object],
    manifest: Mapping[str, object],
    metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    timestamps = cast(Mapping[str, object], manifest["timestamps"])
    manifest_case = cast(Mapping[str, object], manifest["audit_case"])
    provenance = cast(Mapping[str, object], case["provenance"])
    run_id = cast(str, manifest["run_id"])
    return {
        "schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "report_id": f"report-{run_id.removeprefix('run-')}",
        "status": "completed",
        "generated_at": timestamps["ended_at"],
        "manifest": manifest,
        "audit_case": {
            "case_id": case["case_id"],
            "schema_version": case["schema_version"],
            "artifact": manifest_case["artifact"],
            "dataset": {
                "name": provenance["dataset_name"],
                "version": provenance.get("dataset_version", "unspecified"),
                "license": provenance["dataset_license"],
                "source": provenance["source"],
            },
        },
        "metrics": [dict(metric) for metric in metrics],
        "structural_evidence": [],
        "limitations": [
            (
                "This report was assembled from caller-supplied data-only snapshots; "
                "KEditAudit did not load or execute a model, checkpoint, or editor."
            ),
            (
                "Snapshot provenance is caller-supplied, and this report contains no "
                "structural-weight or causal-tracing evidence."
            ),
            (
                "Audit results are diagnostic evidence and do not certify model safety "
                "or the absence of harmful behavior."
            ),
        ],
        "warnings": [],
    }


def _case_probe_ids(case: Mapping[str, object]) -> tuple[set[str], set[str]]:
    probes = cast(Mapping[str, Sequence[Mapping[str, object]]], case["probes"])
    targets = {
        cast(str, probe["probe_id"])
        for group_name in _TARGET_GROUPS
        for probe in probes[group_name]
    }
    controls = {cast(str, probe["probe_id"]) for probe in probes["control"]}
    return targets, controls


def _snapshot_probe_ids(snapshot: Mapping[str, object]) -> tuple[set[str], set[str]]:
    measurements = cast(Mapping[str, Sequence[Mapping[str, object]]], snapshot["measurements"])
    target_ids = {
        cast(str, entry["probe_id"]) for entry in measurements["target_scores"]
    }
    control_ids = {
        cast(str, entry["probe_id"]) for entry in measurements["control_logits"]
    }
    return target_ids, control_ids


def _target_score_index(snapshot: Mapping[str, object]) -> dict[str, float]:
    measurements = cast(Mapping[str, Sequence[Mapping[str, object]]], snapshot["measurements"])
    return {
        cast(str, entry["probe_id"]): cast(float, entry["mean_target_log_probability"])
        for entry in measurements["target_scores"]
    }


def _control_logits_index(
    snapshot: Mapping[str, object],
) -> dict[str, Sequence[Sequence[float]]]:
    measurements = cast(Mapping[str, Sequence[Mapping[str, object]]], snapshot["measurements"])
    return {
        cast(str, entry["probe_id"]): cast(
            Sequence[Sequence[float]], entry["logits"]
        )
        for entry in measurements["control_logits"]
    }


def _content_hash_identity(artifact: Mapping[str, object]) -> tuple[object, ...] | None:
    content_hash = artifact.get("content_hash")
    if not isinstance(content_hash, Mapping):
        return None
    return (
        content_hash.get("algorithm"),
        content_hash.get("encoding"),
        content_hash.get("digest"),
    )


def _read_utc_clock(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AuditPipelineInputError("clock must return a timezone-aware datetime")
    utc_value = value.astimezone(UTC)
    try:
        timestamp = utc_value.timestamp()
    except (OSError, OverflowError, ValueError) as error:
        raise AuditPipelineInputError("clock must return a representable UTC timestamp") from error
    if not math.isfinite(timestamp):
        raise AuditPipelineInputError("clock must return a finite UTC timestamp")
    return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(UTC)
