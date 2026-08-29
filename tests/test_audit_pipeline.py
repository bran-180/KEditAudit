from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import kedit_audit.audit.pipeline as pipeline_module
from kedit_audit.artifacts import hash_json, validate_audit_report, validate_run_manifest
from kedit_audit.audit import (
    AuditExecutionError,
    AuditPipelineInputError,
    run_audit_pipeline,
)
from kedit_audit.reporting import AuditReportWriteError

FIXTURES = Path(__file__).parent / "fixtures"
CASE_FIXTURE = FIXTURES / "audit_cases" / "valid" / "basic.json"
SNAPSHOT_FIXTURES = FIXTURES / "audit_snapshots" / "valid"


def _document(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _case() -> dict[str, object]:
    return _document(CASE_FIXTURE)


def _baseline() -> dict[str, object]:
    return _document(SNAPSHOT_FIXTURES / "baseline.json")


def _edited() -> dict[str, object]:
    return _document(SNAPSHOT_FIXTURES / "edited.json")


def _clock() -> Callable[[], datetime]:
    values = iter(
        (
            datetime(2026, 8, 29, 1, 2, 3, tzinfo=UTC),
            datetime(2026, 8, 29, 1, 2, 5, tzinfo=UTC),
        )
    )
    return lambda: next(values)


def test_pipeline_writes_completed_manifest_json_and_markdown(tmp_path: Path) -> None:
    audit_case = _case()
    result = run_audit_pipeline(
        audit_case=audit_case,
        baseline_snapshot=_baseline(),
        edited_snapshot=_edited(),
        output_directory=tmp_path,
        clock=_clock(),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    validate_run_manifest(manifest)
    validate_audit_report(report)
    assert manifest["status"] == "completed"
    assert report["manifest"] == manifest
    assert report["generated_at"] == "2026-08-29T01:02:05.000000Z"
    assert result.report_markdown_path.is_file()
    metric_by_id = {metric["metric_id"]: metric for metric in report["metrics"]}
    assert set(metric_by_id) == {
        "efficacy.mean_target_log_probability_delta",
        "generality.mean_target_log_probability_delta",
        "locality.mean_absolute_target_log_probability_delta",
        "portability.mean_expected_target_log_probability_delta",
        "control.mean_kl_divergence",
    }
    assert metric_by_id["efficacy.mean_target_log_probability_delta"]["aggregate"] == 2.0
    assert metric_by_id["generality.mean_target_log_probability_delta"]["aggregate"] == 0.75
    assert metric_by_id[
        "locality.mean_absolute_target_log_probability_delta"
    ]["aggregate"] == pytest.approx(0.1)
    assert metric_by_id[
        "portability.mean_expected_target_log_probability_delta"
    ]["aggregate"] == 1.0
    case_hash = report["audit_case"]["artifact"]["content_hash"]
    assert case_hash == hash_json(audit_case).as_dict()
    serialized_outputs = result.report_json_path.read_text(
        encoding="utf-8"
    ) + result.report_markdown_path.read_text(encoding="utf-8")
    assert "The Eiffel Tower" not in serialized_outputs
    assert "Water freezes at" not in serialized_outputs
    assert not list(tmp_path.glob("*.tmp"))


def test_preflight_incompatibility_creates_no_misleading_manifest(tmp_path: Path) -> None:
    edited = _edited()
    model = cast(dict[str, object], edited["model"])
    model["tokenizer_revision"] = "incompatible-revision"

    with pytest.raises(AuditPipelineInputError, match="tokenizer_revision"):
        run_audit_pipeline(
            audit_case=_case(),
            baseline_snapshot=_baseline(),
            edited_snapshot=edited,
            output_directory=tmp_path,
            clock=_clock(),
        )

    assert not list(tmp_path.iterdir())


def test_metric_failure_leaves_failed_manifest_without_private_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_text = "private model output must not enter the manifest"

    def fail_reduction(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(private_text)

    monkeypatch.setattr(
        pipeline_module,
        "reduce_control_kl_divergence",
        fail_reduction,
    )

    with pytest.raises(AuditExecutionError) as raised:
        run_audit_pipeline(
            audit_case=_case(),
            baseline_snapshot=_baseline(),
            edited_snapshot=_edited(),
            output_directory=tmp_path,
            clock=_clock(),
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    manifest_text = (tmp_path / "run-manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    validate_run_manifest(manifest)
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "audit-pipeline"
    assert private_text not in manifest_text
    assert not (tmp_path / "audit-report.json").exists()
    assert not (tmp_path / "audit-report.md").exists()


def test_report_finalization_failure_replaces_running_manifest_with_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_text = "private report value"

    def fail_writer(*_args: object, **_kwargs: object) -> object:
        raise AuditReportWriteError(private_text)

    monkeypatch.setattr(pipeline_module, "write_audit_report", fail_writer)

    with pytest.raises(AuditExecutionError) as raised:
        run_audit_pipeline(
            audit_case=copy.deepcopy(_case()),
            baseline_snapshot=copy.deepcopy(_baseline()),
            edited_snapshot=copy.deepcopy(_edited()),
            output_directory=tmp_path,
            clock=_clock(),
        )

    assert isinstance(raised.value.__cause__, AuditReportWriteError)
    manifest_text = (tmp_path / "run-manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "failed"
    assert manifest["failure"]["error_type"] == "AuditReportWriteError"
    assert private_text not in manifest_text
