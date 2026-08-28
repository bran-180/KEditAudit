from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import cast

import pytest
from jsonschema.validators import validator_for

from kedit_audit.artifacts import (
    REPORT_COMPARISON_SCHEMA_VERSION,
    ReportComparisonValidationError,
    load_report_comparison_schema,
    validate_audit_report,
    validate_report_comparison,
)
from kedit_audit.cli import main
from kedit_audit.reporting import ReportComparisonError, compare_audit_reports

FIXTURE = Path(__file__).parent / "fixtures" / "audit_reports" / "valid" / "completed.json"


def _report() -> dict[str, object]:
    return cast(dict[str, object], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _report_b() -> dict[str, object]:
    report = _report()
    report["report_id"] = "report-20260820-002"
    manifest = cast(dict[str, object], report["manifest"])
    manifest["run_id"] = "run-20260820-002"
    model = cast(dict[str, object], manifest["model"])
    edited = cast(dict[str, object], model["edited"])
    edited["artifact_id"] = "tiny-edited-b"
    cast(dict[str, object], edited["content_hash"])["digest"] = "3" * 64
    metrics = cast(list[dict[str, object]], report["metrics"])
    metrics[0]["aggregate"] = 0.5
    validate_audit_report(report)
    return report


def test_comparison_schema_is_packaged_draft_2020_12() -> None:
    schema = load_report_comparison_schema()
    validator_for(schema).check_schema(schema)
    assert schema["properties"]["schema_version"]["const"] == (
        REPORT_COMPARISON_SCHEMA_VERSION
    )


def test_compatible_reports_emit_sorted_descriptive_metric_deltas() -> None:
    comparison = compare_audit_reports(_report(), _report_b())
    document = comparison.as_dict()

    validate_report_comparison(document)
    assert document["report_a"]["content_hash"]["digest"] != (  # type: ignore[index]
        document["report_b"]["content_hash"]["digest"]  # type: ignore[index]
    )
    rows = cast(list[dict[str, object]], document["metrics"])
    assert [row["metric_id"] for row in rows] == sorted(
        row["metric_id"] for row in rows
    )
    generality = next(
        row
        for row in rows
        if row["metric_id"] == "generality.mean_target_log_probability_delta"
    )
    assert generality["presence"] == "both"
    assert generality["comparable"] is True
    assert generality["aggregate_delta_b_minus_a"] == pytest.approx(0.125)
    assert any("descriptive" in warning for warning in document["warnings"])


def test_missing_metric_is_explicit_not_zero_filled() -> None:
    report_b = _report_b()
    metrics = cast(list[dict[str, object]], report_b["metrics"])
    metrics.pop(0)
    validate_audit_report(report_b)

    document = compare_audit_reports(_report(), report_b).as_dict()
    row = next(
        row
        for row in cast(list[dict[str, object]], document["metrics"])
        if row["metric_id"] == "generality.mean_target_log_probability_delta"
    )

    assert row["presence"] == "only-a"
    assert row["report_b"] is None
    assert row["aggregate_delta_b_minus_a"] is None


def test_metric_contract_mismatch_suppresses_delta() -> None:
    report_b = _report_b()
    cast(list[dict[str, object]], report_b["metrics"])[0]["unit"] = "bits"
    validate_audit_report(report_b)

    document = compare_audit_reports(_report(), report_b).as_dict()
    row = next(
        row
        for row in cast(list[dict[str, object]], document["metrics"])
        if row["metric_id"] == "generality.mean_target_log_probability_delta"
    )

    assert row["comparable"] is False
    assert row["unit"] is None
    assert row["aggregate_delta_b_minus_a"] is None
    assert "unit" in cast(list[str], row["warnings"])[0]


def test_mismatched_case_or_baseline_context_fails_closed() -> None:
    report_b = _report_b()
    case = cast(dict[str, object], report_b["audit_case"])
    case_artifact = cast(dict[str, object], case["artifact"])
    cast(dict[str, object], case_artifact["content_hash"])["digest"] = "f" * 64
    manifest = cast(dict[str, object], report_b["manifest"])
    manifest_case = cast(dict[str, object], manifest["audit_case"])
    manifest_artifact = cast(dict[str, object], manifest_case["artifact"])
    cast(dict[str, object], manifest_artifact["content_hash"])["digest"] = "f" * 64
    validate_audit_report(report_b)

    with pytest.raises(ReportComparisonError, match=r"\$\.audit_case\.artifact"):
        compare_audit_reports(_report(), report_b)


def test_comparison_does_not_mutate_reports() -> None:
    report_a = _report()
    report_b = _report_b()
    original_a = copy.deepcopy(report_a)
    original_b = copy.deepcopy(report_b)

    compare_audit_reports(report_a, report_b)

    assert report_a == original_a
    assert report_b == original_b


def test_comparison_validator_rejects_tampered_delta() -> None:
    document = compare_audit_reports(_report(), _report_b()).as_dict()
    rows = cast(list[dict[str, object]], document["metrics"])
    rows[0]["aggregate_delta_b_minus_a"] = 999.0

    with pytest.raises(ReportComparisonValidationError) as raised:
        validate_report_comparison(document)

    assert any("report_b.aggregate" in issue.message for issue in raised.value.issues)


def test_non_finite_subtraction_is_suppressed_with_warning() -> None:
    report_a = _report()
    report_b = _report_b()
    cast(list[dict[str, object]], report_a["metrics"])[0]["aggregate"] = -1e308
    cast(list[dict[str, object]], report_b["metrics"])[0]["aggregate"] = 1e308

    document = compare_audit_reports(report_a, report_b).as_dict()
    row = next(
        row
        for row in cast(list[dict[str, object]], document["metrics"])
        if row["metric_id"] == "generality.mean_target_log_probability_delta"
    )

    assert row["aggregate_delta_b_minus_a"] is None
    assert "finite float range" in cast(list[str], row["warnings"])[0]
    validate_report_comparison(document)


def test_compare_cli_prints_valid_json(tmp_path: Path) -> None:
    report_a_path = tmp_path / "a.json"
    report_b_path = tmp_path / "b.json"
    report_a_path.write_text(json.dumps(_report()), encoding="utf-8")
    report_b_path.write_text(json.dumps(_report_b()), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["compare", str(report_a_path), str(report_b_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    document = json.loads(stdout.getvalue())
    validate_report_comparison(document)
