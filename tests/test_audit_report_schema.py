import copy
import json
import math
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from kedit_audit.artifacts import (
    AUDIT_REPORT_SCHEMA_VERSION,
    METRIC_RESULT_SCHEMA_VERSION,
    ArtifactHash,
    AuditReportValidationError,
    MetricResultValidationError,
    canonical_json_bytes,
    hash_json,
    load_audit_report_schema,
    load_metric_result_schema,
    validate_audit_report,
    validate_metric_result,
)

FIXTURE = Path(__file__).parent / "fixtures" / "audit_reports" / "valid" / "completed.json"


def _load_report() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_report_contract_schemas_are_valid_draft_2020_12() -> None:
    metric_schema = load_metric_result_schema()
    report_schema = load_audit_report_schema()

    validator_for(metric_schema).check_schema(metric_schema)
    validator_for(report_schema).check_schema(report_schema)
    assert metric_schema["properties"]["schema_version"]["const"] == (
        METRIC_RESULT_SCHEMA_VERSION
    )
    assert report_schema["properties"]["schema_version"]["const"] == (
        AUDIT_REPORT_SCHEMA_VERSION
    )


def test_completed_report_survives_json_round_trip_with_raw_evidence() -> None:
    report = _load_report()

    validate_audit_report(report)
    encoded = canonical_json_bytes(report)
    round_tripped = json.loads(encoded)
    validate_audit_report(round_tripped)

    assert round_tripped == report
    assert round_tripped["metrics"][0]["probes"][0]["values"] == {
        "absolute_delta": 1.0,
        "baseline_mean_log_probability": -2.0,
        "contribution": 1.0,
        "edited_mean_log_probability": -1.0,
        "signed_delta": 1.0,
    }
    assert round_tripped["limitations"]
    assert hash_json(round_tripped) == hash_json(report) == ArtifactHash(
        algorithm="sha256",
        encoding="kedit-audit-canonical-json-v1",
        digest="67bb4e1adbc3e0e321e44fa25536c760549519ca6ad16efc68a54c52a778bf6b",
        size_bytes=4354,
    )


def test_metric_result_rejects_inconsistent_coverage_with_actionable_path() -> None:
    metric = copy.deepcopy(_load_report()["metrics"][0])
    metric["coverage"]["evaluated"] = 0

    with pytest.raises(MetricResultValidationError) as error:
        validate_metric_result(metric)

    assert any(
        issue.path == "$.coverage.evaluated" and "evaluated probes" in issue.message
        for issue in error.value.issues
    )


def test_report_rejects_duplicate_metric_ids() -> None:
    report = _load_report()
    report["metrics"][1]["metric_id"] = report["metrics"][0]["metric_id"]

    with pytest.raises(AuditReportValidationError) as error:
        validate_audit_report(report)

    assert any(
        issue.path == "$.metrics[1].metric_id" and "first declared" in issue.message
        for issue in error.value.issues
    )


def test_report_rejects_case_reference_that_disagrees_with_manifest() -> None:
    report = _load_report()
    report["audit_case"]["artifact"]["content_hash"]["digest"] = "f" * 64

    with pytest.raises(AuditReportValidationError) as error:
        validate_audit_report(report)

    assert any(
        issue.path == "$.audit_case.artifact.content_hash.digest"
        and "manifest" in issue.message
        for issue in error.value.issues
    )


def test_report_prefixes_nested_manifest_semantic_errors() -> None:
    report = _load_report()
    report["manifest"]["timestamps"]["ended_at"] = "2026-08-20T01:02:02Z"

    with pytest.raises(AuditReportValidationError) as error:
        validate_audit_report(report)

    assert any(
        issue.path == "$.manifest.timestamps.ended_at" and "started_at" in issue.message
        for issue in error.value.issues
    )


def test_report_cannot_be_generated_before_the_run_ended() -> None:
    report = _load_report()
    report["generated_at"] = "2026-08-20T01:02:04Z"

    with pytest.raises(AuditReportValidationError) as error:
        validate_audit_report(report)

    assert any(
        issue.path == "$.generated_at" and "ended_at" in issue.message
        for issue in error.value.issues
    )


def test_completed_report_cannot_hide_incomplete_metric() -> None:
    report = _load_report()
    report["metrics"][0]["status"] = "incomplete"
    report["metrics"][0]["warnings"] = ["fixture simulates an incomplete metric"]

    with pytest.raises(AuditReportValidationError) as error:
        validate_audit_report(report)

    paths = {issue.path for issue in error.value.issues}
    assert "$.metrics[0].status" in paths
    assert "$.status" in paths


def test_report_rejects_non_finite_structural_value() -> None:
    report = _load_report()
    report["structural_evidence"] = [
        {
            "evidence_id": "weight-diff-placeholder",
            "evidence_type": "weight-diff",
            "status": "complete",
            "values": {"frobenius_norm": math.inf},
            "warnings": [],
            "citations": [
                {
                    "title": "Synthetic test method",
                    "source": "https://example.invalid/method",
                }
            ],
        }
    ]

    with pytest.raises(AuditReportValidationError) as error:
        validate_audit_report(report)

    assert any(
        issue.path == "$.structural_evidence[0].values.frobenius_norm"
        and "finite" in issue.message
        for issue in error.value.issues
    )
