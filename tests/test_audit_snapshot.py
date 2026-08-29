from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from kedit_audit.artifacts import (
    AuditSnapshotValidationError,
    load_audit_snapshot_schema,
    validate_audit_snapshot,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "audit_snapshots" / "valid"


def _snapshot(name: str = "baseline.json") -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads((FIXTURE_DIRECTORY / name).read_text(encoding="utf-8")),
    )


def test_packaged_snapshot_schema_and_fixtures_validate() -> None:
    schema = load_audit_snapshot_schema()

    assert schema["$id"] == "urn:kedit-audit:schema:audit-snapshot:1.0.0"
    validate_audit_snapshot(_snapshot("baseline.json"))
    validate_audit_snapshot(_snapshot("edited.json"))


def test_duplicate_probe_ids_fail_across_measurement_groups() -> None:
    snapshot = _snapshot()
    measurements = cast(dict[str, object], snapshot["measurements"])
    controls = cast(list[dict[str, object]], measurements["control_logits"])
    controls[0]["probe_id"] = "exact-1"

    with pytest.raises(AuditSnapshotValidationError) as raised:
        validate_audit_snapshot(snapshot)

    assert any(
        issue.path == "$.measurements.control_logits[0].probe_id"
        and "duplicate probe_id" in issue.message
        for issue in raised.value.issues
    )


def test_snapshot_rejects_nonfinite_scores_and_unbounded_fields() -> None:
    snapshot = _snapshot()
    original = copy.deepcopy(snapshot)
    measurements = cast(dict[str, object], snapshot["measurements"])
    scores = cast(list[dict[str, object]], measurements["target_scores"])
    scores[0]["mean_target_log_probability"] = float("nan")
    snapshot["private_extra"] = "not permitted"

    with pytest.raises(AuditSnapshotValidationError) as raised:
        validate_audit_snapshot(snapshot)

    paths = {issue.path for issue in raised.value.issues}
    assert "$" in paths
    assert "$.measurements.target_scores[0].mean_target_log_probability" in paths
    assert original != snapshot
