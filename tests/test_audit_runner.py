from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from kedit_audit.artifacts import validate_run_manifest
from kedit_audit.audit import (
    MANIFEST_FILENAME,
    AuditExecutionError,
    AuditRunnerValidationError,
    execute_audit,
)

FIXTURE = Path(__file__).parent / "fixtures" / "run_manifests" / "valid" / "completed.json"


def _running_manifest() -> dict[str, object]:
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["timestamps"]["ended_at"] = None
    manifest["failure"] = None
    validate_run_manifest(manifest)
    return cast(dict[str, object], manifest)


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 29, 1, 2, 5, tzinfo=UTC)


def test_success_persists_running_then_completed_manifest(tmp_path: Path) -> None:
    observed_running: dict[str, object] = {}

    def operation() -> str:
        path = tmp_path / MANIFEST_FILENAME
        observed_running.update(json.loads(path.read_text(encoding="utf-8")))
        return "evidence-ready"

    result = execute_audit(
        initial_manifest=_running_manifest(),
        output_directory=tmp_path,
        operation=operation,
        clock=_fixed_clock,
    )

    assert observed_running["status"] == "running"
    assert result.value == "evidence-ready"
    assert result.manifest["status"] == "completed"
    assert result.manifest["failure"] is None
    assert result.manifest["timestamps"]["ended_at"] == "2026-08-29T01:02:05.000000Z"  # type: ignore[index]
    persisted = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert persisted == result.manifest
    validate_run_manifest(persisted)
    assert not list(tmp_path.glob("*.tmp"))


def test_failure_manifest_is_persisted_without_private_exception_text(tmp_path: Path) -> None:
    private_text = "private prompt must not be persisted"

    def operation() -> None:
        raise RuntimeError(private_text)

    with pytest.raises(AuditExecutionError) as raised:
        execute_audit(
            initial_manifest=_running_manifest(),
            output_directory=tmp_path,
            operation=operation,
            failure_stage="metric-evaluation",
            clock=_fixed_clock,
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    validate_run_manifest(manifest)
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "stage": "metric-evaluation",
        "error_type": "RuntimeError",
        "message": (
            "audit operation failed; inspect the local exception chain for private "
            "diagnostic details"
        ),
    }
    assert private_text not in json.dumps(manifest)


def test_invalid_initial_manifest_fails_before_creating_output(tmp_path: Path) -> None:
    invalid = _running_manifest()
    invalid["status"] = "completed"
    output = tmp_path / "not-created"

    with pytest.raises(AuditRunnerValidationError, match="RunManifest contract"):
        execute_audit(
            initial_manifest=invalid,
            output_directory=output,
            operation=lambda: None,
            clock=_fixed_clock,
        )

    assert not output.exists()


def test_runner_does_not_mutate_caller_manifest(tmp_path: Path) -> None:
    manifest = _running_manifest()
    original = copy.deepcopy(manifest)

    execute_audit(
        initial_manifest=manifest,
        output_directory=tmp_path,
        operation=lambda: None,
        clock=_fixed_clock,
    )

    assert manifest == original


def test_invalid_failure_stage_fails_before_writing(tmp_path: Path) -> None:
    with pytest.raises(AuditRunnerValidationError, match="failure_stage"):
        execute_audit(
            initial_manifest=_running_manifest(),
            output_directory=tmp_path,
            operation=lambda: None,
            failure_stage="private stage with spaces",
            clock=_fixed_clock,
        )

    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_naive_clock_failure_still_preserves_a_failure_manifest(tmp_path: Path) -> None:
    with pytest.raises(AuditExecutionError) as raised:
        execute_audit(
            initial_manifest=_running_manifest(),
            output_directory=tmp_path,
            operation=lambda: "completed-evaluation",
            clock=lambda: datetime(2026, 8, 29, 1, 2, 5),  # noqa: DTZ001
        )

    assert isinstance(raised.value.__cause__, AuditRunnerValidationError)
    persisted = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
