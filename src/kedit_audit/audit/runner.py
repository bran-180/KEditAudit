"""Manifest-first audit execution with persistent failure evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, TypeVar, cast

from kedit_audit.artifacts import (
    RUN_MANIFEST_SCHEMA_VERSION,
    ArtifactWriteError,
    RunManifestValidationError,
    canonical_json_bytes,
    validate_run_manifest,
    write_bytes_atomically,
)

MANIFEST_FILENAME = "run-manifest.json"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PUBLIC_FAILURE_MESSAGE = (
    "audit operation failed; inspect the local exception chain for private diagnostic details"
)
T = TypeVar("T")


class AuditRunnerValidationError(ValueError):
    """Raised before evaluation when runner inputs or output state are unsafe."""


class AuditExecutionError(RuntimeError):
    """Raised after an operation fails and its failure manifest is persisted."""

    def __init__(self, *, manifest_path: Path, manifest: Mapping[str, object]) -> None:
        self.manifest_path = manifest_path
        self.manifest = manifest
        super().__init__(f"audit failed; failure manifest written to {manifest_path}")


@dataclass(frozen=True)
class AuditExecutionResult(Generic[T]):
    """Successful operation value plus its validated completed manifest."""

    value: T
    manifest_path: Path
    manifest: Mapping[str, object]


def execute_audit(
    *,
    initial_manifest: Mapping[str, object],
    output_directory: str | Path,
    operation: Callable[[], T],
    finalize: Callable[[T, Mapping[str, object]], None] | None = None,
    failure_stage: str = "evaluation",
    clock: Callable[[], datetime] | None = None,
) -> AuditExecutionResult[T]:
    """Persist running/final states around one caller-owned audit operation."""

    _require_identifier(failure_stage, path="failure_stage")
    manifest = _copy_json_mapping(initial_manifest)
    try:
        validate_run_manifest(manifest)
    except RunManifestValidationError as error:
        raise AuditRunnerValidationError(
            "initial_manifest must satisfy the RunManifest contract"
        ) from error
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise AuditRunnerValidationError(
            f"initial_manifest.schema_version must equal {RUN_MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("status") != "running":
        raise AuditRunnerValidationError("initial_manifest.status must equal 'running'")

    manifest_path = write_run_manifest(manifest, output_directory=output_directory)
    read_clock = clock if clock is not None else _utc_now
    try:
        value = operation()
        completed = _terminal_manifest(
            manifest,
            status="completed",
            ended_at=_read_utc_clock(read_clock),
            failure=None,
        )
        if finalize is not None:
            finalize(value, completed)
        manifest_path = write_run_manifest(
            completed,
            output_directory=output_directory,
        )
    except Exception as error:
        failed = _terminal_manifest(
            manifest,
            status="failed",
            ended_at=_failure_ended_at(read_clock, manifest=manifest),
            failure={
                "stage": failure_stage,
                "error_type": _public_error_type(error),
                "message": _PUBLIC_FAILURE_MESSAGE,
            },
        )
        manifest_path = write_run_manifest(failed, output_directory=output_directory)
        raise AuditExecutionError(
            manifest_path=manifest_path,
            manifest=failed,
        ) from error

    return AuditExecutionResult(
        value=value,
        manifest_path=manifest_path,
        manifest=completed,
    )


def write_run_manifest(
    manifest: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> Path:
    """Validate and atomically persist the fixed-name RunManifest artifact."""

    normalized = _copy_json_mapping(manifest)
    try:
        validate_run_manifest(normalized)
    except RunManifestValidationError as error:
        raise AuditRunnerValidationError(
            "manifest must satisfy the RunManifest contract before writing"
        ) from error
    target = Path(output_directory) / MANIFEST_FILENAME
    encoded = canonical_json_bytes(normalized) + b"\n"
    try:
        write_bytes_atomically(target, encoded)
    except ArtifactWriteError as error:
        raise AuditRunnerValidationError("run manifest could not be written atomically") from error
    return target


def _terminal_manifest(
    initial: dict[str, object],
    *,
    status: str,
    ended_at: str,
    failure: dict[str, object] | None,
) -> dict[str, object]:
    result = _copy_json_mapping(initial)
    timestamps = cast(dict[str, object], result["timestamps"])
    timestamps["ended_at"] = ended_at
    result["status"] = status
    result["failure"] = failure
    try:
        validate_run_manifest(result)
    except RunManifestValidationError as error:
        raise AuditRunnerValidationError(
            f"terminal {status!r} manifest does not satisfy the RunManifest contract"
        ) from error
    return result


def _copy_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AuditRunnerValidationError("manifest must be a mapping")
    try:
        encoded = canonical_json_bytes(value)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as error:
        raise AuditRunnerValidationError("manifest must contain bounded finite JSON values") from error
    if not isinstance(normalized, dict):
        raise AuditRunnerValidationError("manifest must be a JSON object")
    return cast(dict[str, object], normalized)


def _read_utc_clock(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AuditRunnerValidationError("clock must return a timezone-aware datetime")
    utc_value = value.astimezone(UTC)
    return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _failure_ended_at(
    clock: Callable[[], datetime],
    *,
    manifest: Mapping[str, object],
) -> str:
    timestamps = cast(Mapping[str, object], manifest["timestamps"])
    started_at = cast(str, timestamps["started_at"])
    try:
        candidate = _read_utc_clock(clock)
    except Exception:  # noqa: BLE001 - failure persistence must not mask the primary error.
        return started_at
    if _parse_utc_timestamp(candidate) < _parse_utc_timestamp(started_at):
        return started_at
    return candidate


def _parse_utc_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _public_error_type(error: Exception) -> str:
    candidate = type(error).__name__
    if _IDENTIFIER.fullmatch(candidate) is None:
        return "Exception"
    return candidate


def _require_identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise AuditRunnerValidationError(
            f"{path} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}"
        )
    return value
