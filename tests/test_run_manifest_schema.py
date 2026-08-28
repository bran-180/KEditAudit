import copy
import json
import math
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from kedit_audit.artifacts import (
    RUN_MANIFEST_SCHEMA_VERSION,
    RunManifestValidationError,
    load_run_manifest_schema,
    validate_run_manifest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "run_manifests" / "valid" / "completed.json"


def _load_manifest() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_run_manifest_schema_is_valid_draft_2020_12() -> None:
    schema = load_run_manifest_schema()
    validator = validator_for(schema)

    validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == RUN_MANIFEST_SCHEMA_VERSION


def test_completed_manifest_survives_json_round_trip() -> None:
    manifest = _load_manifest()

    validate_run_manifest(manifest)
    round_tripped = json.loads(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    validate_run_manifest(round_tripped)

    assert round_tripped == manifest


def test_hash_unavailable_requires_an_explicit_reason() -> None:
    manifest = _load_manifest()
    baseline = manifest["model"]["baseline"]
    del baseline["content_hash"]
    baseline["hash_unavailable"] = {
        "reason": "not-permitted",
        "details": "Checkpoint license prohibits redistribution-derived fingerprints.",
    }

    validate_run_manifest(manifest)
    del baseline["hash_unavailable"]["reason"]

    with pytest.raises(RunManifestValidationError) as error:
        validate_run_manifest(manifest)

    assert any(issue.path == "$.model.baseline.hash_unavailable" for issue in error.value.issues)


def test_baseline_and_edited_must_be_logically_distinct() -> None:
    manifest = _load_manifest()
    edited = manifest["model"]["edited"]
    edited["artifact_id"] = manifest["model"]["baseline"]["artifact_id"]
    edited["content_hash"] = copy.deepcopy(manifest["model"]["baseline"]["content_hash"])

    with pytest.raises(RunManifestValidationError) as error:
        validate_run_manifest(manifest)

    paths = {issue.path for issue in error.value.issues}
    assert "$.model.edited.artifact_id" in paths
    assert "$.model.edited.content_hash.digest" in paths


def test_failed_manifest_requires_failure_details() -> None:
    manifest = _load_manifest()
    manifest["status"] = "failed"

    with pytest.raises(RunManifestValidationError) as error:
        validate_run_manifest(manifest)

    assert any(issue.path == "$.failure" for issue in error.value.issues)


def test_end_timestamp_cannot_precede_start() -> None:
    manifest = _load_manifest()
    manifest["timestamps"]["ended_at"] = "2026-08-20T01:02:02Z"

    with pytest.raises(RunManifestValidationError) as error:
        validate_run_manifest(manifest)

    assert any(issue.path == "$.timestamps.ended_at" for issue in error.value.issues)


def test_manifest_rejects_non_finite_configuration_number() -> None:
    manifest = _load_manifest()
    manifest["generation"]["temperature"] = math.inf

    with pytest.raises(RunManifestValidationError) as error:
        validate_run_manifest(manifest)

    assert any(issue.path == "$.generation.temperature" for issue in error.value.issues)
