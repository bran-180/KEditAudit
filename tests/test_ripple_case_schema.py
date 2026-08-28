import copy
import json
from pathlib import Path

import pytest

from kedit_audit.artifacts import (
    RIPPLE_CASE_SCHEMA_VERSION,
    RippleCaseValidationError,
    load_ripple_case_schema,
    validate_ripple_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "ripple_cases" / "valid" / "synthetic.json"


def _case() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_synthetic_ripple_case_validates_and_round_trips() -> None:
    schema = load_ripple_case_schema()
    case = _case()

    validate_ripple_case(case)

    assert schema["$id"].endswith(f"/{RIPPLE_CASE_SCHEMA_VERSION}")
    assert case["artifact_kind"] == "synthetic-contract-fixture"
    assert case["provenance"]["dataset_license"] == "Apache-2.0"
    assert json.loads(json.dumps(case, sort_keys=True)) == case


def test_probe_ids_must_be_unique_across_all_ripple_categories() -> None:
    case = _case()
    duplicate = copy.deepcopy(case["probes"]["logical_generalization"][0])
    duplicate["probe_id"] = "relation-specificity-1"
    case["probes"]["logical_generalization"].append(duplicate)

    with pytest.raises(RippleCaseValidationError) as error:
        validate_ripple_case(case)

    assert any(
        issue.path == "$.probes.logical_generalization[1].probe_id"
        and "first declared" in issue.message
        for issue in error.value.issues
    )


def test_edit_targets_must_differ_and_at_least_one_probe_is_required() -> None:
    case = _case()
    case["edit"]["target_original"] = "Harbor City"
    for group in case["probes"].values():
        group.clear()

    with pytest.raises(RippleCaseValidationError) as error:
        validate_ripple_case(case)

    paths = {issue.path for issue in error.value.issues}
    assert "$.edit.target_original" in paths
    assert "$.probes" in paths


def test_dataset_license_and_source_revision_are_mandatory() -> None:
    case = _case()
    del case["provenance"]["dataset_license"]
    case["provenance"]["source_revision"] = "main"

    with pytest.raises(RippleCaseValidationError) as error:
        validate_ripple_case(case)

    paths = {issue.path for issue in error.value.issues}
    assert "$.provenance" in paths
    assert "$.provenance.source_revision" in paths
