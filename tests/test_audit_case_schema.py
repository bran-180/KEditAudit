import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from kedit_audit.artifacts import (
    AUDIT_CASE_SCHEMA_VERSION,
    AuditCaseValidationError,
    load_audit_case_schema,
    validate_audit_case,
)

FIXTURES = Path(__file__).parent / "fixtures" / "audit_cases"


def _load_fixture(kind: str, name: str) -> object:
    return json.loads((FIXTURES / kind / name).read_text(encoding="utf-8"))


def test_audit_case_schema_is_valid_draft_2020_12() -> None:
    schema = load_audit_case_schema()
    validator = validator_for(schema)

    validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == AUDIT_CASE_SCHEMA_VERSION


def test_valid_fixture_survives_json_round_trip() -> None:
    audit_case = _load_fixture("valid", "basic.json")

    validate_audit_case(audit_case)
    round_tripped = json.loads(json.dumps(audit_case, ensure_ascii=False, sort_keys=True))
    validate_audit_case(round_tripped)

    assert round_tripped == audit_case
    assert round_tripped["probes"]["portability"][0]["expected_relationship"] == {
        "relation": "entails",
        "target": " Italy",
    }
    assert round_tripped["provenance"]["dataset_license"] == "CC0-1.0"


@pytest.mark.parametrize(
    ("fixture_name", "expected_path", "expected_message"),
    [
        (
            "missing_dataset_license.json",
            "$.provenance",
            "dataset_license",
        ),
        (
            "bad_prompt_template.json",
            "$.edit.prompt_template",
            "{subject}",
        ),
        (
            "duplicate_probe_id.json",
            "$.probes.paraphrase[0].probe_id",
            "reused-id",
        ),
    ],
)
def test_invalid_fixtures_report_actionable_paths(
    fixture_name: str,
    expected_path: str,
    expected_message: str,
) -> None:
    audit_case = _load_fixture("invalid", fixture_name)

    with pytest.raises(AuditCaseValidationError) as error:
        validate_audit_case(audit_case)

    assert any(
        issue.path == expected_path and expected_message in issue.message
        for issue in error.value.issues
    )
