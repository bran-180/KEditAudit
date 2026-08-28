import copy
import json
from pathlib import Path

import pytest

from kedit_audit.artifacts import (
    EDITOR_ARTIFACT_SCHEMA_VERSION,
    EditorArtifactManifestValidationError,
    load_editor_artifact_schema,
    validate_editor_artifact_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "editor_artifacts" / "valid" / "rome.json"


def _manifest() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_editor_artifact_schema_is_packaged_and_validates_fixture() -> None:
    schema = load_editor_artifact_schema()
    manifest = _manifest()

    assert schema["$id"].endswith(f"/{EDITOR_ARTIFACT_SCHEMA_VERSION}")
    validate_editor_artifact_manifest(manifest)
    assert json.loads(json.dumps(manifest, sort_keys=True)) == manifest


@pytest.mark.parametrize(
    ("mutation", "path_fragment"),
    [
        (lambda value: value["model"]["edited"].update({"state_id": "baseline-rome-fixture"}),
         "$.model.edited.state_id"),
        (lambda value: value["changed_tensors"].append(copy.deepcopy(value["changed_tensors"][0])),
         "$.changed_tensors[1].name"),
        (lambda value: value["changed_tensors"][0].update({"edited_sha256": "3" * 64}),
         "$.changed_tensors[0].edited_sha256"),
    ],
)
def test_editor_artifact_semantics_reject_ambiguous_state_or_inventory(
    mutation: object,
    path_fragment: str,
) -> None:
    manifest = _manifest()
    mutation(manifest)  # type: ignore[operator]

    with pytest.raises(EditorArtifactManifestValidationError) as error:
        validate_editor_artifact_manifest(manifest)

    assert any(issue.path == path_fragment for issue in error.value.issues)
