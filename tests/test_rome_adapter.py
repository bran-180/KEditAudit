import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kedit_audit.adapters import (
    AdapterCompatibilityError,
    FakeModelAdapter,
    ModelMetadata,
    RomeArtifactAdapter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "editor_artifacts" / "valid" / "rome.json"


class GPT2LMHeadModel:
    __module__ = "transformers.models.gpt2.modeling_gpt2"


def _manifest() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _model_adapter(*, state_kind: str, state_id: str, root: object) -> FakeModelAdapter:
    return FakeModelAdapter(
        metadata=ModelMetadata(
            model_id="synthetic/rome-model",
            model_revision="model-revision-1",
            tokenizer_id="synthetic/tokenizer",
            tokenizer_revision="tokenizer-revision-1",
            state_id=state_id,
            state_kind=state_kind,  # type: ignore[arg-type]
            device="cpu",
            dtype="float32",
        ),
        token_ids_by_text={"prompt": [0], " target": [1]},
        logits_by_prompt_target={("prompt", " target"): [[0.0, 1.0]]},
        module_root=root,
    )


def test_rome_importer_retains_pinned_provenance_and_inventory() -> None:
    adapter = RomeArtifactAdapter.from_manifest(_manifest())
    serialized = adapter.as_dict()

    assert adapter.metadata.editor_name == "ROME"
    assert adapter.metadata.editor_revision == "0874014cd9837e4365f3e6f3c71400ef11509e04"
    assert adapter.metadata.source_repository == "https://github.com/kmeng01/rome"
    assert adapter.metadata.hyperparameters["layers"] == (0,)
    assert len(adapter.changed_tensors) == 1
    assert adapter.changed_tensors[0].name == "transformer.h.0.mlp.c_proj.weight"
    assert serialized["artifact_kind"] == "synthetic-contract-fixture"
    assert "fixture_notice" in serialized["metadata"]["hyperparameters"]


def test_rome_importer_binds_only_exact_supported_gpt2_roots() -> None:
    adapter = RomeArtifactAdapter.from_manifest(_manifest())
    baseline = _model_adapter(
        state_kind="baseline",
        state_id="baseline-rome-fixture",
        root=GPT2LMHeadModel(),
    )
    edited = _model_adapter(
        state_kind="edited",
        state_id="edited-rome-fixture",
        root=GPT2LMHeadModel(),
    )

    with adapter.bind_states(baseline=baseline, edited=edited) as session:
        pair = session.score_target_pair("prompt", " target")

    assert pair.baseline == pair.edited

    unsupported = _model_adapter(
        state_kind="edited",
        state_id="edited-rome-fixture",
        root=SimpleNamespace(),
    )
    with pytest.raises(AdapterCompatibilityError, match="exact GPT2LMHeadModel"):
        adapter.bind_states(baseline=baseline, edited=unsupported)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "Other", "editor.name must equal 'ROME'"),
        ("source_repository", "https://example.invalid/rome", "official ROME repository"),
    ],
)
def test_rome_importer_rejects_unverified_editor_identity(
    field: str,
    value: str,
    message: str,
) -> None:
    manifest = _manifest()
    manifest["editor"][field] = value  # type: ignore[index]

    with pytest.raises(AdapterCompatibilityError, match=message):
        RomeArtifactAdapter.from_manifest(manifest)
