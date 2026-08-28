import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kedit_audit.adapters import (
    AdapterCompatibilityError,
    EasyEditArtifactAdapter,
    FakeModelAdapter,
    ModelMetadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "editor_artifacts" / "valid" / "easyedit.json"
)


class GPT2LMHeadModel:
    __module__ = "transformers.models.gpt2.modeling_gpt2"


def _manifest() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _model_adapter(*, state_kind: str, state_id: str, root: object) -> FakeModelAdapter:
    return FakeModelAdapter(
        metadata=ModelMetadata(
            model_id="synthetic/easyedit-model",
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


def test_easyedit_importer_retains_revision_hyperparameters_and_weight_inventory() -> None:
    adapter = EasyEditArtifactAdapter.from_manifest(_manifest())
    serialized = adapter.as_dict()

    assert adapter.metadata.editor_name == "EasyEdit"
    assert adapter.metadata.editor_revision == "14cea8245f06715684592ab55184939b99d70784"
    assert adapter.metadata.source_repository == "https://github.com/zjunlp/EasyEdit"
    assert adapter.metadata.hyperparameters["algorithm"] == "ROME"
    assert adapter.metadata.hyperparameters["return_orig_weights"] is True
    assert len(adapter.changed_tensors) == 1
    assert serialized["artifact_kind"] == "synthetic-contract-fixture"
    assert serialized["changed_tensors"][0]["baseline_sha256"] == "7" * 64


def test_easyedit_importer_binds_only_exact_supported_gpt2_roots() -> None:
    adapter = EasyEditArtifactAdapter.from_manifest(_manifest())
    baseline = _model_adapter(
        state_kind="baseline",
        state_id="baseline-easyedit-fixture",
        root=GPT2LMHeadModel(),
    )
    edited = _model_adapter(
        state_kind="edited",
        state_id="edited-easyedit-fixture",
        root=GPT2LMHeadModel(),
    )

    with adapter.bind_states(baseline=baseline, edited=edited) as session:
        pair = session.score_target_pair("prompt", " target")

    assert pair.baseline == pair.edited

    unsupported = _model_adapter(
        state_kind="edited",
        state_id="edited-easyedit-fixture",
        root=SimpleNamespace(),
    )
    with pytest.raises(AdapterCompatibilityError, match="exact GPT2LMHeadModel"):
        adapter.bind_states(baseline=baseline, edited=unsupported)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["editor"].update({"name": "Other"}),
            "editor.name must equal 'EasyEdit'",
        ),
        (
            lambda manifest: manifest["editor"].update(
                {"source_repository": "https://example.invalid/EasyEdit"}
            ),
            "official EasyEdit repository",
        ),
        (
            lambda manifest: manifest["editor"]["hyperparameters"].update(
                {"return_orig_weights": False}
            ),
            "return_orig_weights must be true",
        ),
        (
            lambda manifest: manifest.update({"changed_tensors": []}),
            "at least one changed tensor",
        ),
    ],
)
def test_easyedit_importer_rejects_unverified_or_non_restorable_exports(
    mutation: object,
    message: str,
) -> None:
    manifest = _manifest()
    mutation(manifest)  # type: ignore[operator]

    with pytest.raises(AdapterCompatibilityError, match=message):
        EasyEditArtifactAdapter.from_manifest(manifest)
