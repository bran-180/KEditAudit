from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from kedit_audit.adapters import (
    BaselineContaminationError,
    ChangedTensorRecord,
    EditorAdapterMetadata,
    EditorLifecycleError,
    FakeModelAdapter,
    ModelMetadata,
    bind_editor_states,
)
from kedit_audit.metrics import SequenceLogProbability, target_sequence_log_probability


def _model_metadata(*, state_kind: str, state_id: str) -> ModelMetadata:
    return ModelMetadata(
        model_id="synthetic/editor-model",
        model_revision="model-revision-1",
        tokenizer_id="synthetic/editor-tokenizer",
        tokenizer_revision="tokenizer-revision-1",
        state_id=state_id,
        state_kind=state_kind,  # type: ignore[arg-type]
        device="cpu",
        dtype="float32",
    )


def _adapter(*, state_kind: str, state_id: str, root: object) -> FakeModelAdapter:
    return FakeModelAdapter(
        metadata=_model_metadata(state_kind=state_kind, state_id=state_id),
        token_ids_by_text={"prompt": [0], " target": [1]},
        logits_by_prompt_target={("prompt", " target"): [[0.0, 1.0]]},
        module_root=root,
    )


def _editor_metadata() -> EditorAdapterMetadata:
    return EditorAdapterMetadata(
        editor_name="synthetic-editor",
        editor_revision="a" * 40,
        source_repository="https://example.invalid/synthetic-editor",
        model_architecture="GPT2LMHeadModel",
        baseline_state_id="baseline-state",
        edited_state_id="edited-state",
        baseline_artifact_sha256="1" * 64,
        edited_artifact_sha256="2" * 64,
        hyperparameters={
            "layers": [0, 1],
            "regularization": 0.01,
            "nested": {"enabled": True},
        },
    )


def _changed_tensors() -> tuple[ChangedTensorRecord, ...]:
    return (
        ChangedTensorRecord(
            name="transformer.h.0.mlp.c_proj.weight",
            shape=(8, 8),
            dtype="float32",
            device="cpu",
            baseline_sha256="3" * 64,
            edited_sha256="4" * 64,
        ),
    )


def test_metadata_and_inventory_are_immutable_and_json_ready() -> None:
    hyperparameters = {"layers": [0, 1], "nested": {"enabled": True}}
    metadata = EditorAdapterMetadata(
        editor_name="synthetic-editor",
        editor_revision="a" * 40,
        source_repository="https://example.invalid/synthetic-editor",
        model_architecture="GPT2LMHeadModel",
        baseline_state_id="baseline-state",
        edited_state_id="edited-state",
        baseline_artifact_sha256="1" * 64,
        edited_artifact_sha256="2" * 64,
        hyperparameters=hyperparameters,
    )
    tensor = _changed_tensors()[0]
    hyperparameters["layers"].append(2)  # type: ignore[union-attr]
    serialized = {"metadata": metadata.as_dict(), "changed_tensors": [tensor.as_dict()]}

    assert metadata.as_dict()["hyperparameters"] == {
        "layers": [0, 1],
        "nested": {"enabled": True},
    }
    assert json.loads(json.dumps(serialized, sort_keys=True)) == serialized


def test_session_scores_baseline_then_edited_then_verifies_baseline() -> None:
    events: list[str] = []
    baseline = RecordingAdapter(
        metadata=_model_metadata(state_kind="baseline", state_id="baseline-state"),
        root=object(),
        event="baseline",
        events=events,
        logits=(0.0, 1.0),
    )
    edited = RecordingAdapter(
        metadata=_model_metadata(state_kind="edited", state_id="edited-state"),
        root=object(),
        event="edited",
        events=events,
        logits=(0.0, 2.0),
    )

    with bind_editor_states(
        metadata=_editor_metadata(),
        changed_tensors=_changed_tensors(),
        baseline=baseline,
        edited=edited,
    ) as session:
        pair = session.score_target_pair("prompt", " target")

    assert events == ["baseline", "edited", "baseline"]
    assert pair.baseline.mean_log_probability < pair.edited.mean_log_probability
    assert pair.as_dict()["baseline"]["target_token_ids"] == [1]
    assert session.closed
    with pytest.raises(EditorLifecycleError, match="closed"):
        session.score_target_pair("prompt", " target")


def test_same_model_root_is_rejected_before_scoring() -> None:
    shared_root = object()
    baseline = _adapter(state_kind="baseline", state_id="baseline-state", root=shared_root)
    edited = _adapter(state_kind="edited", state_id="edited-state", root=shared_root)

    with pytest.raises(EditorLifecycleError, match="distinct model roots"):
        bind_editor_states(
            metadata=_editor_metadata(),
            changed_tensors=_changed_tensors(),
            baseline=baseline,
            edited=edited,
        )


def test_edited_evaluation_that_mutates_baseline_fails_closed() -> None:
    baseline = RecordingAdapter(
        metadata=_model_metadata(state_kind="baseline", state_id="baseline-state"),
        root=object(),
        event="baseline",
        events=[],
        logits=(0.0, 1.0),
    )

    def mutate_baseline() -> None:
        baseline.logits = (0.0, -1.0)

    edited = RecordingAdapter(
        metadata=_model_metadata(state_kind="edited", state_id="edited-state"),
        root=object(),
        event="edited",
        events=[],
        logits=(0.0, 2.0),
        before_score=mutate_baseline,
    )
    session = bind_editor_states(
        metadata=_editor_metadata(),
        changed_tensors=_changed_tensors(),
        baseline=baseline,
        edited=edited,
    )

    with pytest.raises(BaselineContaminationError, match="changed during edited evaluation"):
        session.score_target_pair("prompt", " target")


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            EditorAdapterMetadata(
                editor_name="synthetic-editor",
                editor_revision="a" * 40,
                source_repository="https://example.invalid/editor",
                model_architecture="GPT2LMHeadModel",
                baseline_state_id="wrong-baseline",
                edited_state_id="edited-state",
                baseline_artifact_sha256="1" * 64,
                edited_artifact_sha256="2" * 64,
                hyperparameters={},
            ),
            "baseline state_id",
        ),
        (
            EditorAdapterMetadata(
                editor_name="synthetic-editor",
                editor_revision="a" * 40,
                source_repository="https://example.invalid/editor",
                model_architecture="GPT2LMHeadModel",
                baseline_state_id="baseline-state",
                edited_state_id="wrong-edited",
                baseline_artifact_sha256="1" * 64,
                edited_artifact_sha256="2" * 64,
                hyperparameters={},
            ),
            "edited state_id",
        ),
    ],
)
def test_manifest_state_ids_must_match_bound_adapters(
    metadata: EditorAdapterMetadata,
    message: str,
) -> None:
    with pytest.raises(EditorLifecycleError, match=message):
        bind_editor_states(
            metadata=metadata,
            changed_tensors=_changed_tensors(),
            baseline=_adapter(
                state_kind="baseline",
                state_id="baseline-state",
                root=object(),
            ),
            edited=_adapter(
                state_kind="edited",
                state_id="edited-state",
                root=object(),
            ),
        )


class RecordingAdapter:
    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        root: object,
        event: str,
        events: list[str],
        logits: tuple[float, float],
        before_score: Callable[[], None] | None = None,
    ) -> None:
        self._metadata = metadata
        self._root = root
        self.event = event
        self.events = events
        self.logits = logits
        self.before_score = before_score

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def module_root(self) -> object:
        return self._root

    def tokenize(self, text: str) -> tuple[int, ...]:
        return (0,) if text == "prompt" else (1,)

    def score_target(self, prompt: str, target: str) -> SequenceLogProbability:
        assert prompt == "prompt"
        assert target == " target"
        if self.before_score is not None:
            self.before_score()
        self.events.append(self.event)
        return target_sequence_log_probability((self.logits,), (1,))

    def subject_token_span(self, prompt: str, subject: str) -> object:
        raise NotImplementedError
