import json
import math
from collections import OrderedDict
from dataclasses import dataclass

import pytest

from kedit_audit.adapters import FakeModelAdapter, ModelMetadata, ModulePathError, TokenSpan
from kedit_audit.causal import (
    CAUSAL_TRACE_RESULT_VERSION,
    CausalTraceRequest,
    CleanTraceRun,
    TraceValidationError,
    run_causal_trace,
)


class ModuleNode:
    def __init__(self, **children: object) -> None:
        self._modules = OrderedDict(children)


@dataclass(frozen=True)
class FakeCorruptionTensor:
    seed: int
    subject_span: TokenSpan
    values: tuple[float, ...]


class RecordingTraceAdapter(FakeModelAdapter):
    def __init__(
        self,
        *,
        clean_score: float = -1.0,
        corrupted_score: float = -3.0,
        restored_scores: dict[str, float] | None = None,
    ) -> None:
        layer_zero = object()
        layer_one = object()
        super().__init__(
            metadata=ModelMetadata(
                model_id="synthetic/trace-model",
                model_revision="revision-1",
                tokenizer_id="synthetic/trace-tokenizer",
                tokenizer_revision="revision-1",
                state_id="baseline-trace-state",
                state_kind="baseline",
                device="cpu",
                dtype="float32",
            ),
            token_ids_by_text={
                "The Eiffel Tower is in Paris": [0, 1, 2, 3, 4, 5],
                "Eiffel Tower": [1, 2],
                " Italy": [0],
            },
            logits_by_prompt_target={
                ("The Eiffel Tower is in Paris", " Italy"): [[0.0]],
            },
            module_root=ModuleNode(layers=ModuleNode(**{"0": layer_zero, "1": layer_one})),
        )
        self.clean_score = clean_score
        self.corrupted_score = corrupted_score
        self.restored_scores = restored_scores or {"layers.0": -2.0, "layers.1": -1.5}
        self.clean_activations = {"layers.0": object(), "layers.1": object()}
        self.created_corruptions: list[FakeCorruptionTensor] = []
        self.used_corruptions: list[FakeCorruptionTensor] = []
        self.restored_activations: list[tuple[str, object]] = []
        self.recorded_spans: list[TokenSpan] = []
        self.clean_run_count = 0

    def run_clean(
        self,
        *,
        prompt: str,
        target: str,
        module_paths: tuple[str, ...],
    ) -> CleanTraceRun:
        assert prompt == "The Eiffel Tower is in Paris"
        assert target == " Italy"
        self.clean_run_count += 1
        return CleanTraceRun(
            target_score=self.clean_score,
            activations={path: self.clean_activations[path] for path in module_paths if path in self.clean_activations},
        )

    def create_corruption(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        subject_span: TokenSpan,
        seed: int,
    ) -> FakeCorruptionTensor:
        self.recorded_spans.append(subject_span)
        tensor = FakeCorruptionTensor(
            seed=seed,
            subject_span=subject_span,
            values=tuple(((seed + token_id * 17) % 101) / 100 for token_id in prompt_token_ids),
        )
        self.created_corruptions.append(tensor)
        return tensor

    def run_corrupted(
        self,
        *,
        prompt: str,
        target: str,
        corruption: object,
    ) -> float:
        assert prompt == "The Eiffel Tower is in Paris"
        assert target == " Italy"
        assert isinstance(corruption, FakeCorruptionTensor)
        self.used_corruptions.append(corruption)
        return self.corrupted_score

    def run_restored(
        self,
        *,
        prompt: str,
        target: str,
        corruption: object,
        module_path: str,
        clean_activation: object,
    ) -> float:
        assert prompt == "The Eiffel Tower is in Paris"
        assert target == " Italy"
        assert isinstance(corruption, FakeCorruptionTensor)
        self.used_corruptions.append(corruption)
        self.restored_activations.append((module_path, clean_activation))
        return self.restored_scores[module_path]


def _request(*, module_paths: tuple[str, ...] = ("layers.0", "layers.1")) -> CausalTraceRequest:
    return CausalTraceRequest(
        prompt="The Eiffel Tower is in Paris",
        subject="Eiffel Tower",
        target=" Italy",
        module_paths=module_paths,
        seed=17,
    )


def test_trace_reuses_one_corruption_object_for_every_paired_run() -> None:
    adapter = RecordingTraceAdapter()

    result = run_causal_trace(adapter, _request())

    assert len(adapter.created_corruptions) == 1
    assert len(adapter.used_corruptions) == 3
    assert all(
        corruption is adapter.created_corruptions[0]
        for corruption in adapter.used_corruptions
    )
    assert adapter.recorded_spans == [TokenSpan(1, 3)]
    assert result.subject_token_span == TokenSpan(1, 3)
    assert result.clean_target_score == -1.0
    assert result.corrupted_target_score == -3.0
    assert result.modules[0].module_path == "layers.0"
    assert result.modules[0].restored_target_score == -2.0
    assert result.modules[0].restoration_delta == 1.0
    assert result.modules[0].recovery_fraction == 0.5
    assert result.modules[1].recovery_fraction == 0.75
    assert adapter.restored_activations == [
        ("layers.0", adapter.clean_activations["layers.0"]),
        ("layers.1", adapter.clean_activations["layers.1"]),
    ]


def test_trace_result_is_deterministic_and_json_round_trippable() -> None:
    first_adapter = RecordingTraceAdapter()
    second_adapter = RecordingTraceAdapter()

    first = run_causal_trace(first_adapter, _request()).as_dict()
    second = run_causal_trace(second_adapter, _request()).as_dict()
    round_tripped = json.loads(json.dumps(first, sort_keys=True))

    assert first == second == round_tripped
    assert first["schema_version"] == CAUSAL_TRACE_RESULT_VERSION
    assert first_adapter.created_corruptions[0].values == second_adapter.created_corruptions[0].values
    assert first["modules"][0] == {
        "module_path": "layers.0",
        "clean_target_score": -1.0,
        "corrupted_target_score": -3.0,
        "restored_target_score": -2.0,
        "restoration_delta": 1.0,
        "recovery_fraction": 0.5,
    }


def test_missing_clean_activation_fails_before_corruption_is_created() -> None:
    adapter = RecordingTraceAdapter()
    del adapter.clean_activations["layers.1"]

    with pytest.raises(TraceValidationError, match="missing clean activations.*layers.1"):
        run_causal_trace(adapter, _request())

    assert adapter.created_corruptions == []
    assert adapter.used_corruptions == []


def test_module_paths_are_resolved_before_model_execution() -> None:
    adapter = RecordingTraceAdapter()

    with pytest.raises(ModulePathError, match="layers.2"):
        run_causal_trace(adapter, _request(module_paths=("layers.2",)))

    assert adapter.clean_run_count == 0
    assert adapter.created_corruptions == []


def test_request_rejects_duplicate_paths_and_invalid_seed() -> None:
    with pytest.raises(ValueError, match="module_paths must be unique"):
        _request(module_paths=("layers.0", "layers.0"))

    with pytest.raises(ValueError, match="seed"):
        CausalTraceRequest(
            prompt="prompt",
            subject="subject",
            target="target",
            module_paths=("layers.0",),
            seed=-1,
        )


def test_non_finite_adapter_score_is_rejected_with_stage() -> None:
    adapter = RecordingTraceAdapter(corrupted_score=math.nan)

    with pytest.raises(TraceValidationError, match="corrupted target score must be finite"):
        run_causal_trace(adapter, _request())


def test_zero_clean_corrupted_gap_emits_warning_without_division() -> None:
    adapter = RecordingTraceAdapter(
        clean_score=-1.0,
        corrupted_score=-1.0,
        restored_scores={"layers.0": -0.5},
    )

    result = run_causal_trace(adapter, _request(module_paths=("layers.0",)))

    assert result.modules[0].restoration_delta == 0.5
    assert result.modules[0].recovery_fraction is None
    assert result.warnings == (
        "recovery fraction unavailable for layers.0 because clean and corrupted scores are equal",
    )
