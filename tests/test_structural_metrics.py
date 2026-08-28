from __future__ import annotations

import json
import math

import pytest

from kedit_audit.adapters import ChangedTensorRecord
from kedit_audit.metrics import (
    StructuralDifferenceValidationError,
    WeightTensorPair,
    analyze_weight_differences,
)


def _record(name: str, shape: tuple[int, ...], digest: str = "a") -> ChangedTensorRecord:
    edited_digest = "b" if digest != "b" else "c"
    return ChangedTensorRecord(
        name=name,
        shape=shape,
        dtype="float32",
        device="cpu",
        baseline_sha256=digest * 64,
        edited_sha256=edited_digest * 64,
    )


def test_frobenius_aggregates_and_spectral_estimate_retain_raw_evidence() -> None:
    inventory = (
        _record("layer.matrix", (2, 2)),
        _record("layer.vector", (2,), digest="c"),
    )
    result = analyze_weight_differences(
        inventory,
        (
            WeightTensorPair(
                name="layer.vector",
                baseline_values=[0.0, 0.0],
                edited_values=[1.0, 2.0],
            ),
            WeightTensorPair(
                name="layer.matrix",
                baseline_values=[1.0, 0.0, 0.0, 1.0],
                edited_values=[4.0, 0.0, 0.0, 5.0],
            ),
        ),
        spectral_iterations=40,
    )

    assert result.status == "complete"
    assert result.coverage == 1.0
    assert result.aggregate_delta_frobenius_norm == pytest.approx(math.sqrt(30.0))
    assert result.aggregate_relative_delta_frobenius_norm == pytest.approx(
        math.sqrt(30.0) / math.sqrt(2.0)
    )
    assert [evidence.name for evidence in result.tensors] == [
        "layer.matrix",
        "layer.vector",
    ]
    matrix = result.tensors[0]
    assert matrix.delta_frobenius_norm == 5.0
    assert matrix.delta_spectral_norm_estimate == pytest.approx(4.0)
    assert matrix.spectral_iterations == 40
    assert matrix.baseline_sha256 == "a" * 64
    assert matrix.edited_sha256 == "b" * 64
    vector = result.tensors[1]
    assert vector.delta_frobenius_norm == pytest.approx(math.sqrt(5.0))
    assert vector.delta_spectral_norm_estimate is None
    assert "not rank 2" in vector.warnings[1]
    assert any("not automatically interpreted" in warning for warning in result.warnings)

    serialized = result.as_dict()
    assert json.loads(json.dumps(serialized)) == serialized
    assert "baseline_values" not in json.dumps(serialized)
    assert len(serialized["tensors"]) == 2


def test_missing_inventory_values_are_not_zero_filled() -> None:
    inventory = (
        _record("layer.present", (1,)),
        _record("layer.partial", (1,), digest="c"),
        _record("layer.unsupplied", (1,), digest="d"),
    )
    result = analyze_weight_differences(
        inventory,
        (
            WeightTensorPair("layer.present", [2.0], [5.0]),
            WeightTensorPair(
                "layer.partial",
                [2.0],
                None,
                missing_reason="edited tensor export unavailable",
            ),
        ),
    )

    assert result.status == "incomplete"
    assert result.aggregate_delta_frobenius_norm == 3.0
    assert result.evaluated_tensor_count == 1
    assert result.missing_tensor_count == 2
    assert result.coverage == pytest.approx(1 / 3)
    assert result.tensors[1].delta_frobenius_norm is None
    assert result.tensors[1].missing_reason == "edited tensor export unavailable"
    assert result.tensors[2].missing_reason == "no numeric tensor pair was supplied"
    assert "aggregates exclude 2 missing tensors" in result.warnings[0]


def test_no_numeric_pairs_produces_unavailable_not_zero() -> None:
    result = analyze_weight_differences((_record("layer.weight", (2,)),), ())

    assert result.status == "unavailable"
    assert result.aggregate_delta_frobenius_norm is None
    assert result.aggregate_relative_delta_frobenius_norm is None
    assert result.coverage == 0.0
    assert "evaluated 0 of 1" in result.warnings[0]


def test_zero_baseline_marks_relative_norm_unavailable() -> None:
    result = analyze_weight_differences(
        (_record("layer.weight", (2,)),),
        (WeightTensorPair("layer.weight", [0.0, 0.0], [1.0, 0.0]),),
    )

    assert result.aggregate_delta_frobenius_norm == 1.0
    assert result.aggregate_relative_delta_frobenius_norm is None
    assert result.tensors[0].relative_delta_frobenius_norm is None
    assert any("baseline norm is zero" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ((WeightTensorPair("other", [0.0], [1.0]),), "not present"),
        (
            (
                WeightTensorPair("layer.weight", [0.0], [1.0]),
                WeightTensorPair("layer.weight", [0.0], [1.0]),
            ),
            "outnumber",
        ),
        ((WeightTensorPair("layer.weight", [], [1.0]),), "must contain 1"),
        ((WeightTensorPair("layer.weight", [math.nan], [1.0]),), "must be finite"),
        ((WeightTensorPair("layer.weight", None, [1.0]),), "missing_reason is required"),
        (
            (WeightTensorPair("layer.weight", [0.0], [1.0], "unexpected"),),
            "must be null",
        ),
    ],
)
def test_invalid_numeric_pairs_fail_closed(
    pairs: tuple[WeightTensorPair, ...],
    message: str,
) -> None:
    with pytest.raises(StructuralDifferenceValidationError, match=message):
        analyze_weight_differences((_record("layer.weight", (1,)),), pairs)


@pytest.mark.parametrize("iterations", [True, 0, 1001, 1.5])
def test_invalid_spectral_iterations_fail_closed(iterations: object) -> None:
    with pytest.raises(StructuralDifferenceValidationError, match="spectral_iterations"):
        analyze_weight_differences(
            (_record("layer.weight", (1, 1)),),
            (WeightTensorPair("layer.weight", [0.0], [1.0]),),
            spectral_iterations=iterations,  # type: ignore[arg-type]
        )


def test_spectral_estimate_is_deterministic() -> None:
    arguments = (
        (_record("layer.weight", (2, 2)),),
        (WeightTensorPair("layer.weight", [0.0] * 4, [1.0, 2.0, 3.0, 4.0]),),
    )

    first = analyze_weight_differences(*arguments, spectral_iterations=25)
    second = analyze_weight_differences(*arguments, spectral_iterations=25)

    assert first.as_dict() == second.as_dict()
