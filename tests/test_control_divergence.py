import json
import math
import re

import pytest

from kedit_audit.metrics import (
    ControlDivergenceValidationError,
    PairedControlLogits,
    reduce_control_kl_divergence,
)


def test_control_kl_retains_every_position_and_reduces_probe_means() -> None:
    probes = [
        PairedControlLogits(
            "control-1",
            baseline_logits=((0.0, 0.0), (0.0, 0.0)),
            edited_logits=(
                (math.log(0.75), math.log(0.25)),
                (math.log(0.5), math.log(0.5)),
            ),
        ),
        PairedControlLogits(
            "control-2",
            baseline_logits=((math.log(0.9), math.log(0.1)),),
            edited_logits=((math.log(0.8), math.log(0.2)),),
        ),
    ]

    result = reduce_control_kl_divergence(probes)

    expected_first_position = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    expected_second_probe = 0.9 * math.log(0.9 / 0.8) + 0.1 * math.log(0.1 / 0.2)
    assert result.metric_id == "control.mean_kl_divergence"
    assert result.direction == "lower-is-better"
    assert result.unit == "nats"
    assert result.temperature == 1.0
    assert result.total_probe_count == 2
    assert result.evaluated_probe_count == 2
    assert result.total_position_count == 3
    assert result.evaluated_position_count == 3
    assert result.coverage == 1.0
    assert result.probes[0].position_kl_divergences == pytest.approx(
        (expected_first_position, 0.0)
    )
    assert result.probes[0].mean_kl_divergence == pytest.approx(
        expected_first_position / 2
    )
    assert result.probes[1].mean_kl_divergence == pytest.approx(expected_second_probe)
    assert result.aggregate == pytest.approx(
        (expected_first_position / 2 + expected_second_probe) / 2
    )


def test_kl_direction_is_baseline_to_edited_and_temperature_is_recorded() -> None:
    forward = reduce_control_kl_divergence(
        [
            PairedControlLogits(
                "control",
                baseline_logits=((2.0, 0.0),),
                edited_logits=((0.0, 0.0),),
            )
        ],
        temperature=2.0,
    )
    reverse = reduce_control_kl_divergence(
        [
            PairedControlLogits(
                "control",
                baseline_logits=((0.0, 0.0),),
                edited_logits=((2.0, 0.0),),
            )
        ],
        temperature=2.0,
    )

    assert forward.temperature == 2.0
    assert forward.aggregate != pytest.approx(reverse.aggregate)


def test_missing_control_probe_is_explicit_and_json_round_trippable() -> None:
    result = reduce_control_kl_divergence(
        [
            PairedControlLogits(
                "available",
                baseline_logits=((0.0, 0.0),),
                edited_logits=((0.0, 0.0),),
            ),
            PairedControlLogits(
                "missing",
                baseline_logits=None,
                edited_logits=((0.0, 0.0),),
                missing_reason="baseline evaluation failed",
            ),
        ]
    )
    serialized = result.as_dict()

    assert result.aggregate == 0.0
    assert result.total_probe_count == 2
    assert result.evaluated_probe_count == 1
    assert result.total_position_count == 2
    assert result.evaluated_position_count == 1
    assert result.coverage == 0.5
    assert result.probes[1].position_kl_divergences is None
    assert result.probes[1].missing_reason == "baseline evaluation failed"
    assert result.warnings == (
        "partial coverage: evaluated 1 of 2 control probes; aggregate excludes 1 missing probe",
    )
    assert json.loads(json.dumps(serialized, sort_keys=True)) == serialized


def test_extreme_finite_logits_are_reduced_stably() -> None:
    result = reduce_control_kl_divergence(
        [
            PairedControlLogits(
                "extreme",
                baseline_logits=((1000.0, -1000.0),),
                edited_logits=((-1000.0, 1000.0),),
            )
        ]
    )

    assert result.aggregate == pytest.approx(2000.0)
    assert math.isfinite(result.aggregate)


@pytest.mark.parametrize(
    ("probes", "temperature", "message"),
    [
        ([], 1.0, "probes must contain at least one control probe"),
        (
            [PairedControlLogits("same", ((0.0, 0.0),), ((0.0, 0.0),)),
             PairedControlLogits("same", ((0.0, 0.0),), ((0.0, 0.0),))],
            1.0,
            "probes[1].probe_id 'same' duplicates probes[0].probe_id",
        ),
        (
            [PairedControlLogits("missing", None, ((0.0, 0.0),))],
            1.0,
            "probes[0].missing_reason is required when either logits value is missing",
        ),
        (
            [PairedControlLogits("positions", ((0.0, 0.0),), ((0.0, 0.0), (1.0, 0.0)))],
            1.0,
            "probes[0] baseline and edited position counts must match",
        ),
        (
            [PairedControlLogits("vocab", ((0.0, 0.0),), ((0.0, 0.0, 0.0),))],
            1.0,
            "probes[0].edited_logits[0] vocabulary size 3 must match 2",
        ),
        (
            [PairedControlLogits("nan", ((0.0, math.nan),), ((0.0, 0.0),))],
            1.0,
            "probes[0].baseline_logits[0][1] must be finite",
        ),
        (
            [PairedControlLogits("valid", ((0.0, 0.0),), ((0.0, 0.0),))],
            0.0,
            "temperature must be a finite number greater than zero",
        ),
    ],
)
def test_invalid_control_inputs_fail_with_actionable_paths(
    probes: list[PairedControlLogits],
    temperature: float,
    message: str,
) -> None:
    with pytest.raises(ControlDivergenceValidationError, match=rf"^{re.escape(message)}"):
        reduce_control_kl_divergence(probes, temperature=temperature)
