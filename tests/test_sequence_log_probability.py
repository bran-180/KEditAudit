import json
import math
import re

import pytest

from kedit_audit.metrics import LogitsValidationError, target_sequence_log_probability


def test_sequence_score_retains_per_token_log_probabilities() -> None:
    logits = [
        [math.log(0.1), math.log(0.9)],
        [math.log(0.8), math.log(0.2)],
    ]

    result = target_sequence_log_probability(logits, [1, 0])

    assert result.target_token_ids == (1, 0)
    assert result.token_log_probabilities == pytest.approx((math.log(0.9), math.log(0.8)))
    assert result.sum_log_probability == pytest.approx(math.log(0.9) + math.log(0.8))
    assert result.mean_log_probability == pytest.approx((math.log(0.9) + math.log(0.8)) / 2)
    assert result.token_count == 2
    assert result.unit == "nats"


def test_result_survives_json_round_trip_without_losing_raw_values() -> None:
    result = target_sequence_log_probability([[0.0, 0.0], [0.0, 0.0]], [0, 1])

    serialized = result.as_dict()
    round_tripped = json.loads(json.dumps(serialized, sort_keys=True))

    assert round_tripped == serialized
    assert round_tripped["target_token_ids"] == [0, 1]
    assert round_tripped["token_log_probabilities"] == pytest.approx(
        [-math.log(2.0), -math.log(2.0)]
    )


def test_log_sum_exp_is_stable_for_large_logit_magnitudes() -> None:
    logits = [[1000.0, 999.0, 998.0], [-1000.0, -1001.0, -1002.0]]
    normalizer = math.log(1.0 + math.exp(-1.0) + math.exp(-2.0))

    result = target_sequence_log_probability(logits, [0, 2])

    assert result.token_log_probabilities == pytest.approx((-normalizer, -2.0 - normalizer))


def test_score_is_invariant_to_a_constant_shift_per_row() -> None:
    logits = [[0.25, -1.0, 3.0], [10.0, 4.0, -2.0]]
    shifted = [[value + 10000.0 for value in row] for row in logits]

    original = target_sequence_log_probability(logits, [2, 0])
    translated = target_sequence_log_probability(shifted, [2, 0])

    assert translated.token_log_probabilities == pytest.approx(original.token_log_probabilities)
    assert translated.mean_log_probability == pytest.approx(original.mean_log_probability)


@pytest.mark.parametrize(
    ("logits", "target_token_ids", "message"),
    [
        ([], [], "target_token_ids must contain at least one token"),
        ([[0.0, 1.0]], [0, 1], "logits row count 1 must equal target token count 2"),
        ([[]], [0], "logits[0] must contain at least one vocabulary value"),
        ([[0.0, 1.0], [0.0]], [0, 0], "logits[1] vocabulary size 1 does not match 2"),
        ([[0.0, 1.0]], [2], "target_token_ids[0]=2 is outside vocabulary range [0, 2)"),
        ([[0.0, math.nan]], [0], "logits[0][1] must be finite"),
        ([[0.0, math.inf]], [0], "logits[0][1] must be finite"),
        ([[0.0, 1.0]], [True], "target_token_ids[0] must be an integer token ID"),
        (
            [[1e308, -1e308]],
            [1],
            "logits[0][1] produces a non-finite log-probability",
        ),
    ],
)
def test_invalid_inputs_have_actionable_errors(
    logits: list[list[float]],
    target_token_ids: list[int],
    message: str,
) -> None:
    with pytest.raises(LogitsValidationError, match=rf"^{re.escape(message)}"):
        target_sequence_log_probability(logits, target_token_ids)
