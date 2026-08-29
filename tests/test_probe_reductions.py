import json
import math
import re

import pytest

from kedit_audit.metrics import (
    PairedProbeScore,
    ProbeReductionValidationError,
    reduce_efficacy_log_probability_deltas,
    reduce_generality_log_probability_deltas,
    reduce_locality_log_probability_drift,
    reduce_portability_log_probability_deltas,
)


def test_generality_reduces_signed_deltas_and_retains_raw_scores() -> None:
    probes = [
        PairedProbeScore("paraphrase-1", -2.0, -1.0),
        PairedProbeScore("paraphrase-2", -1.5, -1.75),
        PairedProbeScore("paraphrase-3", -3.0, -2.5),
    ]

    result = reduce_generality_log_probability_deltas(probes)

    assert result.metric_id == "generality.mean_target_log_probability_delta"
    assert result.direction == "higher-is-better"
    assert result.aggregate == pytest.approx((1.0 - 0.25 + 0.5) / 3)
    assert result.total_probe_count == 3
    assert result.evaluated_probe_count == 3
    assert result.missing_probe_count == 0
    assert result.coverage == 1.0
    assert result.warnings == ()
    assert result.probes[0].baseline_mean_log_probability == -2.0
    assert result.probes[0].edited_mean_log_probability == -1.0
    assert result.probes[0].signed_delta == 1.0
    assert result.probes[0].absolute_delta == 1.0
    assert result.probes[0].contribution == 1.0


def test_locality_reduces_absolute_drift_but_retains_signed_deltas() -> None:
    probes = [
        PairedProbeScore("locality-1", -1.0, -1.1),
        PairedProbeScore("locality-2", -2.0, -1.6),
    ]

    result = reduce_locality_log_probability_drift(probes)

    assert result.metric_id == "locality.mean_absolute_target_log_probability_delta"
    assert result.direction == "lower-is-better"
    assert result.aggregate == pytest.approx(0.25)
    assert result.probes[0].signed_delta == pytest.approx(-0.1)
    assert result.probes[0].absolute_delta == pytest.approx(0.1)
    assert result.probes[0].contribution == pytest.approx(0.1)
    assert result.probes[1].signed_delta == pytest.approx(0.4)
    assert result.probes[1].contribution == pytest.approx(0.4)


def test_exact_and_portability_reducers_use_explicit_metric_ids() -> None:
    probe = [PairedProbeScore("probe-1", -2.0, -1.25)]

    efficacy = reduce_efficacy_log_probability_deltas(probe)
    portability = reduce_portability_log_probability_deltas(probe)

    assert efficacy.metric_id == "efficacy.mean_target_log_probability_delta"
    assert efficacy.aggregate == 0.75
    assert portability.metric_id == (
        "portability.mean_expected_target_log_probability_delta"
    )
    assert portability.aggregate == 0.75


def test_partial_coverage_is_explicit_and_survives_json_round_trip() -> None:
    probes = [
        PairedProbeScore("paraphrase-1", -2.0, -1.0),
        PairedProbeScore(
            "paraphrase-2",
            None,
            -0.5,
            "baseline evaluation timed out",
        ),
        PairedProbeScore(
            "paraphrase-3",
            None,
            None,
            "both evaluations unavailable",
        ),
    ]

    result = reduce_generality_log_probability_deltas(probes)
    serialized = result.as_dict()
    round_tripped = json.loads(json.dumps(serialized, sort_keys=True))

    assert result.aggregate == 1.0
    assert result.evaluated_probe_count == 1
    assert result.missing_probe_count == 2
    assert result.coverage == pytest.approx(1 / 3)
    assert result.warnings == (
        "partial coverage: evaluated 1 of 3 probes; aggregate excludes 2 missing probes",
    )
    assert result.probes[1].baseline_mean_log_probability is None
    assert result.probes[1].edited_mean_log_probability == -0.5
    assert result.probes[1].signed_delta is None
    assert result.probes[1].contribution is None
    assert result.probes[1].missing_reason == "baseline evaluation timed out"
    assert round_tripped == serialized


def test_all_missing_probes_produce_no_aggregate() -> None:
    result = reduce_locality_log_probability_drift(
        [
            PairedProbeScore("locality-1", None, None, "evaluation failed"),
            PairedProbeScore("locality-2", None, -1.0, "baseline missing"),
        ]
    )

    assert result.aggregate is None
    assert result.evaluated_probe_count == 0
    assert result.missing_probe_count == 2
    assert result.coverage == 0.0
    assert result.warnings == (
        "aggregate unavailable: evaluated 0 of 2 probes because every pair is incomplete",
    )


@pytest.mark.parametrize(
    ("probes", "message"),
    [
        ([], "probes must contain at least one probe"),
        (
            [PairedProbeScore("same-id", -1.0, -0.5), PairedProbeScore("same-id", -2.0, -1.0)],
            "probes[1].probe_id 'same-id' duplicates probes[0].probe_id",
        ),
        (
            [PairedProbeScore("missing-reason", None, -1.0)],
            "probes[0].missing_reason is required when either score is missing",
        ),
        (
            [PairedProbeScore("contradictory", -2.0, -1.0, "not actually missing")],
            "probes[0].missing_reason must be null when both scores are present",
        ),
        (
            [PairedProbeScore("blank-reason", None, -1.0, "   ")],
            "probes[0].missing_reason must not be blank",
        ),
        (
            [PairedProbeScore("non-finite", -1.0, math.nan)],
            "probes[0].edited_mean_log_probability must be finite",
        ),
        (
            [PairedProbeScore("positive", -1.0, 0.1)],
            "probes[0].edited_mean_log_probability must not be positive",
        ),
    ],
)
def test_invalid_probe_inputs_have_actionable_errors(
    probes: list[PairedProbeScore],
    message: str,
) -> None:
    with pytest.raises(ProbeReductionValidationError, match=rf"^{re.escape(message)}"):
        reduce_generality_log_probability_deltas(probes)
