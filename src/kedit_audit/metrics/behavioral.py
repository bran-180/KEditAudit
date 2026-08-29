"""Behavioral metrics computed without loading a model."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, SupportsFloat, cast

MetricId = Literal[
    "efficacy.mean_target_log_probability_delta",
    "generality.mean_target_log_probability_delta",
    "locality.mean_absolute_target_log_probability_delta",
    "portability.mean_expected_target_log_probability_delta",
]
MetricDirection = Literal["higher-is-better", "lower-is-better"]


class LogitsValidationError(ValueError):
    """Raised when supplied logits and token IDs cannot define a sequence score."""


class ProbeReductionValidationError(ValueError):
    """Raised when paired probe scores cannot define a reduction."""


@dataclass(frozen=True)
class SequenceLogProbability:
    """Raw and reduced conditional log-probabilities for one target sequence."""

    target_token_ids: tuple[int, ...]
    token_log_probabilities: tuple[float, ...]
    sum_log_probability: float
    mean_log_probability: float
    unit: Literal["nats"] = "nats"

    @property
    def token_count(self) -> int:
        """Return the number of scored target tokens."""

        return len(self.target_token_ids)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record that retains every token score."""

        return {
            "target_token_ids": list(self.target_token_ids),
            "token_log_probabilities": list(self.token_log_probabilities),
            "sum_log_probability": self.sum_log_probability,
            "mean_log_probability": self.mean_log_probability,
            "token_count": self.token_count,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class PairedProbeScore:
    """Baseline and edited mean target log-probabilities for one stable probe ID."""

    probe_id: str
    baseline_mean_log_probability: float | None
    edited_mean_log_probability: float | None
    missing_reason: str | None = None


@dataclass(frozen=True)
class ProbeScoreEvidence:
    """Raw paired scores and the exact value contributed to an aggregate."""

    probe_id: str
    baseline_mean_log_probability: float | None
    edited_mean_log_probability: float | None
    signed_delta: float | None
    absolute_delta: float | None
    contribution: float | None
    missing_reason: str | None

    def as_dict(self) -> dict[str, object]:
        """Return this per-probe evidence as a JSON-serializable object."""

        return {
            "probe_id": self.probe_id,
            "baseline_mean_log_probability": self.baseline_mean_log_probability,
            "edited_mean_log_probability": self.edited_mean_log_probability,
            "signed_delta": self.signed_delta,
            "absolute_delta": self.absolute_delta,
            "contribution": self.contribution,
            "missing_reason": self.missing_reason,
        }


@dataclass(frozen=True)
class ProbeScoreReduction:
    """A coverage-aware aggregate that retains all supplied probe evidence."""

    metric_id: MetricId
    direction: MetricDirection
    aggregate: float | None
    total_probe_count: int
    evaluated_probe_count: int
    coverage: float
    probes: tuple[ProbeScoreEvidence, ...]
    warnings: tuple[str, ...]
    reduction: Literal["arithmetic_mean_over_available_probes"] = (
        "arithmetic_mean_over_available_probes"
    )
    unit: Literal["nats"] = "nats"

    @property
    def missing_probe_count(self) -> int:
        """Return how many probe pairs did not contribute to the aggregate."""

        return self.total_probe_count - self.evaluated_probe_count

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result without discarding raw probes."""

        return {
            "metric_id": self.metric_id,
            "direction": self.direction,
            "aggregate": self.aggregate,
            "reduction": self.reduction,
            "unit": self.unit,
            "total_probe_count": self.total_probe_count,
            "evaluated_probe_count": self.evaluated_probe_count,
            "missing_probe_count": self.missing_probe_count,
            "coverage": self.coverage,
            "probes": [probe.as_dict() for probe in self.probes],
            "warnings": list(self.warnings),
        }


def target_sequence_log_probability(
    logits: Sequence[Sequence[float]],
    target_token_ids: Sequence[int],
) -> SequenceLogProbability:
    """Score aligned target tokens using stable log-softmax, in natural-log units.

    Row ``i`` must contain the next-token logits for ``target_token_ids[i]``
    conditioned on the prompt and preceding target tokens. Tokenization and
    causal-position alignment are intentionally outside this reducer.
    """

    normalized_targets = _validate_target_token_ids(target_token_ids)
    if len(logits) != len(normalized_targets):
        raise LogitsValidationError(
            f"logits row count {len(logits)} must equal target token count "
            f"{len(normalized_targets)}"
        )

    normalized_logits = _validate_logits(logits)
    vocabulary_size = len(normalized_logits[0])
    for index, token_id in enumerate(normalized_targets):
        if token_id < 0 or token_id >= vocabulary_size:
            raise LogitsValidationError(
                f"target_token_ids[{index}]={token_id} is outside vocabulary range "
                f"[0, {vocabulary_size})"
            )

    token_log_probabilities_list: list[float] = []
    for row_index, (row, token_id) in enumerate(
        zip(normalized_logits, normalized_targets, strict=True)
    ):
        token_log_probability = _selected_log_softmax(row, token_id)
        if not math.isfinite(token_log_probability):
            raise LogitsValidationError(
                f"logits[{row_index}][{token_id}] produces a non-finite log-probability"
            )
        token_log_probabilities_list.append(token_log_probability)
    token_log_probabilities = tuple(token_log_probabilities_list)

    try:
        sum_log_probability = math.fsum(token_log_probabilities)
    except OverflowError as error:
        raise LogitsValidationError(
            "sum_log_probability is outside the finite float range"
        ) from error
    if not math.isfinite(sum_log_probability):
        raise LogitsValidationError("sum_log_probability is outside the finite float range")
    return SequenceLogProbability(
        target_token_ids=normalized_targets,
        token_log_probabilities=token_log_probabilities,
        sum_log_probability=sum_log_probability,
        mean_log_probability=sum_log_probability / len(token_log_probabilities),
    )


def reduce_generality_log_probability_deltas(
    probes: Sequence[PairedProbeScore],
) -> ProbeScoreReduction:
    """Average edited-minus-baseline target scores over available paraphrase probes."""

    return _reduce_probe_scores(
        probes,
        metric_id="generality.mean_target_log_probability_delta",
        direction="higher-is-better",
        absolute_contribution=False,
    )


def reduce_efficacy_log_probability_deltas(
    probes: Sequence[PairedProbeScore],
) -> ProbeScoreReduction:
    """Average edited-minus-baseline target scores over exact edit probes."""

    return _reduce_probe_scores(
        probes,
        metric_id="efficacy.mean_target_log_probability_delta",
        direction="higher-is-better",
        absolute_contribution=False,
    )


def reduce_locality_log_probability_drift(
    probes: Sequence[PairedProbeScore],
) -> ProbeScoreReduction:
    """Average absolute target-score drift over available locality probes."""

    return _reduce_probe_scores(
        probes,
        metric_id="locality.mean_absolute_target_log_probability_delta",
        direction="lower-is-better",
        absolute_contribution=True,
    )


def reduce_portability_log_probability_deltas(
    probes: Sequence[PairedProbeScore],
) -> ProbeScoreReduction:
    """Average edited-minus-baseline expected-target scores over portability probes."""

    return _reduce_probe_scores(
        probes,
        metric_id="portability.mean_expected_target_log_probability_delta",
        direction="higher-is-better",
        absolute_contribution=False,
    )


def _reduce_probe_scores(
    probes: Sequence[PairedProbeScore],
    *,
    metric_id: MetricId,
    direction: MetricDirection,
    absolute_contribution: bool,
) -> ProbeScoreReduction:
    if len(probes) == 0:
        raise ProbeReductionValidationError("probes must contain at least one probe")

    evidence: list[ProbeScoreEvidence] = []
    contributions: list[float] = []
    first_index_by_id: dict[str, int] = {}
    for index, probe in enumerate(probes):
        if not isinstance(probe, PairedProbeScore):
            raise ProbeReductionValidationError(
                f"probes[{index}] must be a PairedProbeScore instance"
            )
        probe_id = _validate_probe_id(probe.probe_id, index)
        first_index = first_index_by_id.setdefault(probe_id, index)
        if first_index != index:
            raise ProbeReductionValidationError(
                f"probes[{index}].probe_id {probe_id!r} duplicates probes[{first_index}].probe_id"
            )

        baseline = _validate_log_probability(
            probe.baseline_mean_log_probability,
            f"probes[{index}].baseline_mean_log_probability",
        )
        edited = _validate_log_probability(
            probe.edited_mean_log_probability,
            f"probes[{index}].edited_mean_log_probability",
        )
        missing_reason = _validate_missing_reason(
            probe.missing_reason,
            pair_is_complete=baseline is not None and edited is not None,
            index=index,
        )

        if baseline is None or edited is None:
            evidence.append(
                ProbeScoreEvidence(
                    probe_id=probe_id,
                    baseline_mean_log_probability=baseline,
                    edited_mean_log_probability=edited,
                    signed_delta=None,
                    absolute_delta=None,
                    contribution=None,
                    missing_reason=missing_reason,
                )
            )
            continue

        signed_delta = edited - baseline
        if not math.isfinite(signed_delta):
            raise ProbeReductionValidationError(
                f"probes[{index}] score delta is outside the finite float range"
            )
        absolute_delta = abs(signed_delta)
        contribution = absolute_delta if absolute_contribution else signed_delta
        contributions.append(contribution)
        evidence.append(
            ProbeScoreEvidence(
                probe_id=probe_id,
                baseline_mean_log_probability=baseline,
                edited_mean_log_probability=edited,
                signed_delta=signed_delta,
                absolute_delta=absolute_delta,
                contribution=contribution,
                missing_reason=None,
            )
        )

    aggregate = _mean_contribution(contributions)
    total_probe_count = len(evidence)
    evaluated_probe_count = len(contributions)
    warnings = _coverage_warnings(evaluated_probe_count, total_probe_count)
    return ProbeScoreReduction(
        metric_id=metric_id,
        direction=direction,
        aggregate=aggregate,
        total_probe_count=total_probe_count,
        evaluated_probe_count=evaluated_probe_count,
        coverage=evaluated_probe_count / total_probe_count,
        probes=tuple(evidence),
        warnings=warnings,
    )


def _validate_probe_id(probe_id: object, index: int) -> str:
    if (
        not isinstance(probe_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", probe_id) is None
    ):
        raise ProbeReductionValidationError(
            f"probes[{index}].probe_id must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}"
        )
    return probe_id


def _validate_log_probability(value: object, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes)):
        raise ProbeReductionValidationError(f"{path} must be a real number or null")
    try:
        normalized = float(cast(SupportsFloat, value))
    except (TypeError, ValueError) as error:
        raise ProbeReductionValidationError(f"{path} must be a real number or null") from error
    if not math.isfinite(normalized):
        raise ProbeReductionValidationError(f"{path} must be finite")
    if normalized > 0.0:
        raise ProbeReductionValidationError(f"{path} must not be positive")
    return normalized


def _validate_missing_reason(
    value: object,
    *,
    pair_is_complete: bool,
    index: int,
) -> str | None:
    path = f"probes[{index}].missing_reason"
    if pair_is_complete:
        if value is not None:
            raise ProbeReductionValidationError(f"{path} must be null when both scores are present")
        return None
    if value is None:
        raise ProbeReductionValidationError(f"{path} is required when either score is missing")
    if not isinstance(value, str):
        raise ProbeReductionValidationError(f"{path} must be a string")
    if not value.strip():
        raise ProbeReductionValidationError(f"{path} must not be blank")
    if len(value) > 4096:
        raise ProbeReductionValidationError(f"{path} must not exceed 4096 characters")
    return value


def _mean_contribution(contributions: Sequence[float]) -> float | None:
    if len(contributions) == 0:
        return None
    try:
        total = math.fsum(contributions)
    except OverflowError as error:
        raise ProbeReductionValidationError(
            "aggregate is outside the finite float range"
        ) from error
    aggregate = total / len(contributions)
    if not math.isfinite(aggregate):
        raise ProbeReductionValidationError("aggregate is outside the finite float range")
    return aggregate


def _coverage_warnings(evaluated: int, total: int) -> tuple[str, ...]:
    missing = total - evaluated
    if missing == 0:
        return ()
    if evaluated == 0:
        return (
            f"aggregate unavailable: evaluated 0 of {total} probes because every pair is incomplete",
        )
    return (
        (
            f"partial coverage: evaluated {evaluated} of {total} probes; "
            f"aggregate excludes {missing} missing probes"
        ),
    )


def _validate_target_token_ids(target_token_ids: Sequence[int]) -> tuple[int, ...]:
    if len(target_token_ids) == 0:
        raise LogitsValidationError("target_token_ids must contain at least one token")

    normalized: list[int] = []
    for index, token_id in enumerate(target_token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise LogitsValidationError(f"target_token_ids[{index}] must be an integer token ID")
        normalized.append(int(token_id))
    return tuple(normalized)


def _validate_logits(logits: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    normalized: list[tuple[float, ...]] = []
    vocabulary_size: int | None = None
    for row_index, row in enumerate(logits):
        if len(row) == 0:
            raise LogitsValidationError(
                f"logits[{row_index}] must contain at least one vocabulary value"
            )
        if vocabulary_size is None:
            vocabulary_size = len(row)
        elif len(row) != vocabulary_size:
            raise LogitsValidationError(
                f"logits[{row_index}] vocabulary size {len(row)} does not match {vocabulary_size}"
            )

        normalized_row: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, (bool, str, bytes)):
                raise LogitsValidationError(
                    f"logits[{row_index}][{column_index}] must be a real number"
                )
            try:
                normalized_value = float(value)
            except (TypeError, ValueError) as error:
                raise LogitsValidationError(
                    f"logits[{row_index}][{column_index}] must be a real number"
                ) from error
            if not math.isfinite(normalized_value):
                raise LogitsValidationError(f"logits[{row_index}][{column_index}] must be finite")
            normalized_row.append(normalized_value)
        normalized.append(tuple(normalized_row))

    return tuple(normalized)


def _selected_log_softmax(logits: tuple[float, ...], token_id: int) -> float:
    maximum = max(logits)
    shifted_exponential_sum = math.fsum(math.exp(value - maximum) for value in logits)
    return logits[token_id] - maximum - math.log(shifted_exponential_sum)
