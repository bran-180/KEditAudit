"""Distributional drift metrics over caller-supplied logits."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, SupportsFloat, cast

_MAX_PROBES = 10_000
_MAX_POSITIONS_PER_PROBE = 4_096
_MAX_VOCABULARY_SIZE = 65_536
_MAX_LOGIT_VALUES = 10_000_000


class ControlDivergenceValidationError(ValueError):
    """Raised when paired control logits cannot define a bounded KL reduction."""


@dataclass(frozen=True)
class PairedControlLogits:
    """Baseline and edited logits for aligned positions in one control probe."""

    probe_id: str
    baseline_logits: Sequence[Sequence[float]] | None
    edited_logits: Sequence[Sequence[float]] | None
    missing_reason: str | None = None


@dataclass(frozen=True)
class ControlProbeEvidence:
    """Per-position KL evidence retained for one control probe."""

    probe_id: str
    position_kl_divergences: tuple[float, ...] | None
    mean_kl_divergence: float | None
    position_count: int
    vocabulary_size: int | None
    missing_reason: str | None

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready evidence without retaining vocabulary-sized logits."""

        return {
            "probe_id": self.probe_id,
            "position_kl_divergences": (
                list(self.position_kl_divergences)
                if self.position_kl_divergences is not None
                else None
            ),
            "mean_kl_divergence": self.mean_kl_divergence,
            "position_count": self.position_count,
            "vocabulary_size": self.vocabulary_size,
            "missing_reason": self.missing_reason,
        }


@dataclass(frozen=True)
class ControlDivergenceReduction:
    """Coverage-aware baseline-to-edited KL divergence in natural-log units."""

    aggregate: float | None
    temperature: float
    total_probe_count: int
    evaluated_probe_count: int
    total_position_count: int
    evaluated_position_count: int
    coverage: float
    probes: tuple[ControlProbeEvidence, ...]
    warnings: tuple[str, ...]
    metric_id: Literal["control.mean_kl_divergence"] = "control.mean_kl_divergence"
    direction: Literal["lower-is-better"] = "lower-is-better"
    reduction: Literal["arithmetic_mean_over_available_probe_means"] = (
        "arithmetic_mean_over_available_probe_means"
    )
    divergence_direction: Literal["baseline||edited"] = "baseline||edited"
    unit: Literal["nats"] = "nats"

    @property
    def missing_probe_count(self) -> int:
        """Return how many control probes did not contribute to the aggregate."""

        return self.total_probe_count - self.evaluated_probe_count

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready result with every probe contribution."""

        return {
            "metric_id": self.metric_id,
            "direction": self.direction,
            "divergence_direction": self.divergence_direction,
            "aggregate": self.aggregate,
            "reduction": self.reduction,
            "unit": self.unit,
            "temperature": self.temperature,
            "total_probe_count": self.total_probe_count,
            "evaluated_probe_count": self.evaluated_probe_count,
            "missing_probe_count": self.missing_probe_count,
            "total_position_count": self.total_position_count,
            "evaluated_position_count": self.evaluated_position_count,
            "coverage": self.coverage,
            "probes": [probe.as_dict() for probe in self.probes],
            "warnings": list(self.warnings),
        }


def reduce_control_kl_divergence(
    probes: Sequence[PairedControlLogits],
    *,
    temperature: float = 1.0,
) -> ControlDivergenceReduction:
    """Reduce aligned logits as mean probe KL(baseline || edited), in nats."""

    if len(probes) == 0:
        raise ControlDivergenceValidationError(
            "probes must contain at least one control probe"
        )
    if len(probes) > _MAX_PROBES:
        raise ControlDivergenceValidationError(
            f"probes must not contain more than {_MAX_PROBES} control probes"
        )
    normalized_temperature = _finite_positive_temperature(temperature)

    evidence: list[ControlProbeEvidence] = []
    probe_means: list[float] = []
    first_index_by_id: dict[str, int] = {}
    total_position_count = 0
    evaluated_position_count = 0
    total_logit_values = 0

    for index, probe in enumerate(probes):
        if not isinstance(probe, PairedControlLogits):
            raise ControlDivergenceValidationError(
                f"probes[{index}] must be a PairedControlLogits instance"
            )
        probe_id = _validate_probe_id(probe.probe_id, index)
        first_index = first_index_by_id.setdefault(probe_id, index)
        if first_index != index:
            raise ControlDivergenceValidationError(
                f"probes[{index}].probe_id {probe_id!r} duplicates "
                f"probes[{first_index}].probe_id"
            )

        baseline, baseline_values = _normalize_optional_logits(
            probe.baseline_logits,
            path=f"probes[{index}].baseline_logits",
        )
        edited, edited_values = _normalize_optional_logits(
            probe.edited_logits,
            path=f"probes[{index}].edited_logits",
        )
        total_logit_values += baseline_values + edited_values
        if total_logit_values > _MAX_LOGIT_VALUES:
            raise ControlDivergenceValidationError(
                f"supplied logits must not exceed {_MAX_LOGIT_VALUES} total values"
            )
        pair_is_complete = baseline is not None and edited is not None
        missing_reason = _validate_missing_reason(
            probe.missing_reason,
            pair_is_complete=pair_is_complete,
            index=index,
        )
        known_position_count = max(
            len(baseline) if baseline is not None else 0,
            len(edited) if edited is not None else 0,
        )
        total_position_count += known_position_count

        if baseline is None or edited is None:
            known_vocabulary_size = _known_vocabulary_size(baseline, edited)
            evidence.append(
                ControlProbeEvidence(
                    probe_id=probe_id,
                    position_kl_divergences=None,
                    mean_kl_divergence=None,
                    position_count=known_position_count,
                    vocabulary_size=known_vocabulary_size,
                    missing_reason=missing_reason,
                )
            )
            continue

        if len(baseline) != len(edited):
            raise ControlDivergenceValidationError(
                f"probes[{index}] baseline and edited position counts must match"
            )
        vocabulary_size = len(baseline[0])
        for position, edited_row in enumerate(edited):
            if len(edited_row) != vocabulary_size:
                raise ControlDivergenceValidationError(
                    f"probes[{index}].edited_logits[{position}] vocabulary size "
                    f"{len(edited_row)} must match {vocabulary_size}"
                )

        position_values = tuple(
            _kl_from_logits(
                baseline_row,
                edited_row,
                temperature=normalized_temperature,
                path=f"probes[{index}].position[{position}]",
            )
            for position, (baseline_row, edited_row) in enumerate(
                zip(baseline, edited, strict=True)
            )
        )
        mean_kl = _finite_mean(
            position_values,
            path=f"probes[{index}].mean_kl_divergence",
        )
        evaluated_position_count += len(position_values)
        probe_means.append(mean_kl)
        evidence.append(
            ControlProbeEvidence(
                probe_id=probe_id,
                position_kl_divergences=position_values,
                mean_kl_divergence=mean_kl,
                position_count=len(position_values),
                vocabulary_size=vocabulary_size,
                missing_reason=None,
            )
        )

    aggregate = (
        _finite_mean(probe_means, path="aggregate") if probe_means else None
    )
    evaluated_probe_count = len(probe_means)
    return ControlDivergenceReduction(
        aggregate=aggregate,
        temperature=normalized_temperature,
        total_probe_count=len(evidence),
        evaluated_probe_count=evaluated_probe_count,
        total_position_count=total_position_count,
        evaluated_position_count=evaluated_position_count,
        coverage=evaluated_probe_count / len(evidence),
        probes=tuple(evidence),
        warnings=_coverage_warnings(evaluated_probe_count, len(evidence)),
    )


def _normalize_optional_logits(
    value: Sequence[Sequence[float]] | None,
    *,
    path: str,
) -> tuple[tuple[tuple[float, ...], ...] | None, int]:
    if value is None:
        return None, 0
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ControlDivergenceValidationError(f"{path} must be a sequence of rows or null")
    if len(value) == 0:
        raise ControlDivergenceValidationError(f"{path} must contain at least one position")
    if len(value) > _MAX_POSITIONS_PER_PROBE:
        raise ControlDivergenceValidationError(
            f"{path} must not contain more than {_MAX_POSITIONS_PER_PROBE} positions"
        )

    rows: list[tuple[float, ...]] = []
    vocabulary_size: int | None = None
    for row_index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ControlDivergenceValidationError(
                f"{path}[{row_index}] must be a sequence of logits"
            )
        if len(row) < 2:
            raise ControlDivergenceValidationError(
                f"{path}[{row_index}] must contain at least two vocabulary logits"
            )
        if len(row) > _MAX_VOCABULARY_SIZE:
            raise ControlDivergenceValidationError(
                f"{path}[{row_index}] must not exceed vocabulary size "
                f"{_MAX_VOCABULARY_SIZE}"
            )
        if vocabulary_size is None:
            vocabulary_size = len(row)
        elif len(row) != vocabulary_size:
            raise ControlDivergenceValidationError(
                f"{path}[{row_index}] vocabulary size {len(row)} must match "
                f"{vocabulary_size}"
            )

        normalized_row: list[float] = []
        for column_index, item in enumerate(row):
            normalized_row.append(
                _finite_number(item, path=f"{path}[{row_index}][{column_index}]")
            )
        rows.append(tuple(normalized_row))
    return tuple(rows), len(rows) * cast(int, vocabulary_size)


def _kl_from_logits(
    baseline_logits: tuple[float, ...],
    edited_logits: tuple[float, ...],
    *,
    temperature: float,
    path: str,
) -> float:
    baseline_log_probabilities = _log_softmax(
        baseline_logits,
        temperature=temperature,
        path=f"{path}.baseline",
    )
    edited_log_probabilities = _log_softmax(
        edited_logits,
        temperature=temperature,
        path=f"{path}.edited",
    )
    try:
        divergence = math.fsum(
            math.exp(baseline_log_probability)
            * (baseline_log_probability - edited_log_probability)
            for baseline_log_probability, edited_log_probability in zip(
                baseline_log_probabilities,
                edited_log_probabilities,
                strict=True,
            )
        )
    except OverflowError as error:
        raise ControlDivergenceValidationError(
            f"{path} KL divergence is outside the finite float range"
        ) from error
    if not math.isfinite(divergence):
        raise ControlDivergenceValidationError(
            f"{path} KL divergence is outside the finite float range"
        )
    if divergence < -1e-12:
        raise ControlDivergenceValidationError(
            f"{path} KL divergence became negative beyond numeric tolerance"
        )
    return max(0.0, divergence)


def _log_softmax(
    logits: tuple[float, ...],
    *,
    temperature: float,
    path: str,
) -> tuple[float, ...]:
    scaled = tuple(value / temperature for value in logits)
    if any(not math.isfinite(value) for value in scaled):
        raise ControlDivergenceValidationError(
            f"{path} temperature-scaled logits must remain finite"
        )
    maximum = max(scaled)
    normalizer = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in scaled)
    )
    return tuple(value - normalizer for value in scaled)


def _finite_positive_temperature(value: object) -> float:
    normalized = _finite_number(value, path="temperature")
    if normalized <= 0.0:
        raise ControlDivergenceValidationError(
            "temperature must be a finite number greater than zero"
        )
    return normalized


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ControlDivergenceValidationError(f"{path} must be a finite real number")
    try:
        normalized = float(cast(SupportsFloat, value))
    except (TypeError, ValueError) as error:
        raise ControlDivergenceValidationError(
            f"{path} must be a finite real number"
        ) from error
    if not math.isfinite(normalized):
        raise ControlDivergenceValidationError(f"{path} must be finite")
    return normalized


def _validate_probe_id(value: object, index: int) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None
    ):
        raise ControlDivergenceValidationError(
            f"probes[{index}].probe_id must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}"
        )
    return value


def _validate_missing_reason(
    value: object,
    *,
    pair_is_complete: bool,
    index: int,
) -> str | None:
    path = f"probes[{index}].missing_reason"
    if pair_is_complete:
        if value is not None:
            raise ControlDivergenceValidationError(
                f"{path} must be null when both logits values are present"
            )
        return None
    if value is None:
        raise ControlDivergenceValidationError(
            f"{path} is required when either logits value is missing"
        )
    if not isinstance(value, str):
        raise ControlDivergenceValidationError(f"{path} must be a string")
    if not value.strip():
        raise ControlDivergenceValidationError(f"{path} must not be blank")
    if len(value) > 4096:
        raise ControlDivergenceValidationError(f"{path} must not exceed 4096 characters")
    return value


def _known_vocabulary_size(
    baseline: tuple[tuple[float, ...], ...] | None,
    edited: tuple[tuple[float, ...], ...] | None,
) -> int | None:
    if baseline is not None:
        return len(baseline[0])
    if edited is not None:
        return len(edited[0])
    return None


def _finite_mean(values: Sequence[float], *, path: str) -> float:
    try:
        result = math.fsum(values) / len(values)
    except OverflowError as error:
        raise ControlDivergenceValidationError(
            f"{path} is outside the finite float range"
        ) from error
    if not math.isfinite(result):
        raise ControlDivergenceValidationError(f"{path} is outside the finite float range")
    return result


def _coverage_warnings(evaluated: int, total: int) -> tuple[str, ...]:
    missing = total - evaluated
    if missing == 0:
        return ()
    if evaluated == 0:
        return (
            (
                f"aggregate unavailable: evaluated 0 of {total} control probes because every "
                "pair is incomplete"
            ),
        )
    suffix = "probe" if missing == 1 else "probes"
    return (
        (
            f"partial coverage: evaluated {evaluated} of {total} control probes; aggregate "
            f"excludes {missing} missing {suffix}"
        ),
    )
