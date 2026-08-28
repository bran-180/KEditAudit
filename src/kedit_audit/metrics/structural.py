"""Coverage-aware structural differences over caller-supplied tensor values."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, SupportsFloat, cast

if TYPE_CHECKING:
    from kedit_audit.adapters.editor import ChangedTensorRecord

_MAX_TENSORS = 10_000
_MAX_ELEMENTS_PER_TENSOR = 1_000_000
_MAX_TOTAL_ELEMENTS = 10_000_000
_MAX_SPECTRAL_ITERATIONS = 1_000
_MAX_SPECTRAL_MULTIPLICATIONS = 50_000_000
_SEMANTIC_WARNING = (
    "structural magnitude is descriptive evidence only and is not automatically "
    "interpreted as semantic harm or model safety"
)


class StructuralDifferenceValidationError(ValueError):
    """Raised when a changed-tensor inventory or numeric pair is invalid."""


@dataclass(frozen=True)
class WeightTensorPair:
    """Flattened baseline and edited values for one inventoried tensor."""

    name: str
    baseline_values: Sequence[float] | None
    edited_values: Sequence[float] | None
    missing_reason: str | None = None


@dataclass(frozen=True)
class TensorDifferenceEvidence:
    """Raw per-tensor structural evidence linked to the artifact hashes."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    baseline_sha256: str
    edited_sha256: str
    status: Literal["evaluated", "missing"]
    baseline_frobenius_norm: float | None
    edited_frobenius_norm: float | None
    delta_frobenius_norm: float | None
    relative_delta_frobenius_norm: float | None
    delta_spectral_norm_estimate: float | None
    spectral_iterations: int | None
    missing_reason: str | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready evidence without serializing tensor values."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "baseline_sha256": self.baseline_sha256,
            "edited_sha256": self.edited_sha256,
            "status": self.status,
            "baseline_frobenius_norm": self.baseline_frobenius_norm,
            "edited_frobenius_norm": self.edited_frobenius_norm,
            "delta_frobenius_norm": self.delta_frobenius_norm,
            "relative_delta_frobenius_norm": self.relative_delta_frobenius_norm,
            "delta_spectral_norm_estimate": self.delta_spectral_norm_estimate,
            "spectral_iterations": self.spectral_iterations,
            "missing_reason": self.missing_reason,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class StructuralDifferenceResult:
    """Coverage-aware Frobenius and optional spectral change evidence."""

    aggregate_delta_frobenius_norm: float | None
    aggregate_relative_delta_frobenius_norm: float | None
    total_tensor_count: int
    evaluated_tensor_count: int
    coverage: float
    tensors: tuple[TensorDifferenceEvidence, ...]
    warnings: tuple[str, ...]
    metric_id: Literal["structural.weight_difference"] = "structural.weight_difference"
    reduction: Literal["euclidean_norm_over_available_tensor_deltas"] = (
        "euclidean_norm_over_available_tensor_deltas"
    )
    interpretation: Literal["descriptive-only"] = "descriptive-only"

    @property
    def missing_tensor_count(self) -> int:
        """Return the number of inventoried tensors without numeric pairs."""

        return self.total_tensor_count - self.evaluated_tensor_count

    @property
    def status(self) -> Literal["complete", "incomplete", "unavailable"]:
        """Return coverage status without imposing a metric threshold."""

        if self.evaluated_tensor_count == 0:
            return "unavailable"
        if self.evaluated_tensor_count < self.total_tensor_count:
            return "incomplete"
        return "complete"

    def as_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready evidence with all tensor contributions."""

        return {
            "metric_id": self.metric_id,
            "status": self.status,
            "interpretation": self.interpretation,
            "reduction": self.reduction,
            "aggregate_delta_frobenius_norm": self.aggregate_delta_frobenius_norm,
            "aggregate_relative_delta_frobenius_norm": (
                self.aggregate_relative_delta_frobenius_norm
            ),
            "total_tensor_count": self.total_tensor_count,
            "evaluated_tensor_count": self.evaluated_tensor_count,
            "missing_tensor_count": self.missing_tensor_count,
            "coverage": self.coverage,
            "tensors": [tensor.as_dict() for tensor in self.tensors],
            "warnings": list(self.warnings),
        }


def analyze_weight_differences(
    inventory: Sequence[ChangedTensorRecord],
    pairs: Sequence[WeightTensorPair],
    *,
    spectral_iterations: int | None = None,
) -> StructuralDifferenceResult:
    """Measure numeric changes for an artifact's ordered changed-tensor inventory."""

    normalized_inventory = _validate_inventory(inventory)
    normalized_iterations = _validate_spectral_iterations(spectral_iterations)
    pairs_by_name = _validate_pairs(pairs, inventory=normalized_inventory)

    evidence: list[TensorDifferenceEvidence] = []
    delta_norms: list[float] = []
    baseline_norms: list[float] = []
    total_elements = 0
    spectral_multiplications = 0

    for tensor_index, record in enumerate(normalized_inventory):
        element_count = math.prod(record.shape)
        if element_count > _MAX_ELEMENTS_PER_TENSOR:
            raise StructuralDifferenceValidationError(
                f"inventory[{tensor_index}] tensor {record.name!r} has {element_count} "
                f"elements; limit is {_MAX_ELEMENTS_PER_TENSOR}"
            )
        total_elements += element_count
        if total_elements > _MAX_TOTAL_ELEMENTS:
            raise StructuralDifferenceValidationError(
                f"inventory tensors must not exceed {_MAX_TOTAL_ELEMENTS} total elements"
            )

        pair = pairs_by_name.get(record.name)
        if pair is None:
            evidence.append(
                _missing_evidence(record, reason="no numeric tensor pair was supplied")
            )
            continue

        baseline = _normalize_optional_values(
            pair.baseline_values,
            path=f"pairs[{pair.index}].baseline_values",
            expected_count=element_count,
        )
        edited = _normalize_optional_values(
            pair.edited_values,
            path=f"pairs[{pair.index}].edited_values",
            expected_count=element_count,
        )
        complete = baseline is not None and edited is not None
        missing_reason = _validate_missing_reason(
            pair.missing_reason,
            complete=complete,
            path=f"pairs[{pair.index}].missing_reason",
        )
        if baseline is None or edited is None:
            evidence.append(_missing_evidence(record, reason=cast(str, missing_reason)))
            continue

        deltas = _finite_deltas(baseline, edited, pair_index=pair.index)
        baseline_norm = _euclidean_norm(baseline, path=f"pairs[{pair.index}].baseline")
        edited_norm = _euclidean_norm(edited, path=f"pairs[{pair.index}].edited")
        delta_norm = _euclidean_norm(deltas, path=f"pairs[{pair.index}].delta")
        relative_norm = delta_norm / baseline_norm if baseline_norm != 0.0 else None
        tensor_warnings: list[str] = []
        if baseline_norm == 0.0:
            tensor_warnings.append(
                "relative delta Frobenius norm unavailable because baseline norm is zero"
            )

        spectral_estimate: float | None = None
        used_iterations: int | None = None
        if normalized_iterations is not None:
            if len(record.shape) != 2:
                tensor_warnings.append(
                    "spectral estimate unavailable because the tensor is not rank 2"
                )
            else:
                spectral_multiplications += 2 * element_count * normalized_iterations
                if spectral_multiplications > _MAX_SPECTRAL_MULTIPLICATIONS:
                    raise StructuralDifferenceValidationError(
                        "requested spectral estimates exceed the bounded computation limit "
                        f"of {_MAX_SPECTRAL_MULTIPLICATIONS} multiply-add terms"
                    )
                spectral_estimate = _spectral_norm_power_estimate(
                    deltas,
                    rows=record.shape[0],
                    columns=record.shape[1],
                    iterations=normalized_iterations,
                    scale=delta_norm,
                )
                used_iterations = normalized_iterations
                tensor_warnings.append(
                    "spectral norm is a deterministic power-iteration estimate, not an exact SVD"
                )

        baseline_norms.append(baseline_norm)
        delta_norms.append(delta_norm)
        evidence.append(
            TensorDifferenceEvidence(
                name=record.name,
                shape=record.shape,
                dtype=record.dtype,
                device=record.device,
                baseline_sha256=record.baseline_sha256,
                edited_sha256=record.edited_sha256,
                status="evaluated",
                baseline_frobenius_norm=baseline_norm,
                edited_frobenius_norm=edited_norm,
                delta_frobenius_norm=delta_norm,
                relative_delta_frobenius_norm=relative_norm,
                delta_spectral_norm_estimate=spectral_estimate,
                spectral_iterations=used_iterations,
                missing_reason=None,
                warnings=tuple(tensor_warnings),
            )
        )

    evaluated = len(delta_norms)
    aggregate_delta = _euclidean_norm(delta_norms, path="aggregate delta") if delta_norms else None
    aggregate_baseline = (
        _euclidean_norm(baseline_norms, path="aggregate baseline")
        if baseline_norms
        else None
    )
    aggregate_relative = (
        aggregate_delta / aggregate_baseline
        if aggregate_delta is not None
        and aggregate_baseline is not None
        and aggregate_baseline != 0.0
        else None
    )
    warnings = list(_coverage_warnings(evaluated, len(normalized_inventory)))
    if evaluated and aggregate_baseline == 0.0:
        warnings.append(
            "aggregate relative delta unavailable because evaluated baseline norm is zero"
        )
    warnings.append(_SEMANTIC_WARNING)
    return StructuralDifferenceResult(
        aggregate_delta_frobenius_norm=aggregate_delta,
        aggregate_relative_delta_frobenius_norm=aggregate_relative,
        total_tensor_count=len(normalized_inventory),
        evaluated_tensor_count=evaluated,
        coverage=evaluated / len(normalized_inventory),
        tensors=tuple(evidence),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class _IndexedPair:
    index: int
    baseline_values: Sequence[float] | None
    edited_values: Sequence[float] | None
    missing_reason: str | None


def _validate_inventory(
    inventory: Sequence[ChangedTensorRecord],
) -> tuple[ChangedTensorRecord, ...]:
    # Local import keeps the public metrics package importable while the adapter
    # package is still initializing. Validation still requires the concrete type.
    from kedit_audit.adapters.editor import ChangedTensorRecord

    if isinstance(inventory, (str, bytes)) or not isinstance(inventory, Sequence):
        raise StructuralDifferenceValidationError(
            "inventory must be a sequence of ChangedTensorRecord values"
        )
    if len(inventory) == 0:
        raise StructuralDifferenceValidationError("inventory must contain at least one tensor")
    if len(inventory) > _MAX_TENSORS:
        raise StructuralDifferenceValidationError(
            f"inventory must not contain more than {_MAX_TENSORS} tensors"
        )
    normalized = tuple(inventory)
    first_index_by_name: dict[str, int] = {}
    for index, record in enumerate(normalized):
        if not isinstance(record, ChangedTensorRecord):
            raise StructuralDifferenceValidationError(
                f"inventory[{index}] must be ChangedTensorRecord"
            )
        first_index = first_index_by_name.setdefault(record.name, index)
        if first_index != index:
            raise StructuralDifferenceValidationError(
                f"inventory[{index}].name {record.name!r} duplicates "
                f"inventory[{first_index}].name"
            )
    return normalized


def _validate_pairs(
    pairs: Sequence[WeightTensorPair],
    *,
    inventory: tuple[ChangedTensorRecord, ...],
) -> dict[str, _IndexedPair]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise StructuralDifferenceValidationError(
            "pairs must be a sequence of WeightTensorPair values"
        )
    if len(pairs) > len(inventory):
        raise StructuralDifferenceValidationError(
            "pairs must not outnumber the changed-tensor inventory"
        )
    inventory_names = {record.name for record in inventory}
    by_name: dict[str, _IndexedPair] = {}
    first_index_by_name: dict[str, int] = {}
    for index, pair in enumerate(pairs):
        if not isinstance(pair, WeightTensorPair):
            raise StructuralDifferenceValidationError(
                f"pairs[{index}] must be WeightTensorPair"
            )
        if pair.name not in inventory_names:
            raise StructuralDifferenceValidationError(
                f"pairs[{index}].name {pair.name!r} is not present in the inventory"
            )
        first_index = first_index_by_name.setdefault(pair.name, index)
        if first_index != index:
            raise StructuralDifferenceValidationError(
                f"pairs[{index}].name {pair.name!r} duplicates pairs[{first_index}].name"
            )
        by_name[pair.name] = _IndexedPair(
            index=index,
            baseline_values=pair.baseline_values,
            edited_values=pair.edited_values,
            missing_reason=pair.missing_reason,
        )
    return by_name


def _normalize_optional_values(
    value: Sequence[float] | None,
    *,
    path: str,
    expected_count: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StructuralDifferenceValidationError(f"{path} must be a numeric sequence or null")
    if len(value) != expected_count:
        raise StructuralDifferenceValidationError(
            f"{path} must contain {expected_count} flattened values; got {len(value)}"
        )
    return tuple(
        _finite_number(item, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _finite_deltas(
    baseline: tuple[float, ...],
    edited: tuple[float, ...],
    *,
    pair_index: int,
) -> tuple[float, ...]:
    deltas: list[float] = []
    for index, (baseline_value, edited_value) in enumerate(
        zip(baseline, edited, strict=True)
    ):
        delta = edited_value - baseline_value
        if not math.isfinite(delta):
            raise StructuralDifferenceValidationError(
                f"pairs[{pair_index}].delta[{index}] is outside the finite float range"
            )
        deltas.append(delta)
    return tuple(deltas)


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise StructuralDifferenceValidationError(f"{path} must be a finite real number")
    try:
        normalized = float(cast(SupportsFloat, value))
    except (TypeError, ValueError) as error:
        raise StructuralDifferenceValidationError(
            f"{path} must be a finite real number"
        ) from error
    if not math.isfinite(normalized):
        raise StructuralDifferenceValidationError(f"{path} must be finite")
    return normalized


def _euclidean_norm(values: Sequence[float], *, path: str) -> float:
    result = 0.0
    for value in values:
        result = math.hypot(result, value)
    if not math.isfinite(result):
        raise StructuralDifferenceValidationError(
            f"{path} Frobenius norm is outside the finite float range"
        )
    return result


def _spectral_norm_power_estimate(
    deltas: tuple[float, ...],
    *,
    rows: int,
    columns: int,
    iterations: int,
    scale: float,
) -> float:
    if scale == 0.0:
        return 0.0
    matrix = tuple(value / scale for value in deltas)
    row_vectors = tuple(
        tuple(matrix[row * columns + column] for column in range(columns))
        for row in range(rows)
    )
    initial = max(
        row_vectors,
        key=lambda row: _euclidean_norm(row, path="spectral initial row"),
    )
    initial_norm = _euclidean_norm(initial, path="spectral initial vector")
    vector = tuple(value / initial_norm for value in initial)

    for _ in range(iterations):
        left = tuple(
            math.fsum(
                matrix[row * columns + column] * vector[column]
                for column in range(columns)
            )
            for row in range(rows)
        )
        right = tuple(
            math.fsum(
                matrix[row * columns + column] * left[row]
                for row in range(rows)
            )
            for column in range(columns)
        )
        right_norm = _euclidean_norm(right, path="spectral power vector")
        if right_norm == 0.0:
            return 0.0
        vector = tuple(value / right_norm for value in right)

    left = tuple(
        math.fsum(
            matrix[row * columns + column] * vector[column]
            for column in range(columns)
        )
        for row in range(rows)
    )
    estimate = scale * _euclidean_norm(left, path="spectral estimate")
    if not math.isfinite(estimate):
        raise StructuralDifferenceValidationError(
            "spectral estimate is outside the finite float range"
        )
    return estimate


def _validate_spectral_iterations(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise StructuralDifferenceValidationError(
            "spectral_iterations must be an integer or null"
        )
    if value < 1 or value > _MAX_SPECTRAL_ITERATIONS:
        raise StructuralDifferenceValidationError(
            f"spectral_iterations must be between 1 and {_MAX_SPECTRAL_ITERATIONS}"
        )
    return value


def _validate_missing_reason(value: object, *, complete: bool, path: str) -> str | None:
    if complete:
        if value is not None:
            raise StructuralDifferenceValidationError(
                f"{path} must be null when both tensor values are present"
            )
        return None
    if value is None:
        raise StructuralDifferenceValidationError(
            f"{path} is required when either tensor value is missing"
        )
    if not isinstance(value, str) or not value.strip():
        raise StructuralDifferenceValidationError(f"{path} must be a non-blank string")
    if len(value) > 4096:
        raise StructuralDifferenceValidationError(f"{path} must not exceed 4096 characters")
    return value


def _missing_evidence(
    record: ChangedTensorRecord,
    *,
    reason: str,
) -> TensorDifferenceEvidence:
    return TensorDifferenceEvidence(
        name=record.name,
        shape=record.shape,
        dtype=record.dtype,
        device=record.device,
        baseline_sha256=record.baseline_sha256,
        edited_sha256=record.edited_sha256,
        status="missing",
        baseline_frobenius_norm=None,
        edited_frobenius_norm=None,
        delta_frobenius_norm=None,
        relative_delta_frobenius_norm=None,
        delta_spectral_norm_estimate=None,
        spectral_iterations=None,
        missing_reason=reason,
        warnings=("tensor is excluded from structural aggregates",),
    )


def _coverage_warnings(evaluated: int, total: int) -> tuple[str, ...]:
    missing = total - evaluated
    if missing == 0:
        return ()
    if evaluated == 0:
        return (
            f"structural aggregates unavailable: evaluated 0 of {total} inventoried tensors",
        )
    suffix = "tensor" if missing == 1 else "tensors"
    return (
        (
            f"partial structural coverage: evaluated {evaluated} of {total} inventoried "
            f"tensors; aggregates exclude {missing} missing {suffix}"
        ),
    )
