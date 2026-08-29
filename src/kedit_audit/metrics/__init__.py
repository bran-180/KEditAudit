"""Deterministic metric reducers operating on caller-supplied values."""

from kedit_audit.metrics.behavioral import (
    LogitsValidationError,
    PairedProbeScore,
    ProbeReductionValidationError,
    ProbeScoreEvidence,
    ProbeScoreReduction,
    SequenceLogProbability,
    reduce_efficacy_log_probability_deltas,
    reduce_generality_log_probability_deltas,
    reduce_locality_log_probability_drift,
    reduce_portability_log_probability_deltas,
    target_sequence_log_probability,
)
from kedit_audit.metrics.distributional import (
    ControlDivergenceReduction,
    ControlDivergenceValidationError,
    ControlProbeEvidence,
    PairedControlLogits,
    reduce_control_kl_divergence,
)
from kedit_audit.metrics.structural import (
    StructuralDifferenceResult,
    StructuralDifferenceValidationError,
    TensorDifferenceEvidence,
    WeightTensorPair,
    analyze_weight_differences,
)

__all__ = [
    "ControlDivergenceReduction",
    "ControlDivergenceValidationError",
    "ControlProbeEvidence",
    "LogitsValidationError",
    "PairedControlLogits",
    "PairedProbeScore",
    "ProbeReductionValidationError",
    "ProbeScoreEvidence",
    "ProbeScoreReduction",
    "SequenceLogProbability",
    "StructuralDifferenceResult",
    "StructuralDifferenceValidationError",
    "TensorDifferenceEvidence",
    "WeightTensorPair",
    "analyze_weight_differences",
    "reduce_control_kl_divergence",
    "reduce_efficacy_log_probability_deltas",
    "reduce_generality_log_probability_deltas",
    "reduce_locality_log_probability_drift",
    "reduce_portability_log_probability_deltas",
    "target_sequence_log_probability",
]
