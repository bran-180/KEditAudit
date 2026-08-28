"""Editor-independent clean/corrupt/restore causal-tracing coordinator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from kedit_audit.adapters import ModelAdapter, TokenSpan, resolve_module_path

CAUSAL_TRACE_RESULT_VERSION = "1.0.0"


class TraceValidationError(ValueError):
    """Raised when a trace request or adapter result is internally inconsistent."""


@dataclass(frozen=True)
class CausalTraceRequest:
    """One deterministic tracing request over an ordered set of module paths."""

    prompt: str
    subject: str
    target: str
    module_paths: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        for field_name in ("prompt", "subject", "target"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if not isinstance(self.module_paths, tuple) or not self.module_paths:
            raise ValueError("module_paths must be a non-empty tuple")
        if any(not isinstance(path, str) or not path for path in self.module_paths):
            raise ValueError("every module path must be a non-empty string")
        if len(set(self.module_paths)) != len(self.module_paths):
            raise ValueError("module_paths must be unique")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 4294967295
        ):
            raise ValueError("seed must be an integer in [0, 4294967295]")


@dataclass(frozen=True)
class CleanTraceRun:
    """Clean target score and captured activations keyed by requested path."""

    target_score: float
    activations: Mapping[str, object]

    def __post_init__(self) -> None:
        normalized_score = _require_finite_score(self.target_score, name="clean target score")
        object.__setattr__(self, "target_score", normalized_score)
        if not isinstance(self.activations, Mapping):
            raise TraceValidationError("clean activations must be a mapping keyed by module path")
        copied: dict[str, object] = {}
        for path, activation in self.activations.items():
            if not isinstance(path, str) or not path:
                raise TraceValidationError("clean activation paths must be non-empty strings")
            if activation is None:
                raise TraceValidationError(f"clean activation for {path} must not be null")
            copied[path] = activation
        object.__setattr__(self, "activations", MappingProxyType(copied))


@dataclass(frozen=True)
class ModuleRestorationEvidence:
    """Raw paired scores and recovery reduction for one restored module."""

    module_path: str
    clean_target_score: float
    corrupted_target_score: float
    restored_target_score: float
    restoration_delta: float
    recovery_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready heatmap evidence."""

        return {
            "module_path": self.module_path,
            "clean_target_score": self.clean_target_score,
            "corrupted_target_score": self.corrupted_target_score,
            "restored_target_score": self.restored_target_score,
            "restoration_delta": self.restoration_delta,
            "recovery_fraction": self.recovery_fraction,
        }


@dataclass(frozen=True)
class CausalTraceResult:
    """JSON-ready tracing evidence that excludes prompts and activation tensors."""

    model_state_id: str
    model_revision: str
    tokenizer_revision: str
    seed: int
    subject_token_span: TokenSpan
    clean_target_score: float
    corrupted_target_score: float
    modules: tuple[ModuleRestorationEvidence, ...]
    warnings: tuple[str, ...]
    schema_version: str = CAUSAL_TRACE_RESULT_VERSION

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic artifact without raw prompts or tensors."""

        return {
            "schema_version": self.schema_version,
            "model_state_id": self.model_state_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "seed": self.seed,
            "subject_token_span": {
                "start": self.subject_token_span.start,
                "end": self.subject_token_span.end,
            },
            "clean_target_score": self.clean_target_score,
            "corrupted_target_score": self.corrupted_target_score,
            "modules": [module.as_dict() for module in self.modules],
            "warnings": list(self.warnings),
        }


class CausalTraceAdapter(ModelAdapter, Protocol):
    """Additional execution operations required by the tracing coordinator."""

    def run_clean(
        self,
        *,
        prompt: str,
        target: str,
        module_paths: tuple[str, ...],
    ) -> CleanTraceRun:
        """Run clean scoring and capture every requested module activation."""

    def create_corruption(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        subject_span: TokenSpan,
        seed: int,
    ) -> object:
        """Create the one corruption tensor reused by every paired run."""

    def run_corrupted(
        self,
        *,
        prompt: str,
        target: str,
        corruption: object,
    ) -> float:
        """Score the corrupted run with the supplied fixed corruption."""

    def run_restored(
        self,
        *,
        prompt: str,
        target: str,
        corruption: object,
        module_path: str,
        clean_activation: object,
    ) -> float:
        """Restore one clean activation while reusing the fixed corruption."""


def run_causal_trace(
    adapter: CausalTraceAdapter,
    request: CausalTraceRequest,
) -> CausalTraceResult:
    """Run clean/corrupt/restore comparisons with exactly one corruption object."""

    for module_path in request.module_paths:
        resolve_module_path(adapter.module_root, module_path)

    prompt_token_ids = adapter.tokenize(request.prompt)
    adapter.tokenize(request.target)
    subject_span = adapter.subject_token_span(request.prompt, request.subject)
    if subject_span.end > len(prompt_token_ids):
        raise TraceValidationError("subject token span exceeds the tokenized prompt length")

    clean_run = adapter.run_clean(
        prompt=request.prompt,
        target=request.target,
        module_paths=request.module_paths,
    )
    _validate_clean_activations(clean_run.activations, request.module_paths)

    corruption = adapter.create_corruption(
        prompt_token_ids=prompt_token_ids,
        subject_span=subject_span,
        seed=request.seed,
    )
    if corruption is None:
        raise TraceValidationError("adapter create_corruption must not return null")

    corrupted_score = adapter.run_corrupted(
        prompt=request.prompt,
        target=request.target,
        corruption=corruption,
    )
    corrupted_score = _require_finite_score(
        corrupted_score,
        name="corrupted target score",
    )

    warnings: list[str] = []
    modules: list[ModuleRestorationEvidence] = []
    for module_path in request.module_paths:
        restored_score = adapter.run_restored(
            prompt=request.prompt,
            target=request.target,
            corruption=corruption,
            module_path=module_path,
            clean_activation=clean_run.activations[module_path],
        )
        restored_score = _require_finite_score(
            restored_score,
            name=f"restored target score for {module_path}",
        )
        restoration_delta = _require_finite_score(
            restored_score - corrupted_score,
            name=f"restoration delta for {module_path}",
        )
        clean_corrupted_gap = _require_finite_score(
            clean_run.target_score - corrupted_score,
            name="clean/corrupted target-score gap",
        )
        recovery_fraction: float | None
        if clean_corrupted_gap == 0.0:
            recovery_fraction = None
            warnings.append(
                f"recovery fraction unavailable for {module_path} because "
                "clean and corrupted scores are equal"
            )
        else:
            recovery_fraction = _require_finite_score(
                restoration_delta / clean_corrupted_gap,
                name=f"recovery fraction for {module_path}",
            )
        modules.append(
            ModuleRestorationEvidence(
                module_path=module_path,
                clean_target_score=clean_run.target_score,
                corrupted_target_score=corrupted_score,
                restored_target_score=restored_score,
                restoration_delta=restoration_delta,
                recovery_fraction=recovery_fraction,
            )
        )

    metadata = adapter.metadata
    return CausalTraceResult(
        model_state_id=metadata.state_id,
        model_revision=metadata.model_revision,
        tokenizer_revision=metadata.tokenizer_revision,
        seed=request.seed,
        subject_token_span=subject_span,
        clean_target_score=clean_run.target_score,
        corrupted_target_score=corrupted_score,
        modules=tuple(modules),
        warnings=tuple(warnings),
    )


def _validate_clean_activations(
    activations: Mapping[str, object],
    module_paths: tuple[str, ...],
) -> None:
    requested = set(module_paths)
    supplied = set(activations)
    missing = sorted(requested - supplied)
    extra = sorted(supplied - requested)
    issues: list[str] = []
    if missing:
        issues.append(f"missing clean activations for {', '.join(missing)}")
    if extra:
        issues.append(f"unexpected clean activations for {', '.join(extra)}")
    if issues:
        raise TraceValidationError("; ".join(issues))


def _require_finite_score(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceValidationError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise TraceValidationError(f"{name} must be finite")
    return normalized
