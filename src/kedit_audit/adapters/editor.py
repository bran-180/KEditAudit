"""Data-only editor artifact contracts and contamination-detecting lifecycle."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, Self, TypeAlias, cast, runtime_checkable
from urllib.parse import urlsplit

from kedit_audit.adapters.model import ModelAdapter, validate_adapter_pair
from kedit_audit.metrics.behavioral import SequenceLogProbability

EDITOR_ADAPTER_SCHEMA_VERSION = "1.0.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TENSOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,511}$")
_MAX_JSON_DEPTH = 8
_MAX_JSON_ITEMS = 4_096
_MAX_JSON_STRING = 4_096
_MAX_TENSOR_RANK = 16

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
FrozenJSONValue: TypeAlias = (
    JSONScalar | tuple["FrozenJSONValue", ...] | Mapping[str, "FrozenJSONValue"]
)


class EditorArtifactValidationError(ValueError):
    """Raised when editor provenance or changed-tensor metadata is invalid."""


class EditorLifecycleError(RuntimeError):
    """Raised when baseline and edited states cannot be used as an isolated pair."""


class BaselineContaminationError(EditorLifecycleError):
    """Raised when edited evaluation changes the baseline score."""


@dataclass(frozen=True)
class EditorAdapterMetadata:
    """Immutable provenance for one imported editor artifact."""

    editor_name: str
    editor_revision: str
    source_repository: str
    model_architecture: str
    baseline_state_id: str
    edited_state_id: str
    baseline_artifact_sha256: str
    edited_artifact_sha256: str
    hyperparameters: Mapping[str, JSONValue]
    schema_version: str = EDITOR_ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EDITOR_ADAPTER_SCHEMA_VERSION:
            raise EditorArtifactValidationError(
                f"schema_version must equal {EDITOR_ADAPTER_SCHEMA_VERSION!r}"
            )
        if _IDENTIFIER.fullmatch(self.editor_name) is None:
            raise EditorArtifactValidationError(
                "editor_name must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
            )
        if _REVISION.fullmatch(self.editor_revision) is None:
            raise EditorArtifactValidationError(
                "editor_revision must be a lowercase 40-character Git commit"
            )
        _validate_https_repository(self.source_repository)
        if _IDENTIFIER.fullmatch(self.model_architecture) is None:
            raise EditorArtifactValidationError(
                "model_architecture must be a stable public identifier"
            )
        _nonblank(self.baseline_state_id, path="baseline_state_id")
        _nonblank(self.edited_state_id, path="edited_state_id")
        if self.baseline_state_id == self.edited_state_id:
            raise EditorArtifactValidationError(
                "baseline_state_id and edited_state_id must differ"
            )
        _sha256(self.baseline_artifact_sha256, path="baseline_artifact_sha256")
        _sha256(self.edited_artifact_sha256, path="edited_artifact_sha256")
        frozen_hyperparameters = _freeze_mapping(
            self.hyperparameters,
            path="hyperparameters",
        )
        object.__setattr__(self, "hyperparameters", frozen_hyperparameters)

    def as_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready provenance."""

        return {
            "schema_version": self.schema_version,
            "editor_name": self.editor_name,
            "editor_revision": self.editor_revision,
            "source_repository": self.source_repository,
            "model_architecture": self.model_architecture,
            "baseline_state_id": self.baseline_state_id,
            "edited_state_id": self.edited_state_id,
            "baseline_artifact_sha256": self.baseline_artifact_sha256,
            "edited_artifact_sha256": self.edited_artifact_sha256,
            "hyperparameters": _thaw_json(
                cast(Mapping[str, FrozenJSONValue], self.hyperparameters)
            ),
        }


@dataclass(frozen=True)
class ChangedTensorRecord:
    """Hash-only inventory entry for a tensor reported as changed by an editor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    baseline_sha256: str
    edited_sha256: str

    def __post_init__(self) -> None:
        if _TENSOR_NAME.fullmatch(self.name) is None:
            raise EditorArtifactValidationError(
                "changed tensor name must be a public dotted identifier"
            )
        if not isinstance(self.shape, tuple) or not self.shape:
            raise EditorArtifactValidationError("changed tensor shape must be a non-empty tuple")
        if len(self.shape) > _MAX_TENSOR_RANK:
            raise EditorArtifactValidationError(
                f"changed tensor shape must not exceed rank {_MAX_TENSOR_RANK}"
            )
        for index, dimension in enumerate(self.shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise EditorArtifactValidationError(
                    f"changed tensor shape[{index}] must be a positive integer"
                )
        _nonblank(self.dtype, path="changed tensor dtype")
        _nonblank(self.device, path="changed tensor device")
        _sha256(self.baseline_sha256, path="changed tensor baseline_sha256")
        _sha256(self.edited_sha256, path="changed tensor edited_sha256")
        if self.baseline_sha256 == self.edited_sha256:
            raise EditorArtifactValidationError(
                "changed tensor baseline_sha256 and edited_sha256 must differ"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready inventory entry without tensor values."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "baseline_sha256": self.baseline_sha256,
            "edited_sha256": self.edited_sha256,
        }


@dataclass(frozen=True)
class PairedTargetScoreEvidence:
    """One baseline/edited target-score pair verified against contamination."""

    baseline: SequenceLogProbability
    edited: SequenceLogProbability

    def as_dict(self) -> dict[str, object]:
        """Return raw score evidence without prompts."""

        return {
            "baseline": self.baseline.as_dict(),
            "edited": self.edited.as_dict(),
        }


@runtime_checkable
class EditorArtifactAdapter(Protocol):
    """Bind a data-only external-editor artifact to already-separated states."""

    @property
    def metadata(self) -> EditorAdapterMetadata:
        """Return immutable editor and artifact provenance."""

    @property
    def changed_tensors(self) -> tuple[ChangedTensorRecord, ...]:
        """Return the complete reported changed-tensor inventory."""

    def bind_states(
        self,
        *,
        baseline: ModelAdapter,
        edited: ModelAdapter,
    ) -> EditorArtifactSession:
        """Validate and bind distinct baseline and edited states."""


class EditorArtifactSession:
    """Short-lived evaluator that detects shared roots and baseline score changes."""

    def __init__(
        self,
        *,
        metadata: EditorAdapterMetadata,
        changed_tensors: tuple[ChangedTensorRecord, ...],
        baseline: ModelAdapter,
        edited: ModelAdapter,
    ) -> None:
        validate_adapter_pair(baseline, edited)
        if baseline.module_root is edited.module_root:
            raise EditorLifecycleError(
                "baseline and edited adapters must expose distinct model roots"
            )
        if baseline.metadata.state_id != metadata.baseline_state_id:
            raise EditorLifecycleError(
                "editor artifact baseline state_id does not match the bound baseline adapter"
            )
        if edited.metadata.state_id != metadata.edited_state_id:
            raise EditorLifecycleError(
                "editor artifact edited state_id does not match the bound edited adapter"
            )
        _validate_changed_tensors(changed_tensors)
        self._metadata = metadata
        self._changed_tensors = tuple(changed_tensors)
        self._baseline = baseline
        self._edited = edited
        self._closed = False

    @property
    def metadata(self) -> EditorAdapterMetadata:
        """Return editor provenance for the bound pair."""

        return self._metadata

    @property
    def changed_tensors(self) -> tuple[ChangedTensorRecord, ...]:
        """Return the immutable changed-tensor inventory."""

        return self._changed_tensors

    @property
    def closed(self) -> bool:
        """Return whether this session has been disposed."""

        return self._closed

    def score_target_pair(self, prompt: str, target: str) -> PairedTargetScoreEvidence:
        """Score baseline, edited, then baseline again to detect contamination."""

        self._require_open()
        baseline_before = self._baseline.score_target(prompt, target)
        try:
            edited_score = self._edited.score_target(prompt, target)
        except Exception as edited_error:
            self._verify_baseline_after_failure(
                prompt=prompt,
                target=target,
                baseline_before=baseline_before,
                edited_error=edited_error,
            )
            raise

        try:
            baseline_after = self._baseline.score_target(prompt, target)
        except Exception as error:
            self.close()
            raise BaselineContaminationError(
                "baseline state could not be verified after edited evaluation"
            ) from error
        if baseline_after != baseline_before:
            self.close()
            raise BaselineContaminationError(
                "baseline target score changed during edited evaluation"
            )
        return PairedTargetScoreEvidence(
            baseline=baseline_before,
            edited=edited_score,
        )

    def close(self) -> None:
        """Dispose this non-owning binding and reject future evaluations."""

        self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        self.close()

    def _verify_baseline_after_failure(
        self,
        *,
        prompt: str,
        target: str,
        baseline_before: SequenceLogProbability,
        edited_error: Exception,
    ) -> None:
        try:
            baseline_after = self._baseline.score_target(prompt, target)
        except Exception as verification_error:  # noqa: BLE001 - preserve primary error.
            self.close()
            edited_error.add_note(
                "baseline verification also failed after edited evaluation error: "
                f"{type(verification_error).__name__}: {verification_error}"
            )
            return
        if baseline_after != baseline_before:
            self.close()
            raise BaselineContaminationError(
                "baseline target score changed during failed edited evaluation"
            ) from edited_error

    def _require_open(self) -> None:
        if self._closed:
            raise EditorLifecycleError("editor artifact session is closed")


def bind_editor_states(
    *,
    metadata: EditorAdapterMetadata,
    changed_tensors: Sequence[ChangedTensorRecord],
    baseline: ModelAdapter,
    edited: ModelAdapter,
) -> EditorArtifactSession:
    """Create an isolated non-owning session for an imported editor artifact."""

    if not isinstance(metadata, EditorAdapterMetadata):
        raise EditorArtifactValidationError("metadata must be EditorAdapterMetadata")
    if not isinstance(baseline, ModelAdapter):
        raise EditorLifecycleError("baseline must satisfy ModelAdapter")
    if not isinstance(edited, ModelAdapter):
        raise EditorLifecycleError("edited must satisfy ModelAdapter")
    normalized_tensors = tuple(changed_tensors)
    return EditorArtifactSession(
        metadata=metadata,
        changed_tensors=normalized_tensors,
        baseline=baseline,
        edited=edited,
    )


def _validate_changed_tensors(changed_tensors: tuple[ChangedTensorRecord, ...]) -> None:
    first_index_by_name: dict[str, int] = {}
    for index, tensor in enumerate(changed_tensors):
        if not isinstance(tensor, ChangedTensorRecord):
            raise EditorArtifactValidationError(
                f"changed_tensors[{index}] must be ChangedTensorRecord"
            )
        first_index = first_index_by_name.setdefault(tensor.name, index)
        if first_index != index:
            raise EditorArtifactValidationError(
                f"changed_tensors[{index}].name {tensor.name!r} duplicates "
                f"changed_tensors[{first_index}].name"
            )


def _validate_https_repository(value: object) -> None:
    if not isinstance(value, str) or len(value) > 2048:
        raise EditorArtifactValidationError(
            "source_repository must be an HTTPS URL no longer than 2048 characters"
        )
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise EditorArtifactValidationError(
            "source_repository must be a credential-free HTTPS repository URL without query or fragment"
        )


def _freeze_mapping(value: object, *, path: str) -> Mapping[str, FrozenJSONValue]:
    if not isinstance(value, Mapping):
        raise EditorArtifactValidationError(f"{path} must be a JSON object")
    budget = [0]
    frozen = _freeze_json(value, path=path, depth=0, budget=budget)
    return cast(Mapping[str, FrozenJSONValue], frozen)


def _freeze_json(
    value: object,
    *,
    path: str,
    depth: int,
    budget: list[int],
) -> FrozenJSONValue:
    if depth > _MAX_JSON_DEPTH:
        raise EditorArtifactValidationError(
            f"{path} must not exceed nesting depth {_MAX_JSON_DEPTH}"
        )
    budget[0] += 1
    if budget[0] > _MAX_JSON_ITEMS:
        raise EditorArtifactValidationError(
            f"hyperparameters must not exceed {_MAX_JSON_ITEMS} total values"
        )
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EditorArtifactValidationError(f"{path} must contain only finite numbers")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_JSON_STRING:
            raise EditorArtifactValidationError(
                f"{path} strings must not exceed {_MAX_JSON_STRING} characters"
            )
        return value
    if isinstance(value, Mapping):
        copied: dict[str, FrozenJSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 256:
                raise EditorArtifactValidationError(
                    f"{path} keys must be non-empty strings no longer than 256 characters"
                )
            copied[key] = _freeze_json(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                budget=budget,
            )
        return MappingProxyType(copied)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                budget=budget,
            )
            for index, item in enumerate(value)
        )
    raise EditorArtifactValidationError(
        f"{path} must contain only JSON scalar, array, or object values"
    )


def _thaw_json(value: FrozenJSONValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _nonblank(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise EditorArtifactValidationError(
            f"{path} must be a non-blank string no longer than 4096 characters"
        )
    return value


def _sha256(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EditorArtifactValidationError(
            f"{path} must be a lowercase 64-character SHA-256 digest"
        )
    return value
