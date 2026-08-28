"""Parse validated data-only editor artifact manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from kedit_audit.adapters.editor import (
    ChangedTensorRecord,
    EditorAdapterMetadata,
    JSONValue,
)
from kedit_audit.artifacts import validate_editor_artifact_manifest


@dataclass(frozen=True)
class ImportedEditorManifest:
    """Normalized immutable editor manifest without model or tensor values."""

    artifact_id: str
    artifact_kind: str
    metadata: EditorAdapterMetadata
    changed_tensors: tuple[ChangedTensorRecord, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the normalized manifest in a deterministic JSON-ready shape."""

        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "metadata": self.metadata.as_dict(),
            "changed_tensors": [tensor.as_dict() for tensor in self.changed_tensors],
        }


def parse_editor_artifact_manifest(instance: object) -> ImportedEditorManifest:
    """Validate and normalize one untrusted editor artifact mapping."""

    validate_editor_artifact_manifest(instance)
    root = cast(Mapping[str, object], instance)
    editor = cast(Mapping[str, object], root["editor"])
    model = cast(Mapping[str, object], root["model"])
    baseline = cast(Mapping[str, object], model["baseline"])
    edited = cast(Mapping[str, object], model["edited"])
    raw_tensors = cast(Sequence[Mapping[str, object]], root["changed_tensors"])

    metadata = EditorAdapterMetadata(
        editor_name=cast(str, editor["name"]),
        editor_revision=cast(str, editor["revision"]),
        source_repository=cast(str, editor["source_repository"]),
        model_architecture=cast(str, model["architecture"]),
        baseline_state_id=cast(str, baseline["state_id"]),
        edited_state_id=cast(str, edited["state_id"]),
        baseline_artifact_sha256=cast(str, baseline["artifact_sha256"]),
        edited_artifact_sha256=cast(str, edited["artifact_sha256"]),
        hyperparameters=cast(Mapping[str, JSONValue], editor["hyperparameters"]),
    )
    tensors = tuple(
        ChangedTensorRecord(
            name=cast(str, tensor["name"]),
            shape=tuple(cast(Sequence[int], tensor["shape"])),
            dtype=cast(str, tensor["dtype"]),
            device=cast(str, tensor["device"]),
            baseline_sha256=cast(str, tensor["baseline_sha256"]),
            edited_sha256=cast(str, tensor["edited_sha256"]),
        )
        for tensor in raw_tensors
    )
    return ImportedEditorManifest(
        artifact_id=cast(str, root["artifact_id"]),
        artifact_kind=cast(str, root["artifact_kind"]),
        metadata=metadata,
        changed_tensors=tensors,
    )
