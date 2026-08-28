"""Fail-closed importer for normalized EasyEdit artifact manifests."""

from __future__ import annotations

from kedit_audit.adapters.editor import (
    ChangedTensorRecord,
    EditorAdapterMetadata,
    EditorArtifactSession,
    bind_editor_states,
)
from kedit_audit.adapters.manifest import (
    ImportedEditorManifest,
    parse_editor_artifact_manifest,
)
from kedit_audit.adapters.model import ModelAdapter
from kedit_audit.adapters.transformers import AdapterCompatibilityError

EASYEDIT_SOURCE_REPOSITORY = "https://github.com/zjunlp/EasyEdit"
_SUPPORTED_ARCHITECTURE = "GPT2LMHeadModel"
_SUPPORTED_MODEL_MODULE = "transformers.models.gpt2.modeling_gpt2"


class EasyEditArtifactAdapter:
    """Bind a validated parameter-edit export without importing EasyEdit."""

    def __init__(self, manifest: ImportedEditorManifest) -> None:
        metadata = manifest.metadata
        if metadata.editor_name != "EasyEdit":
            raise AdapterCompatibilityError("editor.name must equal 'EasyEdit'")
        if metadata.source_repository != EASYEDIT_SOURCE_REPOSITORY:
            raise AdapterCompatibilityError(
                "EasyEdit source_repository must equal the official EasyEdit repository "
                f"{EASYEDIT_SOURCE_REPOSITORY!r}"
            )
        if metadata.model_architecture != _SUPPORTED_ARCHITECTURE:
            raise AdapterCompatibilityError(
                "EasyEdit artifact model.architecture must equal 'GPT2LMHeadModel'"
            )
        if not manifest.changed_tensors:
            raise AdapterCompatibilityError(
                "EasyEdit parameter-edit artifact must report at least one changed tensor"
            )
        algorithm = metadata.hyperparameters.get("algorithm")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise AdapterCompatibilityError(
                "EasyEdit artifact hyperparameters.algorithm must be recorded"
            )
        sequential_edit = metadata.hyperparameters.get("sequential_edit")
        if not isinstance(sequential_edit, bool):
            raise AdapterCompatibilityError(
                "EasyEdit artifact hyperparameters.sequential_edit must be recorded as a boolean"
            )
        if metadata.hyperparameters.get("return_orig_weights") is not True:
            raise AdapterCompatibilityError(
                "EasyEdit artifact hyperparameters.return_orig_weights must be true"
            )
        self._manifest = manifest

    @classmethod
    def from_manifest(cls, instance: object) -> EasyEditArtifactAdapter:
        """Validate and import a data-only KEditAudit EasyEdit manifest."""

        return cls(parse_editor_artifact_manifest(instance))

    @property
    def metadata(self) -> EditorAdapterMetadata:
        """Return immutable EasyEdit revision and export provenance."""

        return self._manifest.metadata

    @property
    def changed_tensors(self) -> tuple[ChangedTensorRecord, ...]:
        """Return the exported hash-only changed-tensor inventory."""

        return self._manifest.changed_tensors

    def bind_states(
        self,
        *,
        baseline: ModelAdapter,
        edited: ModelAdapter,
    ) -> EditorArtifactSession:
        """Bind only exact, separate Transformers GPT-2 model roots."""

        _require_exact_gpt2_root(baseline.module_root, state="baseline")
        _require_exact_gpt2_root(edited.module_root, state="edited")
        return bind_editor_states(
            metadata=self.metadata,
            changed_tensors=self.changed_tensors,
            baseline=baseline,
            edited=edited,
        )

    def as_dict(self) -> dict[str, object]:
        """Return normalized data-only import evidence."""

        return self._manifest.as_dict()


def _require_exact_gpt2_root(root: object, *, state: str) -> None:
    root_class = type(root)
    if (
        root_class.__name__ != _SUPPORTED_ARCHITECTURE
        or root_class.__module__ != _SUPPORTED_MODEL_MODULE
    ):
        raise AdapterCompatibilityError(
            f"EasyEdit {state} model root must be an exact GPT2LMHeadModel from the pinned "
            "Transformers implementation"
        )
