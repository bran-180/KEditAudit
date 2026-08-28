"""Fail-closed importer for normalized ROME editor artifact manifests."""

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

ROME_SOURCE_REPOSITORY = "https://github.com/kmeng01/rome"
_SUPPORTED_ARCHITECTURE = "GPT2LMHeadModel"
_SUPPORTED_MODEL_MODULE = "transformers.models.gpt2.modeling_gpt2"
_REQUIRED_HYPERPARAMETERS = ("layers", "fact_token")


class RomeArtifactAdapter:
    """Bind a validated ROME export manifest without executing ROME code."""

    def __init__(self, manifest: ImportedEditorManifest) -> None:
        metadata = manifest.metadata
        if metadata.editor_name != "ROME":
            raise AdapterCompatibilityError("editor.name must equal 'ROME'")
        if metadata.source_repository != ROME_SOURCE_REPOSITORY:
            raise AdapterCompatibilityError(
                f"ROME source_repository must equal the official ROME repository "
                f"{ROME_SOURCE_REPOSITORY!r}"
            )
        if metadata.model_architecture != _SUPPORTED_ARCHITECTURE:
            raise AdapterCompatibilityError(
                "ROME artifact model.architecture must equal 'GPT2LMHeadModel'"
            )
        if not manifest.changed_tensors:
            raise AdapterCompatibilityError(
                "ROME artifact must report at least one changed tensor"
            )
        missing_hyperparameters = tuple(
            name for name in _REQUIRED_HYPERPARAMETERS if name not in metadata.hyperparameters
        )
        if missing_hyperparameters:
            raise AdapterCompatibilityError(
                "ROME artifact is missing required hyperparameters: "
                + ", ".join(missing_hyperparameters)
            )
        self._manifest = manifest

    @classmethod
    def from_manifest(cls, instance: object) -> RomeArtifactAdapter:
        """Validate and import a data-only KEditAudit ROME manifest."""

        return cls(parse_editor_artifact_manifest(instance))

    @property
    def metadata(self) -> EditorAdapterMetadata:
        """Return immutable ROME revision and artifact provenance."""

        return self._manifest.metadata

    @property
    def changed_tensors(self) -> tuple[ChangedTensorRecord, ...]:
        """Return the reported hash-only changed-tensor inventory."""

        return self._manifest.changed_tensors

    def bind_states(
        self,
        *,
        baseline: ModelAdapter,
        edited: ModelAdapter,
    ) -> EditorArtifactSession:
        """Bind only two exact, separate Transformers GPT-2 model roots."""

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
            f"ROME {state} model root must be an exact GPT2LMHeadModel from the pinned "
            "Transformers implementation"
        )
