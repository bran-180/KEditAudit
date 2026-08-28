"""Editor-independent model adapter contracts and offline test doubles."""

from kedit_audit.adapters.editor import (
    EDITOR_ADAPTER_SCHEMA_VERSION,
    BaselineContaminationError,
    ChangedTensorRecord,
    EditorAdapterMetadata,
    EditorArtifactAdapter,
    EditorArtifactSession,
    EditorArtifactValidationError,
    EditorLifecycleError,
    PairedTargetScoreEvidence,
    bind_editor_states,
)
from kedit_audit.adapters.model import (
    AdapterInputError,
    AdapterPairValidationError,
    FakeModelAdapter,
    ModelAdapter,
    ModelMetadata,
    TokenSpan,
    validate_adapter_pair,
)
from kedit_audit.adapters.modules import ModulePathError, resolve_module_path
from kedit_audit.adapters.transformers import (
    SUPPORTED_TORCH_VERSION,
    SUPPORTED_TRANSFORMERS_VERSION,
    AdapterCompatibilityError,
    GPT2CausalLMAdapter,
)

__all__ = [
    "EDITOR_ADAPTER_SCHEMA_VERSION",
    "SUPPORTED_TORCH_VERSION",
    "SUPPORTED_TRANSFORMERS_VERSION",
    "AdapterCompatibilityError",
    "AdapterInputError",
    "AdapterPairValidationError",
    "BaselineContaminationError",
    "ChangedTensorRecord",
    "EditorAdapterMetadata",
    "EditorArtifactAdapter",
    "EditorArtifactSession",
    "EditorArtifactValidationError",
    "EditorLifecycleError",
    "FakeModelAdapter",
    "GPT2CausalLMAdapter",
    "ModelAdapter",
    "ModelMetadata",
    "ModulePathError",
    "PairedTargetScoreEvidence",
    "TokenSpan",
    "bind_editor_states",
    "resolve_module_path",
    "validate_adapter_pair",
]
