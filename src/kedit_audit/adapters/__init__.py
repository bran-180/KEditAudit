"""Editor-independent model adapter contracts and offline test doubles."""

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
    "SUPPORTED_TORCH_VERSION",
    "SUPPORTED_TRANSFORMERS_VERSION",
    "AdapterCompatibilityError",
    "AdapterInputError",
    "AdapterPairValidationError",
    "FakeModelAdapter",
    "GPT2CausalLMAdapter",
    "ModelAdapter",
    "ModelMetadata",
    "ModulePathError",
    "TokenSpan",
    "resolve_module_path",
    "validate_adapter_pair",
]
