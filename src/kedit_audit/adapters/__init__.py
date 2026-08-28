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

__all__ = [
    "AdapterInputError",
    "AdapterPairValidationError",
    "FakeModelAdapter",
    "ModelAdapter",
    "ModelMetadata",
    "ModulePathError",
    "TokenSpan",
    "resolve_module_path",
    "validate_adapter_pair",
]
