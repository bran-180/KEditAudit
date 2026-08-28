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

__all__ = [
    "AdapterInputError",
    "AdapterPairValidationError",
    "FakeModelAdapter",
    "ModelAdapter",
    "ModelMetadata",
    "TokenSpan",
    "validate_adapter_pair",
]
