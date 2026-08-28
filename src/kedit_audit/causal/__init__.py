"""Deterministic causal-analysis primitives with explicit lifecycle control."""

from kedit_audit.causal.gpt2 import (
    GPT2_CORRUPTION_STANDARD_DEVIATION,
    GPT2CausalTraceAdapter,
)
from kedit_audit.causal.hooks import (
    ForwardHook,
    ForwardHookModule,
    HookCleanupError,
    HookManager,
    HookRegistrationError,
    RemovableHookHandle,
)
from kedit_audit.causal.tracer import (
    CAUSAL_TRACE_RESULT_VERSION,
    CausalTraceAdapter,
    CausalTraceRequest,
    CausalTraceResult,
    CleanTraceRun,
    ModuleRestorationEvidence,
    TraceValidationError,
    run_causal_trace,
)

__all__ = [
    "CAUSAL_TRACE_RESULT_VERSION",
    "GPT2_CORRUPTION_STANDARD_DEVIATION",
    "CausalTraceAdapter",
    "CausalTraceRequest",
    "CausalTraceResult",
    "CleanTraceRun",
    "ForwardHook",
    "ForwardHookModule",
    "GPT2CausalTraceAdapter",
    "HookCleanupError",
    "HookManager",
    "HookRegistrationError",
    "ModuleRestorationEvidence",
    "RemovableHookHandle",
    "TraceValidationError",
    "run_causal_trace",
]
