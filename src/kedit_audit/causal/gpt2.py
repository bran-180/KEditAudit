"""Pinned GPT-2 activation adapter for deterministic causal tracing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from kedit_audit.adapters import (
    AdapterCompatibilityError,
    AdapterInputError,
    GPT2CausalLMAdapter,
    ModelMetadata,
    TokenSpan,
    resolve_module_path,
)
from kedit_audit.adapters.transformers import _load_torch
from kedit_audit.causal.hooks import ForwardHook, ForwardHookModule, HookManager
from kedit_audit.causal.tracer import CleanTraceRun

GPT2_CORRUPTION_STANDARD_DEVIATION = 1.0
_GPT2_BLOCK_MODULE = "transformers.models.gpt2.modeling_gpt2"
_GPT2_BLOCK_PATH = re.compile(r"^transformer\.h\.(0|[1-9][0-9]*)$")


class _TraceTensor(Protocol):
    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def device(self) -> object: ...

    @property
    def dtype(self) -> object: ...

    def __getitem__(self, key: object) -> _TraceTensor: ...

    def __setitem__(self, key: object, value: object) -> None: ...

    def __add__(self, other: object) -> _TraceTensor: ...

    def __mul__(self, other: object) -> _TraceTensor: ...

    def clone(self) -> _TraceTensor: ...

    def detach(self) -> _TraceTensor: ...


class _Generator(Protocol):
    def manual_seed(self, seed: int) -> _Generator: ...


class _TraceTorch(Protocol):
    float32: object

    def Generator(self, *, device: str) -> _Generator: ...

    def is_tensor(self, value: object) -> bool: ...

    def randn(
        self,
        size: tuple[int, int, int],
        *,
        generator: _Generator,
        dtype: object,
        device: str,
    ) -> _TraceTensor: ...


@dataclass(frozen=True, eq=False, repr=False)
class _GPT2EmbeddingCorruption:
    owner: object
    prompt_token_ids: tuple[int, ...]
    subject_span: TokenSpan
    seed: int
    noise: _TraceTensor


class GPT2CausalTraceAdapter(GPT2CausalLMAdapter):
    """Run a pinned GPT-2 clean/corrupt/restore experiment on block outputs.

    Corruption is fixed unit-standard-deviation Gaussian noise added to the
    subject's token embeddings. Restoration replaces the same subject token
    positions at one supported GPT-2 block with their clean hidden states.
    """

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        metadata: ModelMetadata,
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, metadata=metadata)
        self._corruption_owner = object()

    def run_clean(
        self,
        *,
        prompt: str,
        target: str,
        module_paths: tuple[str, ...],
    ) -> CleanTraceRun:
        """Score clean input and capture one detached hidden-state tensor per block."""

        if not isinstance(module_paths, tuple) or not module_paths:
            raise AdapterInputError("module_paths must be a non-empty tuple")
        if len(set(module_paths)) != len(module_paths):
            raise AdapterInputError("module_paths must be unique")
        modules = tuple(self._resolve_trace_block(path) for path in module_paths)
        activations: dict[str, _TraceTensor] = {}
        torch = _trace_torch()

        with HookManager() as hooks:
            for module_path, module in zip(module_paths, modules, strict=True):
                hooks.register_forward_hook(
                    cast(ForwardHookModule, module),
                    self._capture_hook(
                        module_path=module_path,
                        activations=activations,
                        torch=torch,
                    ),
                )
            result = self.score_target(prompt, target)

        missing = tuple(path for path in module_paths if path not in activations)
        if missing:
            raise AdapterCompatibilityError(
                f"GPT-2 forward did not emit activations for {', '.join(missing)}"
            )
        return CleanTraceRun(
            target_score=result.sum_log_probability,
            activations=activations,
        )

    def create_corruption(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        subject_span: TokenSpan,
        seed: int,
    ) -> object:
        """Create deterministic local-generator noise without changing global RNG state."""

        _validate_prompt_token_ids(prompt_token_ids)
        if len(prompt_token_ids) > self._maximum_positions:
            raise AdapterInputError(
                f"prompt exceeds model limit {self._maximum_positions} tokens"
            )
        if subject_span.end > len(prompt_token_ids):
            raise AdapterInputError("subject token span exceeds the tokenized prompt length")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
            raise AdapterInputError("seed must be an integer in [0, 4294967295]")
        hidden_size = getattr(self._model.config, "n_embd", None)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise AdapterCompatibilityError("model config.n_embd must be a positive integer")

        torch = _trace_torch()
        _require_single_thread(torch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = (
            torch.randn(
                (1, subject_span.end - subject_span.start, hidden_size),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            * GPT2_CORRUPTION_STANDARD_DEVIATION
        )
        return _GPT2EmbeddingCorruption(
            owner=self._corruption_owner,
            prompt_token_ids=prompt_token_ids,
            subject_span=subject_span,
            seed=seed,
            noise=noise,
        )

    def run_corrupted(
        self,
        *,
        prompt: str,
        target: str,
        corruption: object,
    ) -> float:
        """Score target after applying the supplied fixed embedding corruption."""

        normalized = self._validate_corruption(prompt, corruption)
        embedding = resolve_module_path(self.module_root, "transformer.wte")
        torch = _trace_torch()
        with HookManager() as hooks:
            hooks.register_forward_hook(
                cast(ForwardHookModule, embedding),
                self._corruption_hook(normalized, torch=torch),
            )
            return self.score_target(prompt, target).sum_log_probability

    def run_restored(
        self,
        *,
        prompt: str,
        target: str,
        corruption: object,
        module_path: str,
        clean_activation: object,
    ) -> float:
        """Score with fixed embedding noise and one clean subject-state restoration."""

        normalized = self._validate_corruption(prompt, corruption)
        traced_module = self._resolve_trace_block(module_path)
        embedding = resolve_module_path(self.module_root, "transformer.wte")
        torch = _trace_torch()
        clean_tensor = _require_tensor(
            clean_activation,
            torch=torch,
            name="clean activation",
        )
        with HookManager() as hooks:
            hooks.register_forward_hook(
                cast(ForwardHookModule, embedding),
                self._corruption_hook(normalized, torch=torch),
            )
            hooks.register_forward_hook(
                cast(ForwardHookModule, traced_module),
                self._restoration_hook(
                    clean_tensor=clean_tensor,
                    subject_span=normalized.subject_span,
                    torch=torch,
                ),
            )
            return self.score_target(prompt, target).sum_log_probability

    def _resolve_trace_block(self, module_path: str) -> object:
        if not isinstance(module_path, str) or _GPT2_BLOCK_PATH.fullmatch(module_path) is None:
            raise AdapterCompatibilityError(
                "GPT-2 causal tracing supports only paths shaped like 'transformer.h.<index>'"
            )
        module = resolve_module_path(self.module_root, module_path)
        module_class = type(module)
        if (
            module_class.__name__ != "GPT2Block"
            or module_class.__module__ != _GPT2_BLOCK_MODULE
        ):
            raise AdapterCompatibilityError(
                f"module {module_path!r} must resolve to an exact Transformers GPT2Block"
            )
        if not callable(getattr(module, "register_forward_hook", None)):
            raise AdapterCompatibilityError(
                f"module {module_path!r} does not support forward hooks"
            )
        return module

    def _validate_corruption(
        self,
        prompt: str,
        corruption: object,
    ) -> _GPT2EmbeddingCorruption:
        if (
            not isinstance(corruption, _GPT2EmbeddingCorruption)
            or corruption.owner is not self._corruption_owner
        ):
            raise AdapterInputError("corruption was not created by this GPT-2 trace adapter")
        if self.tokenize(prompt) != corruption.prompt_token_ids:
            raise AdapterInputError("corruption prompt token IDs do not match the supplied prompt")
        return corruption

    @staticmethod
    def _capture_hook(
        *,
        module_path: str,
        activations: dict[str, _TraceTensor],
        torch: _TraceTorch,
    ) -> ForwardHook:
        def capture(
            _module: object,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            tensor = _primary_tensor(output, torch=torch, name=f"activation for {module_path}")
            activations[module_path] = tensor.detach().clone()

        return capture

    @staticmethod
    def _corruption_hook(
        corruption: _GPT2EmbeddingCorruption,
        *,
        torch: _TraceTorch,
    ) -> ForwardHook:
        def corrupt(
            _module: object,
            _inputs: tuple[object, ...],
            output: object,
        ) -> object:
            tensor = _primary_tensor(output, torch=torch, name="GPT-2 token embedding output")
            _validate_hidden_tensor(
                tensor,
                name="GPT-2 token embedding output",
                minimum_sequence_length=len(corruption.prompt_token_ids),
            )
            noise_shape = tuple(corruption.noise.shape)
            expected_noise_shape = (
                1,
                corruption.subject_span.end - corruption.subject_span.start,
                tensor.shape[2],
            )
            if noise_shape != expected_noise_shape:
                raise AdapterCompatibilityError(
                    f"corruption noise shape {noise_shape!r} must equal {expected_noise_shape!r}"
                )
            replacement = tensor.clone()
            span = slice(corruption.subject_span.start, corruption.subject_span.end)
            replacement[(slice(None), span, slice(None))] = (
                tensor[(slice(None), span, slice(None))] + corruption.noise
            )
            return _replace_primary_tensor(output, replacement, torch=torch)

        return corrupt

    @staticmethod
    def _restoration_hook(
        *,
        clean_tensor: _TraceTensor,
        subject_span: TokenSpan,
        torch: _TraceTorch,
    ) -> ForwardHook:
        def restore(
            _module: object,
            _inputs: tuple[object, ...],
            output: object,
        ) -> object:
            current = _primary_tensor(output, torch=torch, name="corrupted block activation")
            _validate_hidden_tensor(
                current,
                name="corrupted block activation",
                minimum_sequence_length=subject_span.end,
            )
            if tuple(clean_tensor.shape) != tuple(current.shape):
                raise AdapterCompatibilityError(
                    "clean activation shape must equal the corrupted block activation shape; "
                    f"received {tuple(clean_tensor.shape)!r} and {tuple(current.shape)!r}"
                )
            if clean_tensor.device != current.device or clean_tensor.dtype != current.dtype:
                raise AdapterCompatibilityError(
                    "clean activation device and dtype must match the corrupted block activation"
                )
            replacement = current.clone()
            span = slice(subject_span.start, subject_span.end)
            replacement[(slice(None), span, slice(None))] = clean_tensor[
                (slice(None), span, slice(None))
            ]
            return _replace_primary_tensor(output, replacement, torch=torch)

        return restore


def _trace_torch() -> _TraceTorch:
    return cast(_TraceTorch, _load_torch())


def _require_single_thread(torch: _TraceTorch) -> None:
    get_num_threads = getattr(torch, "get_num_threads", None)
    if not callable(get_num_threads) or get_num_threads() != 1:
        raise AdapterCompatibilityError(
            "deterministic CPU tracing requires torch.set_num_threads(1) before adapter use"
        )


def _validate_prompt_token_ids(prompt_token_ids: object) -> None:
    if not isinstance(prompt_token_ids, tuple) or not prompt_token_ids:
        raise AdapterInputError("prompt_token_ids must be a non-empty tuple")
    for index, token_id in enumerate(prompt_token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise AdapterInputError(
                f"prompt_token_ids[{index}] must be a non-negative integer"
            )


def _require_tensor(
    value: object,
    *,
    torch: _TraceTorch,
    name: str,
) -> _TraceTensor:
    if not torch.is_tensor(value):
        raise AdapterCompatibilityError(f"{name} must be a Torch tensor")
    return cast(_TraceTensor, value)


def _primary_tensor(
    output: object,
    *,
    torch: _TraceTorch,
    name: str,
) -> _TraceTensor:
    if torch.is_tensor(output):
        return cast(_TraceTensor, output)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return cast(_TraceTensor, output[0])
    raise AdapterCompatibilityError(
        f"{name} must be a tensor or a tuple whose first item is a tensor"
    )


def _replace_primary_tensor(
    output: object,
    replacement: _TraceTensor,
    *,
    torch: _TraceTorch,
) -> object:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (replacement, *output[1:])
    raise AdapterCompatibilityError(
        "GPT-2 activation output must be a tensor or a tuple whose first item is a tensor"
    )


def _validate_hidden_tensor(
    tensor: _TraceTensor,
    *,
    name: str,
    minimum_sequence_length: int,
) -> None:
    shape = tuple(tensor.shape)
    if tensor.ndim != 3 or len(shape) != 3 or shape[0] != 1:
        raise AdapterCompatibilityError(f"{name} must have shape (1, sequence, hidden)")
    if shape[1] < minimum_sequence_length:
        raise AdapterCompatibilityError(
            f"{name} sequence length must be at least {minimum_sequence_length}"
        )
    if shape[2] <= 0:
        raise AdapterCompatibilityError(f"{name} hidden dimension must be positive")
