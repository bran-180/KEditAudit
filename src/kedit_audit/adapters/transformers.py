"""Pinned, fail-closed adapter for one Transformers GPT-2 implementation."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Iterable, Mapping, Sequence, Sized
from contextlib import AbstractContextManager
from typing import Protocol, cast

from kedit_audit.adapters.model import AdapterInputError, ModelMetadata, TokenSpan
from kedit_audit.metrics import SequenceLogProbability, target_sequence_log_probability

SUPPORTED_TRANSFORMERS_VERSION = "5.16.1"
SUPPORTED_TORCH_VERSION = "2.13.0"
_GPT2_MODEL_MODULE = "transformers.models.gpt2.modeling_gpt2"
_MAX_TEXT_CHARACTERS = 4096
_MAX_TARGET_TOKENS = 64
_MAX_VOCABULARY_SIZE = 65536


class AdapterCompatibilityError(RuntimeError):
    """Raised when an optional dependency or supplied model is unsupported."""


class _Parameter(Protocol):
    @property
    def device(self) -> object: ...

    @property
    def dtype(self) -> object: ...


class _Model(Protocol):
    config: object
    training: bool

    def eval(self) -> object: ...

    def parameters(self) -> Iterable[_Parameter]: ...

    def __call__(self, **kwargs: object) -> object: ...


class _Tokenizer(Protocol):
    is_fast: bool

    def encode(self, text: str, *, add_special_tokens: bool) -> object: ...

    def __call__(self, text: str, **kwargs: object) -> object: ...


class _Tensor(Protocol):
    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> Sequence[int]: ...

    def __getitem__(self, key: object) -> _Tensor: ...

    def detach(self) -> _Tensor: ...

    def to(self, *, device: str, dtype: object) -> _Tensor: ...

    def tolist(self) -> object: ...


class _Torch(Protocol):
    long: object
    float64: object

    def tensor(self, data: object, *, dtype: object, device: str) -> _Tensor: ...

    def no_grad(self) -> AbstractContextManager[None]: ...

    def get_num_threads(self) -> int: ...


class GPT2CausalLMAdapter:
    """Adapt one preloaded, pinned Transformers GPT-2 causal LM for CPU audit scoring.

    The constructor deliberately does not accept a model ID or call
    ``from_pretrained``. Loading policy, checkpoint provenance, and any remote
    access remain the caller's responsibility.
    """

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        metadata: ModelMetadata,
    ) -> None:
        _require_distribution_version("transformers", SUPPORTED_TRANSFORMERS_VERSION)
        _require_distribution_version("torch", SUPPORTED_TORCH_VERSION)
        if metadata.device != "cpu":
            raise AdapterCompatibilityError("the pinned GPT-2 adapter supports only device 'cpu'")
        if metadata.dtype != "float32":
            raise AdapterCompatibilityError("the pinned GPT-2 adapter supports only dtype 'float32'")
        model_class = type(model)
        if (
            model_class.__name__ != "GPT2LMHeadModel"
            or model_class.__module__ != _GPT2_MODEL_MODULE
        ):
            raise AdapterCompatibilityError(
                "model must be an exact Transformers GPT2LMHeadModel instance"
            )

        normalized_model = cast(_Model, model)
        config = normalized_model.config
        if getattr(config, "model_type", None) != "gpt2":
            raise AdapterCompatibilityError("model config.model_type must be 'gpt2'")

        normalized_tokenizer = cast(_Tokenizer, tokenizer)
        if getattr(normalized_tokenizer, "is_fast", False) is not True:
            raise AdapterCompatibilityError(
                "tokenizer must be a fast tokenizer with offset-mapping support"
            )
        tokenizer_size = _tokenizer_size(tokenizer)
        vocabulary_size = getattr(config, "vocab_size", None)
        if (
            isinstance(vocabulary_size, bool)
            or not isinstance(vocabulary_size, int)
            or vocabulary_size <= 0
        ):
            raise AdapterCompatibilityError("model config.vocab_size must be a positive integer")
        if tokenizer_size != vocabulary_size:
            raise AdapterCompatibilityError(
                f"tokenizer size {tokenizer_size} must equal model vocabulary size {vocabulary_size}"
            )
        if vocabulary_size > _MAX_VOCABULARY_SIZE:
            raise AdapterCompatibilityError(
                f"model vocabulary size must not exceed {_MAX_VOCABULARY_SIZE}"
            )
        maximum_positions = getattr(config, "n_positions", None)
        if (
            isinstance(maximum_positions, bool)
            or not isinstance(maximum_positions, int)
            or maximum_positions <= 0
        ):
            raise AdapterCompatibilityError("model config.n_positions must be a positive integer")

        parameter = _first_parameter(normalized_model)
        actual_device = str(parameter.device)
        actual_dtype = str(parameter.dtype).removeprefix("torch.")
        if actual_device != metadata.device:
            raise AdapterCompatibilityError(
                f"model parameter device {actual_device!r} does not match metadata.device "
                f"{metadata.device!r}"
            )
        if actual_dtype != metadata.dtype:
            raise AdapterCompatibilityError(
                f"model parameter dtype {actual_dtype!r} does not match metadata.dtype "
                f"{metadata.dtype!r}"
            )

        normalized_model.eval()
        if normalized_model.training:
            raise AdapterCompatibilityError("model.eval() did not disable training mode")

        self._model = normalized_model
        self._tokenizer = normalized_tokenizer
        self._metadata = metadata
        self._maximum_positions = maximum_positions

    @property
    def metadata(self) -> ModelMetadata:
        """Return the caller-supplied immutable provenance for this model state."""

        return self._metadata

    @property
    def module_root(self) -> object:
        """Return the verified GPT-2 model for safe dotted-module resolution."""

        return self._model

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize text without adding model-specific special tokens."""

        if not isinstance(text, str):
            raise AdapterInputError("text must be a string")
        if len(text) > _MAX_TEXT_CHARACTERS:
            raise AdapterInputError(
                f"text must not exceed {_MAX_TEXT_CHARACTERS} characters"
            )
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        return _token_ids(encoded, path="tokenizer.encode output")

    def score_target(self, prompt: str, target: str) -> SequenceLogProbability:
        """Score the contextual target suffix with aligned causal-LM logits."""

        if not isinstance(prompt, str) or not prompt:
            raise AdapterInputError("prompt must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise AdapterInputError("target must be a non-empty string")

        prompt_token_ids = self.tokenize(prompt)
        if not prompt_token_ids:
            raise AdapterInputError("prompt must produce at least one token")
        combined_token_ids = self.tokenize(prompt + target)
        if len(combined_token_ids) <= len(prompt_token_ids):
            raise AdapterInputError("target must add at least one contextual token")
        if combined_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
            raise AdapterInputError(
                "target changes the prompt tokenization boundary; supply a target with an "
                "unambiguous leading boundary"
            )
        target_token_ids = combined_token_ids[len(prompt_token_ids) :]
        if len(combined_token_ids) > self._maximum_positions:
            raise AdapterInputError(
                f"combined prompt and target exceed model limit {self._maximum_positions} tokens"
            )
        if len(target_token_ids) > _MAX_TARGET_TOKENS:
            raise AdapterInputError(
                f"target must not exceed {_MAX_TARGET_TOKENS} contextual tokens"
            )

        torch = _load_torch()
        if torch.get_num_threads() != 1:
            raise AdapterCompatibilityError(
                "deterministic CPU scoring requires torch.set_num_threads(1) before adapter use"
            )
        input_ids = torch.tensor(
            [list(combined_token_ids)],
            dtype=torch.long,
            device="cpu",
        )
        with torch.no_grad():
            output = self._model(input_ids=input_ids, use_cache=False, return_dict=True)
        logits = cast(_Tensor, getattr(output, "logits", None))
        _validate_logits_shape(
            logits,
            sequence_length=len(combined_token_ids),
            vocabulary_size=_tokenizer_size(self._tokenizer),
        )
        aligned = logits[
            (
                0,
                slice(len(prompt_token_ids) - 1, len(combined_token_ids) - 1),
                slice(None),
            )
        ]
        raw_rows = aligned.detach().to(device="cpu", dtype=torch.float64).tolist()
        rows = _logit_rows(raw_rows)
        return target_sequence_log_probability(rows, target_token_ids)

    def subject_token_span(self, prompt: str, subject: str) -> TokenSpan:
        """Map one unambiguous character occurrence to tokenizer offsets."""

        if not isinstance(prompt, str) or not prompt:
            raise AdapterInputError("prompt must be a non-empty string")
        if not isinstance(subject, str) or not subject:
            raise AdapterInputError("subject must be a non-empty string")
        if len(prompt) > _MAX_TEXT_CHARACTERS or len(subject) > _MAX_TEXT_CHARACTERS:
            raise AdapterInputError(
                f"prompt and subject must not exceed {_MAX_TEXT_CHARACTERS} characters"
            )
        character_starts = _substring_starts(prompt, subject)
        if not character_starts:
            raise AdapterInputError("subject does not appear in the prompt")
        if len(character_starts) != 1:
            raise AdapterInputError(
                f"subject appears {len(character_starts)} times in the prompt"
            )
        character_start = character_starts[0]
        character_end = character_start + len(subject)

        encoded = self._tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if not isinstance(encoded, Mapping):
            raise AdapterInputError("tokenizer output must be a mapping")
        token_ids = _token_ids(encoded.get("input_ids"), path="tokenizer input_ids")
        offsets = _offsets(encoded.get("offset_mapping"))
        if len(token_ids) != len(offsets):
            raise AdapterInputError("token IDs and offset mappings must have equal length")

        covered_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > character_start and start < character_end
        ]
        if not covered_indices:
            raise AdapterInputError("tokenizer offsets do not cover the subject")
        if covered_indices != list(range(covered_indices[0], covered_indices[-1] + 1)):
            raise AdapterInputError("subject token offsets are not contiguous")
        _require_character_coverage(
            offsets,
            covered_indices,
            character_start=character_start,
            character_end=character_end,
        )
        return TokenSpan(covered_indices[0], covered_indices[-1] + 1)


def _require_distribution_version(distribution: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise AdapterCompatibilityError(
            f"optional dependency {distribution}=={expected} is required"
        ) from error
    if actual != expected:
        raise AdapterCompatibilityError(
            f"unsupported {distribution} version {actual!r}; expected exactly {expected!r}"
        )


def _tokenizer_size(tokenizer: object) -> int:
    if not isinstance(tokenizer, Sized):
        raise AdapterCompatibilityError("tokenizer must expose a vocabulary size through len()")
    size = len(tokenizer)
    if isinstance(size, bool) or size <= 0:
        raise AdapterCompatibilityError("tokenizer vocabulary size must be a positive integer")
    return size


def _first_parameter(model: _Model) -> _Parameter:
    try:
        parameter = next(iter(model.parameters()))
    except (StopIteration, TypeError) as error:
        raise AdapterCompatibilityError("model must expose at least one parameter") from error
    if not hasattr(parameter, "device") or not hasattr(parameter, "dtype"):
        raise AdapterCompatibilityError("model parameters must expose device and dtype")
    return parameter


def _load_torch() -> _Torch:
    try:
        module = importlib.import_module("torch")
    except ImportError as error:
        raise AdapterCompatibilityError(
            f"optional dependency torch=={SUPPORTED_TORCH_VERSION} is required"
        ) from error
    return cast(_Torch, module)


def _token_ids(value: object, *, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AdapterInputError(f"{path} must be a sequence of token IDs")
    normalized: list[int] = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise AdapterInputError(f"{path}[{index}] must be a non-negative integer")
        normalized.append(token_id)
    return tuple(normalized)


def _offsets(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AdapterInputError("tokenizer offset_mapping must be a sequence")
    normalized: list[tuple[int, int]] = []
    for index, offset in enumerate(value):
        if (
            not isinstance(offset, Sequence)
            or isinstance(offset, (str, bytes))
            or len(offset) != 2
        ):
            raise AdapterInputError(f"tokenizer offset_mapping[{index}] must contain two integers")
        start, end = offset
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise AdapterInputError(
                f"tokenizer offset_mapping[{index}] must be an ordered non-negative pair"
            )
        normalized.append((start, end))
    return tuple(normalized)


def _substring_starts(text: str, substring: str) -> tuple[int, ...]:
    starts: list[int] = []
    start = text.find(substring)
    while start != -1:
        starts.append(start)
        start = text.find(substring, start + 1)
    return tuple(starts)


def _require_character_coverage(
    offsets: Sequence[tuple[int, int]],
    indices: Sequence[int],
    *,
    character_start: int,
    character_end: int,
) -> None:
    cursor = character_start
    for index in indices:
        start, end = offsets[index]
        clipped_start = max(start, character_start)
        clipped_end = min(end, character_end)
        if clipped_start > cursor:
            raise AdapterInputError("tokenizer offsets leave a gap inside the subject")
        cursor = max(cursor, clipped_end)
    if cursor < character_end:
        raise AdapterInputError("tokenizer offsets do not cover the complete subject")


def _validate_logits_shape(
    logits: _Tensor,
    *,
    sequence_length: int,
    vocabulary_size: int,
) -> None:
    if logits is None or not hasattr(logits, "ndim") or not hasattr(logits, "shape"):
        raise AdapterCompatibilityError("model output must expose a logits tensor")
    if logits.ndim != 3:
        raise AdapterCompatibilityError("model logits must have rank 3")
    shape = tuple(logits.shape)
    if shape != (1, sequence_length, vocabulary_size):
        raise AdapterCompatibilityError(
            "model logits shape must equal (1, sequence_length, vocabulary_size); "
            f"received {shape!r}"
        )


def _logit_rows(value: object) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AdapterCompatibilityError("aligned model logits must convert to a sequence of rows")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise AdapterCompatibilityError(
                f"aligned model logits row {row_index} must be a sequence"
            )
        try:
            rows.append(tuple(float(item) for item in row))
        except (TypeError, ValueError) as error:
            raise AdapterCompatibilityError(
                f"aligned model logits row {row_index} contains a non-numeric value"
            ) from error
    return tuple(rows)
