"""Minimal model-adapter protocol plus a deterministic offline fake."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from kedit_audit.metrics import SequenceLogProbability, target_sequence_log_probability

ModelStateKind = Literal["baseline", "edited"]


class AdapterInputError(ValueError):
    """Raised when an adapter cannot safely interpret a requested input."""


class AdapterPairValidationError(ValueError):
    """Raised when baseline and edited adapters are not a comparable pair."""


@dataclass(frozen=True)
class ModelMetadata:
    """Stable identity and execution metadata for one logical model state."""

    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    state_id: str
    state_kind: ModelStateKind
    device: str
    dtype: str

    def __post_init__(self) -> None:
        for field_name in (
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "state_id",
            "device",
            "dtype",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.state_kind not in {"baseline", "edited"}:
            raise ValueError("state_kind must be 'baseline' or 'edited'")


@dataclass(frozen=True)
class TokenSpan:
    """Half-open token span derived by a model adapter's tokenizer."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int) or self.start < 0:
            raise ValueError("start must be a non-negative integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int) or self.end <= self.start:
            raise ValueError("end must be an integer greater than start")

    @property
    def token_indices(self) -> tuple[int, ...]:
        """Return every token index covered by the half-open span."""

        return tuple(range(self.start, self.end))


@runtime_checkable
class ModelAdapter(Protocol):
    """Editor-independent operations required by the audit and tracing layers."""

    @property
    def metadata(self) -> ModelMetadata:
        """Return immutable identity and execution metadata."""

    @property
    def module_root(self) -> object:
        """Return the root used by the tested module-path resolver."""

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize text without silently changing the adapter configuration."""

    def score_target(self, prompt: str, target: str) -> SequenceLogProbability:
        """Return normalized target-sequence evidence for a prompt."""

    def subject_token_span(self, prompt: str, subject: str) -> TokenSpan:
        """Locate one unambiguous subject span using this adapter's tokenizer."""


class FakeModelAdapter:
    """Deterministic adapter backed only by explicitly supplied fixture values."""

    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        token_ids_by_text: Mapping[str, Sequence[int]],
        logits_by_prompt_target: Mapping[
            tuple[str, str],
            Sequence[Sequence[float]],
        ],
        module_root: object,
    ) -> None:
        self._metadata = metadata
        self._module_root = module_root
        self._token_ids_by_text = {
            text: _copy_token_ids(token_ids, text=text)
            for text, token_ids in token_ids_by_text.items()
        }
        self._logits_by_prompt_target = {
            request: _copy_logits(logits, request=request)
            for request, logits in logits_by_prompt_target.items()
        }

    @property
    def metadata(self) -> ModelMetadata:
        """Return fixture metadata."""

        return self._metadata

    @property
    def module_root(self) -> object:
        """Return the fixture's module root without interpreting it."""

        return self._module_root

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Return copied fixture token IDs or fail closed for unknown text."""

        try:
            return self._token_ids_by_text[text]
        except KeyError as error:
            raise AdapterInputError("no token fixture exists for the supplied text") from error

    def score_target(self, prompt: str, target: str) -> SequenceLogProbability:
        """Score one fixture pair using the package's authoritative reducer."""

        self.tokenize(prompt)
        target_token_ids = self.tokenize(target)
        try:
            logits = self._logits_by_prompt_target[(prompt, target)]
        except KeyError as error:
            raise AdapterInputError(
                "no logits fixture exists for the supplied prompt/target pair"
            ) from error
        return target_sequence_log_probability(logits, target_token_ids)

    def subject_token_span(self, prompt: str, subject: str) -> TokenSpan:
        """Find exactly one token-subsequence match for the supplied subject."""

        prompt_ids = self.tokenize(prompt)
        subject_ids = self.tokenize(subject)
        width = len(subject_ids)
        locations = [
            start
            for start in range(len(prompt_ids) - width + 1)
            if prompt_ids[start : start + width] == subject_ids
        ]
        if not locations:
            raise AdapterInputError("the tokenized subject does not appear in the tokenized prompt")
        if len(locations) != 1:
            raise AdapterInputError(
                f"the tokenized subject appears {len(locations)} times in the tokenized prompt"
            )
        return TokenSpan(locations[0], locations[0] + width)


def validate_adapter_pair(baseline: ModelAdapter, edited: ModelAdapter) -> None:
    """Require distinct states with the same model and tokenizer identities."""

    baseline_metadata = baseline.metadata
    edited_metadata = edited.metadata
    if baseline_metadata.state_kind != "baseline":
        raise AdapterPairValidationError("the baseline adapter must declare state_kind 'baseline'")
    if edited_metadata.state_kind != "edited":
        raise AdapterPairValidationError("the edited adapter must declare state_kind 'edited'")
    if baseline_metadata.state_id == edited_metadata.state_id:
        raise AdapterPairValidationError("baseline and edited state_id values must differ")
    if (
        baseline_metadata.model_id,
        baseline_metadata.model_revision,
    ) != (
        edited_metadata.model_id,
        edited_metadata.model_revision,
    ):
        raise AdapterPairValidationError("baseline and edited model identity must match")
    if (
        baseline_metadata.tokenizer_id,
        baseline_metadata.tokenizer_revision,
    ) != (
        edited_metadata.tokenizer_id,
        edited_metadata.tokenizer_revision,
    ):
        raise AdapterPairValidationError("baseline and edited tokenizer identity must match")


def _copy_token_ids(token_ids: Sequence[int], *, text: str) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)) or not token_ids:
        raise ValueError(f"token fixture for {text!r} must contain at least one token ID")
    copied: list[int] = []
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(
                f"token fixture for {text!r} has an invalid token ID at index {index}"
            )
        copied.append(token_id)
    return tuple(copied)


def _copy_logits(
    logits: Sequence[Sequence[float]],
    *,
    request: tuple[str, str],
) -> tuple[tuple[float, ...], ...]:
    if isinstance(logits, (str, bytes)):
        raise TypeError(f"logits fixture for {request!r} must be a sequence of rows")
    copied: list[tuple[float, ...]] = []
    for row_index, row in enumerate(logits):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"logits fixture row {row_index} for {request!r} is invalid")
        copied.append(tuple(row))
    return tuple(copied)
