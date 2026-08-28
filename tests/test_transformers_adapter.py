from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import kedit_audit.adapters.transformers as adapter_module
from kedit_audit.adapters import (
    AdapterCompatibilityError,
    GPT2CausalLMAdapter,
    ModelAdapter,
    ModelMetadata,
    TokenSpan,
)
from kedit_audit.causal import GPT2CausalTraceAdapter


@dataclass
class FakeParameter:
    device: str = "cpu"
    dtype: str = "torch.float32"


class GPT2LMHeadModel:
    __module__ = "transformers.models.gpt2.modeling_gpt2"

    def __init__(
        self,
        *,
        model_type: str = "gpt2",
        vocabulary_size: int = 5,
        maximum_positions: int = 32,
    ) -> None:
        self.config = SimpleNamespace(
            model_type=model_type,
            vocab_size=vocabulary_size,
            n_positions=maximum_positions,
        )
        self.training = True
        self.eval_count = 0
        self.parameter = FakeParameter()

    def parameters(self) -> tuple[FakeParameter, ...]:
        return (self.parameter,)

    def eval(self) -> GPT2LMHeadModel:
        self.training = False
        self.eval_count += 1
        return self


class FakeFastTokenizer:
    is_fast = True

    def __init__(self, *, vocabulary_size: int = 5) -> None:
        self.vocabulary = {
            "[UNK]": 0,
            "alpha": 1,
            "beta": 2,
            "gamma": 3,
            "delta": 4,
        }
        self.vocabulary_size = vocabulary_size

    def __len__(self) -> int:
        return self.vocabulary_size

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.vocabulary.get(piece, 0) for piece in text.split()]

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"add_special_tokens": False, "return_offsets_mapping": True}
        pieces = text.split()
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for piece in pieces:
            start = text.index(piece, cursor)
            end = start + len(piece)
            offsets.append((start, end))
            cursor = end
        return {
            "input_ids": [self.vocabulary.get(piece, 0) for piece in pieces],
            "offset_mapping": offsets,
        }


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="local/tiny-random-gpt2",
        model_revision="local-config-v1",
        tokenizer_id="local/word-level-fast",
        tokenizer_revision="local-vocab-v1",
        state_id="baseline-local-gpt2",
        state_kind="baseline",
        device="cpu",
        dtype="float32",
    )


@pytest.fixture(autouse=True)
def exact_optional_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "transformers": adapter_module.SUPPORTED_TRANSFORMERS_VERSION,
        "torch": adapter_module.SUPPORTED_TORCH_VERSION,
    }
    monkeypatch.setattr(adapter_module.importlib.metadata, "version", versions.__getitem__)


def test_pinned_adapter_satisfies_protocol_without_importing_ml_libraries() -> None:
    model = GPT2LMHeadModel()
    adapter = GPT2CausalLMAdapter(
        model=model,
        tokenizer=FakeFastTokenizer(),
        metadata=_metadata(),
    )

    assert isinstance(adapter, ModelAdapter)
    assert adapter.metadata == _metadata()
    assert adapter.module_root is model
    assert model.eval_count == 1
    assert model.training is False
    assert adapter.tokenize("alpha beta") == (1, 2)
    assert adapter.subject_token_span("alpha beta gamma", "beta") == TokenSpan(1, 2)


@pytest.mark.parametrize("distribution", ["transformers", "torch"])
def test_pinned_adapter_rejects_any_other_dependency_version(
    monkeypatch: pytest.MonkeyPatch,
    distribution: str,
) -> None:
    def incompatible_version(name: str) -> str:
        if name == distribution:
            return "0.0.0"
        return {
            "transformers": adapter_module.SUPPORTED_TRANSFORMERS_VERSION,
            "torch": adapter_module.SUPPORTED_TORCH_VERSION,
        }[name]

    monkeypatch.setattr(adapter_module.importlib.metadata, "version", incompatible_version)

    with pytest.raises(AdapterCompatibilityError, match=f"unsupported {distribution} version"):
        GPT2CausalLMAdapter(
            model=GPT2LMHeadModel(),
            tokenizer=FakeFastTokenizer(),
            metadata=_metadata(),
        )


@pytest.mark.parametrize(
    ("model", "tokenizer", "message"),
    [
        (object(), FakeFastTokenizer(), "GPT2LMHeadModel"),
        (GPT2LMHeadModel(model_type="other"), FakeFastTokenizer(), "model_type"),
        (GPT2LMHeadModel(), SimpleNamespace(is_fast=False), "fast tokenizer"),
        (GPT2LMHeadModel(), FakeFastTokenizer(vocabulary_size=4), "tokenizer size"),
        (
            GPT2LMHeadModel(vocabulary_size=65537),
            FakeFastTokenizer(vocabulary_size=65537),
            "must not exceed",
        ),
    ],
)
def test_pinned_adapter_fails_closed_for_unsupported_components(
    model: object,
    tokenizer: object,
    message: str,
) -> None:
    with pytest.raises(AdapterCompatibilityError, match=message):
        GPT2CausalLMAdapter(model=model, tokenizer=tokenizer, metadata=_metadata())


def test_subject_span_uses_offsets_and_rejects_ambiguous_occurrences() -> None:
    adapter = GPT2CausalLMAdapter(
        model=GPT2LMHeadModel(),
        tokenizer=FakeFastTokenizer(),
        metadata=_metadata(),
    )

    with pytest.raises(ValueError, match="appears 2 times"):
        adapter.subject_token_span("beta alpha beta", "beta")


def test_scoring_rejects_multithreaded_torch_before_model_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MultiThreadTorch:
        @staticmethod
        def get_num_threads() -> int:
            return 8

    adapter = GPT2CausalLMAdapter(
        model=GPT2LMHeadModel(),
        tokenizer=FakeFastTokenizer(),
        metadata=_metadata(),
    )
    monkeypatch.setattr(adapter_module, "_load_torch", lambda: MultiThreadTorch())

    with pytest.raises(AdapterCompatibilityError, match=r"torch\.set_num_threads\(1\)"):
        adapter.score_target("alpha beta", " gamma")


def test_tokenization_rejects_oversized_untrusted_text() -> None:
    adapter = GPT2CausalLMAdapter(
        model=GPT2LMHeadModel(),
        tokenizer=FakeFastTokenizer(),
        metadata=_metadata(),
    )

    with pytest.raises(ValueError, match="4096 characters"):
        adapter.tokenize("x" * 4097)


def test_trace_adapter_rejects_non_block_path_before_loading_torch() -> None:
    adapter = GPT2CausalTraceAdapter(
        model=GPT2LMHeadModel(),
        tokenizer=FakeFastTokenizer(),
        metadata=_metadata(),
    )

    with pytest.raises(AdapterCompatibilityError, match="transformer.h.<index>"):
        adapter.run_clean(
            prompt="alpha beta",
            target=" gamma",
            module_paths=("transformer.h.0.mlp",),
        )
