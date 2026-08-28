from __future__ import annotations

import importlib.metadata

import pytest

pytestmark = pytest.mark.integration

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from kedit_audit.adapters import (
    SUPPORTED_TORCH_VERSION,
    SUPPORTED_TRANSFORMERS_VERSION,
    GPT2CausalLMAdapter,
    ModelMetadata,
    TokenSpan,
    resolve_module_path,
)


@pytest.fixture(autouse=True)
def deterministic_single_thread_cpu() -> object:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def _tokenizer() -> object:
    vocabulary = {
        "[UNK]": 0,
        "alpha": 1,
        "beta": 2,
        "gamma": 3,
        "delta": 4,
    }
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
    )


def test_pinned_gpt2_adapter_scores_one_cpu_target_without_download() -> None:
    assert importlib.metadata.version("transformers") == SUPPORTED_TRANSFORMERS_VERSION
    assert importlib.metadata.version("torch") == SUPPORTED_TORCH_VERSION

    torch.manual_seed(17)
    tokenizer = _tokenizer()
    config = transformers.GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=16,
        n_ctx=16,
        n_embd=8,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = transformers.GPT2LMHeadModel(config).to(device="cpu", dtype=torch.float32)
    adapter = GPT2CausalLMAdapter(
        model=model,
        tokenizer=tokenizer,
        metadata=ModelMetadata(
            model_id="local/tiny-random-gpt2",
            model_revision="local-config-v1",
            tokenizer_id="local/word-level-fast",
            tokenizer_revision="local-vocab-v1",
            state_id="baseline-local-gpt2",
            state_kind="baseline",
            device="cpu",
            dtype="float32",
        ),
    )

    prompt = "alpha beta gamma"
    target = " delta"
    result = adapter.score_target(prompt, target)
    combined_ids = tokenizer.encode(prompt + target, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([combined_ids], dtype=torch.long),
            use_cache=False,
            return_dict=True,
        )
    expected = torch.log_softmax(output.logits[0, len(prompt_ids) - 1], dim=-1)[
        combined_ids[-1]
    ].item()

    assert result.target_token_ids == (4,)
    assert result.token_log_probabilities == pytest.approx((expected,))
    assert adapter.subject_token_span(prompt, "beta") == TokenSpan(1, 2)
    assert resolve_module_path(adapter.module_root, "transformer.h.0.mlp") is model.transformer.h[0].mlp
