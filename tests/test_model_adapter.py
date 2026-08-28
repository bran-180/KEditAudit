import math

import pytest

from kedit_audit.adapters import (
    AdapterInputError,
    AdapterPairValidationError,
    FakeModelAdapter,
    ModelAdapter,
    ModelMetadata,
    TokenSpan,
    validate_adapter_pair,
)


def _metadata(state_id: str, state_kind: str) -> ModelMetadata:
    return ModelMetadata(
        model_id="synthetic/tiny-causal-lm",
        model_revision="model-revision-1",
        tokenizer_id="synthetic/tiny-tokenizer",
        tokenizer_revision="tokenizer-revision-1",
        state_id=state_id,
        state_kind=state_kind,
        device="cpu",
        dtype="float32",
    )


def _adapter(state_id: str = "baseline-state", state_kind: str = "baseline") -> FakeModelAdapter:
    return FakeModelAdapter(
        metadata=_metadata(state_id, state_kind),
        token_ids_by_text={
            "Paris is in France": [10, 20, 30, 40],
            "France": [40],
            " Italy": [1, 0],
        },
        logits_by_prompt_target={
            ("Paris is in France", " Italy"): [
                [math.log(0.1), math.log(0.9)],
                [math.log(0.8), math.log(0.2)],
            ]
        },
        module_root={"fixture": "root"},
    )


def test_fake_adapter_satisfies_runtime_protocol_and_scores_supplied_logits() -> None:
    adapter = _adapter()

    assert isinstance(adapter, ModelAdapter)
    assert adapter.tokenize("Paris is in France") == (10, 20, 30, 40)
    result = adapter.score_target("Paris is in France", " Italy")

    assert result.target_token_ids == (1, 0)
    assert result.token_log_probabilities == pytest.approx((math.log(0.9), math.log(0.8)))
    assert result.mean_log_probability == pytest.approx((math.log(0.9) + math.log(0.8)) / 2)
    assert adapter.module_root == {"fixture": "root"}


def test_fake_adapter_copies_fixture_inputs() -> None:
    prompt_ids = [1, 2]
    target_ids = [0]
    logits = [[0.0, 1.0]]
    adapter = FakeModelAdapter(
        metadata=_metadata("copied-state", "baseline"),
        token_ids_by_text={"prompt": prompt_ids, " target": target_ids},
        logits_by_prompt_target={("prompt", " target"): logits},
        module_root=object(),
    )

    prompt_ids[0] = 999
    target_ids[0] = 999
    logits[0][0] = 999.0

    assert adapter.tokenize("prompt") == (1, 2)
    assert adapter.score_target("prompt", " target").target_token_ids == (0,)


def test_subject_span_is_derived_from_tokenizer_output() -> None:
    adapter = _adapter()

    assert adapter.subject_token_span("Paris is in France", "France") == TokenSpan(3, 4)
    assert adapter.subject_token_span("Paris is in France", "France").token_indices == (3,)


@pytest.mark.parametrize(
    ("prompt", "subject", "message"),
    [
        ("unknown prompt", "France", "no token fixture"),
        ("Paris is in France", "unknown subject", "no token fixture"),
    ],
)
def test_fake_adapter_fails_closed_for_unknown_text(
    prompt: str,
    subject: str,
    message: str,
) -> None:
    with pytest.raises(AdapterInputError, match=message):
        _adapter().subject_token_span(prompt, subject)


def test_subject_span_rejects_ambiguous_token_matches() -> None:
    adapter = FakeModelAdapter(
        metadata=_metadata("ambiguous-state", "baseline"),
        token_ids_by_text={"A A": [7, 7], "A": [7]},
        logits_by_prompt_target={},
        module_root=object(),
    )

    with pytest.raises(AdapterInputError, match="appears 2 times"):
        adapter.subject_token_span("A A", "A")


def test_adapter_pair_requires_distinct_compatible_states() -> None:
    baseline = _adapter("baseline-state", "baseline")
    edited = _adapter("edited-state", "edited")

    validate_adapter_pair(baseline, edited)

    with pytest.raises(AdapterPairValidationError, match="state_id values must differ"):
        validate_adapter_pair(baseline, _adapter("baseline-state", "edited"))

    incompatible = FakeModelAdapter(
        metadata=ModelMetadata(
            model_id="synthetic/tiny-causal-lm",
            model_revision="model-revision-1",
            tokenizer_id="other/tokenizer",
            tokenizer_revision="tokenizer-revision-2",
            state_id="other-state",
            state_kind="edited",
            device="cpu",
            dtype="float32",
        ),
        token_ids_by_text={},
        logits_by_prompt_target={},
        module_root=object(),
    )
    with pytest.raises(AdapterPairValidationError, match="tokenizer identity"):
        validate_adapter_pair(baseline, incompatible)


def test_metadata_rejects_blank_identity_fields_and_invalid_state_kind() -> None:
    with pytest.raises(ValueError, match="model_id must not be blank"):
        _metadata("state", "baseline").__class__(
            model_id=" ",
            model_revision="revision",
            tokenizer_id="tokenizer",
            tokenizer_revision="revision",
            state_id="state",
            state_kind="baseline",
            device="cpu",
            dtype="float32",
        )

    with pytest.raises(ValueError, match="state_kind"):
        _metadata("state", "unknown")
