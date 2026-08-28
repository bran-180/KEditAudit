# Metric definitions

This document describes metrics that are implemented in KEditAudit. Planned
metrics are not listed as available until their code and offline tests land.

## Target sequence log-probability

Status: implemented.

For a target token sequence \(y_{1:m}\), KEditAudit records the conditional
natural-log probability of every token and two reductions:

\[
\ell_i = \log P(y_i \mid x, y_{<i})
\]

\[
L_{sum} = \sum_{i=1}^{m}\ell_i,
\qquad
L_{mean} = \frac{1}{m}\sum_{i=1}^{m}\ell_i.
\]

`mean_log_probability` is the primary length-normalized score defined in the
[knowledge base](KNOWLEDGE_BASE.md#21-target-sequence-score). Both reductions
and every `token_log_probability` remain in the result. Values use natural
logarithms, so the unit is `nats`; higher, less-negative values indicate greater
model probability for that exact target sequence.

The official ROME CounterFact evaluator likewise applies `log_softmax` to each
target-token position and averages the negative values in its
[`test_batch_prediction`](https://github.com/kmeng01/rome/blob/main/experiments/py/eval_utils_counterfact.py#L97-L134)
implementation. That evaluator reports mean negative log-probability; KEditAudit
reports the sign-reversed mean log-probability and retains each token value. See
also the primary [ROME paper and project](https://rome.baulab.info/).

### Supplied-logits contract

`target_sequence_log_probability(logits, target_token_ids)` does not load a
model or tokenizer. Its input is already aligned:

- `logits[i]` contains the full next-token vocabulary logits for
  `target_token_ids[i]`;
- row `i` is conditioned on the prompt and target tokens before `i`;
- all rows use the same vocabulary and contain finite numeric values;
- target IDs use that vocabulary and are in range.

The reducer performs no causal shift, tokenization, masking, truncation,
temperature scaling, or padding removal. An adapter must perform those steps
and record relevant configuration before calling it. This separation makes an
off-by-one alignment error visible at the adapter boundary instead of hiding
model assumptions inside the metric.

For row logits \(z\), the selected token score is computed with stable
log-sum-exp:

\[
\log P(y) = z_y - a - \log\sum_j \exp(z_j-a),
\qquad a=\max_j z_j.
\]

The implementation depends only on the Python standard library. It rejects
empty or ragged inputs, row/token-count mismatches, non-integer or out-of-range
token IDs, NaN/Infinity, and results outside the finite float range. Errors name
the relevant row, column, or token position.

### Result contract

`SequenceLogProbability.as_dict()` produces a JSON-serializable object with:

- `target_token_ids`;
- `token_log_probabilities`;
- `sum_log_probability`;
- `mean_log_probability`;
- `token_count`;
- `unit`, fixed to `nats`.

This score describes the model probability assigned to a specified token
sequence. A baseline-to-edited increase is evidence about that target only; it
is not proof of factual correctness, generalization, locality, or model safety.

## Generality and locality reductions

Status: implemented for paired mean target log-probabilities.

Knowledge-editing evaluations distinguish behavior on in-scope paraphrases
from preservation of out-of-scope inputs. These dimensions are described by
the primary [EasyEdit system paper](https://aclanthology.org/2024.acl-demos.9/)
and by the [ROME/CounterFact project](https://rome.baulab.info/), which evaluates
generalization and specificity. KEditAudit's exact continuous reductions below
are transparent project definitions; they do not claim numeric equivalence to
the accuracy metrics reported by those systems.

### Paired-probe input

Each `PairedProbeScore` contains:

- a stable `probe_id` from the AuditCase;
- baseline and edited `mean_log_probability` values in nats; or
- whichever value is available plus an explicit `missing_reason`.

Scores must be finite and non-positive. Probe IDs must be unique. Input order is
preserved in the result so every aggregate remains linked to its source probe.

### Generality: mean target log-probability delta

For paraphrase probe set \(\mathcal{P}\):

\[
G_{mean\_delta} = \frac{1}{|\mathcal{P}_{available}|}
\sum_{p\in\mathcal{P}_{available}}
\left[S^{edited}_{target}(p)-S^{baseline}_{target}(p)\right].
\]

The metric ID is `generality.mean_target_log_probability_delta`; higher values
mean the edited model assigned more probability to the intended target across
the evaluated paraphrases. Each probe retains both scores, the signed delta,
absolute delta, and signed aggregate contribution.

### Locality: mean absolute target log-probability delta

For locality probe set \(\mathcal{L}\), where the scored target represents
behavior intended to remain stable:

\[
L_{mean\_absolute\_delta} = \frac{1}{|\mathcal{L}_{available}|}
\sum_{p\in\mathcal{L}_{available}}
\left|S^{edited}_{target}(p)-S^{baseline}_{target}(p)\right|.
\]

The metric ID is `locality.mean_absolute_target_log_probability_delta`; lower
values mean less drift for those exact prompt-target pairs. Signed deltas remain
available even though absolute deltas contribute to the aggregate.

This is a narrow target-score locality diagnostic. It does not measure full
distribution drift, top-k agreement, generation changes, or every possible
side effect. In particular, it is not the KL control-divergence metric planned
below.

### Coverage and result contract

The reduction name is `arithmetic_mean_over_available_probes`. Missing pairs
are never substituted with zero:

- complete coverage produces no warning;
- partial coverage produces an aggregate over complete pairs plus a warning;
- zero complete pairs produces `aggregate: null` plus an unavailable warning.

`ProbeScoreReduction.as_dict()` records metric ID, direction, reduction, unit,
aggregate, total/evaluated/missing counts, coverage, warnings, and all raw
`ProbeScoreEvidence`. A later AuditReport may mark mandatory missing probes as
an incomplete run; this reducer does not silently make that policy decision.

Neither reduction emits a PASS/FAIL threshold. Generality on a finite probe set
does not establish universal generalization, and low measured locality drift is
not a safety guarantee.

## Control-distribution KL divergence

Status: implemented over caller-supplied, position-aligned baseline and edited
logits.

For each control probe position, KEditAudit computes the directed divergence

\[
D_{KL}(P_{baseline}\parallel P_{edited}) =
\sum_v P_{baseline}(v)
\log\frac{P_{baseline}(v)}{P_{edited}(v)}.
\]

The definition follows Kullback and Leibler's primary paper,
[“On Information and Sufficiency”](https://doi.org/10.1214/aoms/1177729694).
Both distributions are obtained with stable log-softmax after dividing logits
by the recorded positive `temperature`. Baseline and edited rows must have the
same positions, vocabulary size, and vocabulary ordering; the reducer cannot
verify tokenizer identity from numeric arrays alone.

The metric ID is `control.mean_kl_divergence`, direction is
`baseline||edited`, unit is `nats`, and lower values mean less measured drift on
the supplied controls. A probe contribution is the arithmetic mean of all its
position KL values. The final aggregate is the arithmetic mean of complete
probe contributions, so probes receive equal weight even when they contain
different numbers of positions.

`ControlDivergenceReduction.as_dict()` retains every position divergence,
per-probe mean, position and vocabulary counts, temperature, probe coverage,
missing reasons, warnings, and the final aggregate. It deliberately does not
retain vocabulary-sized logits. Missing pairs are excluded with an explicit
warning and are never substituted with zero.

The implementation rejects non-finite numbers, inconsistent shapes, duplicate
probe IDs, invalid temperatures, and inputs beyond documented resource limits.
Tiny negative KL values caused only by floating-point rounding are clamped to
zero within a `1e-12` tolerance; larger negative results fail closed.

KL divergence is asymmetric and control-set dependent. A low value does not
show that every unrelated behavior is unchanged and is not evidence that a
model or edit is safe.

## Structural weight differences

Status: implemented over caller-supplied, flattened baseline and edited tensor
values that correspond to a `ChangedTensorRecord` inventory.

For each complete tensor pair, KEditAudit computes:

\[
\Delta W = W_{edited} - W_{baseline},
\qquad
\|\Delta W\|_F = \sqrt{\sum_i |\Delta W_i|^2},
\]

and, when the baseline norm is nonzero, the descriptive relative change

\[
r_F = \frac{\|\Delta W\|_F}{\|W_{baseline}\|_F}.
\]

The Frobenius definition and the matrix 2-norm as the largest singular value
follow Golub and Van Loan's
[Matrix Computations](https://www.press.jhu.edu/books/title/10678/matrix-computations)
and the official [NumPy norm reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).
The core implementation remains Python-standard-library-only and does not copy
NumPy code.

The aggregate delta is the Euclidean norm across the available per-tensor
Frobenius deltas. This is equivalent to the Frobenius norm of the available
tensors treated as one flattened vector. Its relative form uses the same
aggregation over the corresponding available baseline tensors. Neither value
is silently treated as complete when inventory entries are missing.

When `spectral_iterations` is requested, rank-two deltas also receive a
deterministic power-iteration estimate of the largest singular value. It is
explicitly labeled as an estimate, records its iteration count, and is not
presented as an exact SVD. Non-matrix tensors retain a warning and a null
spectral value. Numeric size and work limits fail closed before an unbounded
calculation.

`StructuralDifferenceResult.as_dict()` preserves inventory order and includes
each tensor's name, shape, dtype, device, baseline/edited artifact hashes,
norms, coverage status, missing reason, and warnings. It deliberately omits raw
tensor values and checkpoints. An unsupplied or partial pair produces a
missing evidence row; it never contributes a zero to the aggregate.

These norms measure parameter-space magnitude. A small or large norm alone
does not identify the edited fact, explain behavior, establish causality, prove
locality, or show semantic harm. The result therefore has no automatic
PASS/FAIL direction and always carries a descriptive-only warning.
