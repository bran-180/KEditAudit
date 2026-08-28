# Knowledge base

This document records the domain knowledge needed to implement KEditAudit. It distills the Gemini planning conversation, then corrects it against primary papers and official repositories.

## 1. Core terminology

Let a baseline autoregressive model be \(f_\theta\). A knowledge edit produces parameters \(\theta'\) or an external intervention that changes the model’s response for an edit descriptor \((x_e, y_e^*)\).

The audit must not reduce quality to a single scalar. At minimum it should retain these axes:

- **Efficacy / reliability**: Does the edited model prefer or generate the intended target on the exact edit prompt?
- **Generality**: Does the desired change transfer to paraphrases or semantically equivalent formulations?
- **Locality**: Are unrelated or neighboring behaviors preserved?
- **Portability**: Do consequences that logically depend on the new fact update appropriately?
- **Behavioral drift**: How much does the predictive distribution change on control inputs?
- **Structural drift**: Which parameters or internal representations changed, and by how much?
- **Reproducibility**: Can another user recreate the result with the same revisions, inputs, and seeds?

## 2. Metric design

Metric names must include their exact reduction and unit. Avoid ambiguous names such as `rewrite_score` without a definition.

### 2.1 Target sequence score

For a target token sequence \(y_{1:m}\), record mean conditional log-probability:

\[
S_{target}(x,y)=\frac{1}{m}\sum_{i=1}^{m}\log P(y_i\mid x,y_{<i}).
\]

Record both baseline and edited values. A binary efficacy result may be derived from a documented threshold or comparison, but raw values must remain available.

KEditAudit implements this definition as mean log-probability in natural-log
units and also retains the sum and every token value. This is the sign-reversed
form of the mean negative log-probability used by the official ROME CounterFact
[`test_batch_prediction`](https://github.com/kmeng01/rome/blob/main/experiments/py/eval_utils_counterfact.py#L97-L134)
evaluation. The supplied-logits and alignment contract is documented in
[`METRICS.md`](METRICS.md#target-sequence-log-probability).

### 2.2 Generality

For a fixed versioned paraphrase set \(\mathcal{P}\), report the distribution of target-sequence score changes, not only the mean:

\[
G=\frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}}
\left[S_{target}^{\theta'}(p,y^*)-S_{target}^{\theta}(p,y^*)\right].
\]

The report should also retain each paraphrase result because a high mean can hide failures.

KEditAudit implements this exact mean signed-delta definition over available
paired paraphrase scores. It preserves both state scores and every delta, and
emits coverage warnings rather than substituting missing probes with zero. See
[`METRICS.md`](METRICS.md#generality-and-locality-reductions).

### 2.3 Locality and control drift

For control prompts \(x\sim\mathcal{C}\), use a documented divergence, top-k agreement, or task score difference. One useful distributional measure is:

\[
D_{control}=\frac{1}{|\mathcal{C}|}\sum_{x\in\mathcal{C}}
D_{KL}\left(P_{\theta}(\cdot\mid x)\parallel P_{\theta'}(\cdot\mid x)\right).
\]

The vocabulary support, token position, temperature, and numerical precision must be recorded. Do not call this value a universal “ripple index” unless the exact definition and benchmark are versioned.

KEditAudit implements this directed baseline-to-edited KL divergence over
caller-supplied aligned logits, retaining every per-position and per-probe
value. It also provides a narrower locality diagnostic: mean absolute
baseline-to-edited change in the reference target's mean log-probability over
versioned locality probes. The two metrics are not interchangeable. See
[`METRICS.md`](METRICS.md#control-distribution-kl-divergence) and
[`METRICS.md`](METRICS.md#generality-and-locality-reductions).

### 2.4 Portability and ripple effects

Portability probes test consequences that should change after the edit. Ripple effects are relational and cannot be fully represented by unrelated-corpus perplexity. RippleEdits demonstrates that logical implications and related facts need dedicated evaluation cases.

### 2.5 Structural evidence

For edited tensors, report at least:

- absolute and relative Frobenius norm of \(\Delta W\);
- spectral norm when computationally feasible;
- changed parameter names, shapes, dtypes, and devices;
- optional singular-value summaries;
- hashes or stable identifiers for baseline and edited artifacts.

These values describe magnitude, not semantic harm.

## 3. Causal tracing

The ROME work uses causal tracing to locate computations that mediate factual recall. A defensible implementation separates:

1. clean run and cached activations;
2. corrupted run using a recorded corruption tensor;
3. restored runs that reuse the identical corruption while patching a chosen layer/token activation;
4. an explicit target token or target sequence metric;
5. a heatmap with layer, token, and module semantics.

Important implementation requirements:

- use the same corruption realization across comparisons;
- map subject spans through the tokenizer, not manually guessed token indices;
- define how tuple and model-output objects are handled;
- record all random seeds and noise parameters;
- distinguish activation restoration from weight editing.

## 4. ROME and MEMIT

ROME applies a rank-one update to an MLP projection in a localized model layer. Its complete procedure includes locating a layer, obtaining a subject key representation, optimizing a target value, estimating covariance statistics, and applying a constrained update. A short rank-one matrix expression is not a complete ROME implementation.

MEMIT extends the approach to many memories and distributes updates across a range of mediating layers. The official project reports specific layer ranges for the exact evaluated model configurations; those numbers must not be generalized to LLaMA or Qwen without model- and version-specific validation.

For the first KEditAudit releases, consume outputs through adapters instead of duplicating these algorithms.

## 5. Existing ecosystem and project gap

- The official ROME repository provides causal tracing, editing, CounterFact integration, and evaluation for selected autoregressive Hugging Face models.
- The official MEMIT project scales direct editing to many memories and provides its own evaluation workflow.
- EasyEdit is a mature framework with many editors, benchmarks, rollback, and evaluation dimensions.
- RippleEdits evaluates downstream and logically related effects that simple locality tests miss.

KEditAudit is justified only if it stays focused on portable audit artifacts, reproducibility, evidence retention, and editor-independent comparison.

## 6. Claims that require evidence

Do not state any of the following without experiments and citations:

- a universal best editing layer for a model family;
- that low parameter norm guarantees low behavioral drift;
- that good perplexity preservation implies safety;
- that a generated narrative is a forensic conclusion;
- that an edit is free of backdoors;
- that the toolkit supports a model before a pinned integration test passes.

## Primary sources

- [Locating and Editing Factual Associations in GPT / ROME](https://rome.baulab.info/)
- [Official ROME repository](https://github.com/kmeng01/rome)
- [Mass-Editing Memory in a Transformer / MEMIT](https://memit.baulab.info/)
- [Official MEMIT repository](https://github.com/kmeng01/memit)
- [Evaluating the Ripple Effects of Knowledge Editing](https://aclanthology.org/2024.tacl-1.16/)
- [EasyEdit repository](https://github.com/zjunlp/EasyEdit)
- [EasyEdit system paper](https://aclanthology.org/2024.acl-demos.9/)
- [Long-form evaluation of model editing](https://aclanthology.org/2024.naacl-long.208/)
