# Project brief

## Identity

- Project name: **KEditAudit**
- Original Gemini codename: **EditScope**
- One-line description: An editor-agnostic toolkit for producing reproducible evidence about the intended and unintended effects of a language-model knowledge edit.
- License: Apache-2.0.

The codename was changed because “EditScope” is already used in current
model-editing research. On 2026-08-29, repeat checks of GitHub, PyPI, Crossref,
and general web results found only this project's public GitHub repository and
no unrelated KEditAudit package or paper. Repeat the checks before publishing a
package because availability is not a reservation.

## Problem

An edit can succeed on its exact prompt while failing on paraphrases, related facts, multi-hop consequences, or unrelated control behavior. A single “rewrite success” value therefore hides important failure modes. At the same time, existing frameworks already cover a broad catalogue of editors, so another algorithm wrapper would have weak differentiation.

## Target users

- knowledge-editing researchers comparing methods;
- interpretability researchers studying where an edit propagates;
- open-source model maintainers validating contributed patches or adapters;
- students who need a smaller, auditable path into ROME, MEMIT, and causal tracing.

## Product decision

KEditAudit is an **audit layer**, not an editor catalogue. Editing is performed by an external adapter or imported checkpoint. The core project owns:

- audit-case schemas;
- baseline/edited comparison;
- deterministic metric definitions;
- provenance and artifact formats;
- optional causal tracing and weight-diff analysis;
- reproducible reports.

## Planned MVP use case

When Milestone 5 is complete, a tiny supported autoregressive model, a baseline
state, an edited state, and a small versioned audit case will be sufficient for
one command to produce a JSON and Markdown report containing:

- target-sequence log-probability before and after the edit;
- exact-prompt efficacy;
- paraphrase generality;
- neighborhood locality;
- simple portability probes;
- KL divergence on control prompts;
- model, tokenizer, device, dtype, seed, dataset, and code provenance.

## Differentiation

EasyEdit already provides broad editing and evaluation support. KEditAudit should complement it through:

- a stable editor-independent audit artifact;
- strict reproducibility and provenance requirements;
- baseline-versus-edited diagnostics that retain raw evidence;
- causal and structural explanations that are clearly separated from behavioral scores;
- a lightweight CPU test path with no downloaded model.

## Non-goals for version 0.1

- proving the absence of hidden harmful behavior;
- certifying that a model is safe for deployment;
- implementing ROME and MEMIT from scratch;
- supporting arbitrary architectures through guessed module names;
- running ten-thousand-edit or multi-billion-parameter benchmarks;
- using OpenAI API outputs as ground truth.

## Success criteria

The project is ready for a first public release when:

1. the artifact schema is versioned and documented;
2. local tests run offline and deterministically;
3. one tiny-model integration example completes end to end;
4. an adapter can audit at least one official ROME or EasyEdit output;
5. every metric has a primary citation, definition, and failure interpretation;
6. a second person can reproduce the example from a clean environment;
7. the README contains no unimplemented quick-start claims.

## Responsible-use boundary

Reports are diagnostic evidence, not safety certification. The project should state uncertainty, document probe coverage, avoid publishing harmful edited checkpoints, and make external API use optional and explicit.
