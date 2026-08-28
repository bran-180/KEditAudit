# Agent instructions

## Mission

Build KEditAudit as a reproducible, editor-agnostic audit layer for language-model knowledge editing. Prefer small verified increments over broad feature claims.

## Non-negotiable constraints

- Audit first. Do not begin by reimplementing every editing algorithm.
- Never describe an audit score as proof of safety, alignment, absence of backdoors, or legal forensic validity.
- Keep metric computation deterministic. Record every seed and reuse the same corruption tensor when comparing causal-tracing restoration runs.
- Keep baseline and edited states logically separate. Never use two references to the same mutated model as a comparison.
- Unit tests must run without a network connection, GPU, or model download.
- Any GPU or external-model test must be explicitly marked as integration or slow.
- Resolve model modules through tested adapters. Do not hard-code layer numbers from an article or an AI-generated table.
- Preserve raw per-example results. Human-readable reports must be derived from structured artifacts, not invented by a language model.
- Treat prompts, model outputs, checkpoints, and downloaded datasets as untrusted inputs.
- Do not upload checkpoints, prompts, or datasets to an external API unless the user has explicitly authorized that data flow.
- Cite the primary paper or official repository for every implemented method or benchmark.
- Work on one issue and one acceptance-criteria set at a time.

## Required development loop

1. Read `docs/PROJECT_BRIEF.md`, `docs/ARCHITECTURE.md`, and the relevant milestone in `docs/ROADMAP.md`.
2. Restate the issue boundary and acceptance criteria.
3. Add or update tests before implementation when practical.
4. Implement the smallest end-to-end slice.
5. Run focused tests, then the full local quality gate.
6. Review the diff for scientific claims, determinism, provenance, and accidental scope expansion.
7. Update documentation when a public contract changes.

## Local quality gate

```powershell
python -m pytest
python -m ruff check .
python -m mypy src
```

Do not add a dependency merely to avoid writing a small, well-tested internal function. Do not pin a model-specific layer registry until an integration test verifies the exact model and `transformers` revision.
