# Milestone 0–5 acceptance audit

Audit date: 2026-08-29

This audit maps every deliverable and acceptance statement in
[`ROADMAP.md`](ROADMAP.md) to implementation or test evidence. It is an
engineering acceptance record, not a model-safety certification. Evidence is
scoped to the exact repository revision on which the listed commands pass.

## Summary

| Milestone | Status | Scope boundary |
|---|---|---|
| 0 — Research and governance | Accepted | Name availability is a dated search, not a reservation; third-party dataset terms remain operator responsibilities |
| 1 — Contracts and offline metrics | Accepted | Reducers operate on caller-supplied values; model execution remains adapter-owned |
| 2 — Tiny adapter and causal tracing | Accepted | Exact pinned CPU/float32 `GPT2LMHeadModel` only; synthetic random-weight integration fixture |
| 3 — External editor adapters | Accepted | Data-only normalized ROME/EasyEdit manifests; no upstream code, checkpoint, or tensor value is executed or redistributed |
| 4 — Ripple and structural audit | Accepted | Ripple is an input contract; the Milestone 5 CLI consumes already-computed evidence |
| 5 — CLI and reports | Accepted | Data-only snapshots only; no checkpoint loading, external editor execution, or safety conclusion |

## Milestone 0 — Research and governance

Deliverable evidence:

- problem, audience, differentiation, and non-goals:
  [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md), [`README.md`](../README.md);
- Apache-2.0 decision and full text: [`LICENSE`](../LICENSE),
  [`NAME_AND_CITATION.md`](NAME_AND_CITATION.md), and
  `test_repository_contains_complete_apache_license`;
- method citations plus dataset/artifact redistribution decisions:
  [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) and
  [`SOURCES_AND_LICENSES.md`](SOURCES_AND_LICENSES.md);
- dated name search: [`NAME_AND_CITATION.md`](NAME_AND_CITATION.md);
- checkpoint, prompt, remote-code, artifact, hook, report, and optional-API
  risks: [`THREAT_MODEL.md`](THREAT_MODEL.md) and
  `test_governance_docs_cover_required_threats_and_license_decisions`.

Acceptance evidence:

- README status text separates the implemented core from the Milestone 5 CLI,
  runner, and writers; it contains no runnable unimplemented quick start.
- Every implemented or proposed Milestone 0–4 metric has an equation or exact
  reduction, units/interpretation, limitations, and a primary paper, book, or
  official-project source in [`METRICS.md`](METRICS.md),
  [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md), and the source inventory.
- [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md#differentiation) and
  [`EDITOR_ADAPTER.md`](EDITOR_ADAPTER.md#easyedit-normalized-manifest-importer)
  explicitly distinguish KEditAudit's evidence layer from EasyEdit's editor and
  evaluation framework.

## Milestone 1 — Versioned contracts and offline metric engine

Deliverable evidence:

- packaged Draft 2020-12 contracts:
  `audit_case.schema.json`, `run_manifest.schema.json`,
  `metric_result.schema.json`, and `audit_report.schema.json` under
  `src/kedit_audit/artifacts`;
- offline loaders and semantic validation: `artifacts/schema.py`;
- canonical JSON and raw-byte SHA-256 rules: `artifacts/hashing.py` and
  [`RUN_MANIFEST.md`](RUN_MANIFEST.md);
- supplied-logit target scoring, coverage-aware generality/locality, and
  control KL: `metrics/behavioral.py` and `metrics/distributional.py`.

Acceptance evidence:

- default unit tests need no network, GPU, checkpoint, Torch, or Transformers;
  core dependencies contain no ML framework.
- invalid fixtures and numeric inputs assert JSONPath-like or indexed
  actionable errors in `test_audit_case_schema.py`,
  `test_run_manifest_schema.py`, `test_audit_report_schema.py`,
  `test_sequence_log_probability.py`, `test_probe_reductions.py`, and
  `test_control_divergence.py`.
- raw token, probe, and position values survive JSON round trips.
- `test_same_report_fixture_always_produces_identical_bytes_and_hash` verifies
  repeated canonical bytes and artifact hashes from the same fixture.

## Milestone 2 — Tiny-model adapter and causal tracing

Deliverable evidence:

- exact dependency-version and CPU/float32 GPT-2 gate:
  `adapters/transformers.py` and [`MODEL_ADAPTER.md`](MODEL_ADAPTER.md);
- descriptor-safe dotted/numeric resolution: `adapters/modules.py`;
- reverse-order, exception-safe hook ownership: `causal/hooks.py`;
- clean/corrupt/restore coordinator with one reused corruption object:
  `causal/tracer.py`;
- tokenizer-offset subject spans, deterministic embedding corruption, verified
  transformer-block restoration, and JSON-ready heatmaps: `causal/gpt2.py` and
  [`CAUSAL_TRACING.md`](CAUSAL_TRACING.md).

Acceptance evidence:

- numeric `ModuleList` paths and actionable failures:
  `test_module_resolution.py`;
- tuple/model-output preservation and cleanup on every tested exception path:
  `test_hook_manager.py`;
- reused corruption, deterministic round trips, and zero-gap warnings:
  `test_causal_tracer.py`;
- tokenizer-derived spans and fail-closed compatibility:
  `test_model_adapter.py` and `test_transformers_adapter.py`;
- the two tests in `tests/integration/test_transformers_gpt2_adapter.py` run on
  a tiny local random-weight GPT-2 fixture without downloading a checkpoint.

## Milestone 3 — External editor adapters

Deliverable evidence:

- versioned normalized artifact schema and immutable importer:
  `editor_artifact.schema.json`, `adapters/manifest.py`;
- lifecycle contract and changed-tensor hash inventory:
  `adapters/editor.py`;
- official-source-identity ROME and EasyEdit boundaries:
  `adapters/rome.py`, `adapters/easyedit.py`, and synthetic normalized fixtures
  under `tests/fixtures/editor_artifacts/valid`.

Acceptance evidence:

- manifests require editor revision, state/artifact IDs, tensor hashes, and
  bounded immutable hyperparameters.
- binding requires distinct baseline/edited roots and matching state IDs;
  paired scoring rechecks the baseline after edited evaluation and closes the
  session when observed contamination occurs (`test_editor_artifact.py`).
- ROME/EasyEdit importers reject unofficial source identity, missing lifecycle
  metadata, unsupported architecture, and non-exact model roots with explicit
  errors (`test_rome_adapter.py`, `test_easyedit_adapter.py`).

The accepted claim is narrow: these are normalized data-only adapter contracts,
not proof that arbitrary live upstream checkpoints or algorithms are compatible.

## Milestone 4 — Ripple and structural audit

Deliverable evidence:

- versioned six-category portability/ripple contract, explicit edited-target
  versus baseline-preservation expectations, and fictional fixture:
  `ripple_case.schema.json`, `test_ripple_case_schema.py`, and
  [`RIPPLE_CASE.md`](RIPPLE_CASE.md);
- directed `KL(baseline || edited)` with raw position/probe evidence and
  coverage: `metrics/distributional.py`;
- per-tensor Frobenius changes, optional deterministic rank-two spectral
  estimates, aggregate linkage, hash provenance, and missing coverage:
  `metrics/structural.py` and `test_structural_metrics.py`.

Acceptance evidence:

- KL aggregates retain every available position and probe; structural
  aggregates retain an ordered row for every inventoried tensor.
- neither reducer zero-fills missing values; partial and zero coverage are
  explicit and tested.
- structural results always state `descriptive-only` and warn that magnitude
  is not automatically semantic harm or model safety.
- ROME, EasyEdit, RippleEdits, CounterFact, synthetic fixtures, checkpoints,
  and changed-weight redistribution decisions are recorded in
  [`SOURCES_AND_LICENSES.md`](SOURCES_AND_LICENSES.md).

## Milestone 5 — CLI and reports

Deliverable evidence:

- dependency-light `validate-case`, `audit`, and `compare` commands:
  `src/kedit_audit/cli.py` and [`CLI.md`](CLI.md);
- versioned data-only baseline/edited input contract and fictional fixtures:
  `audit_snapshot.schema.json`, `test_audit_snapshot.py`, and
  [`AUDIT_SNAPSHOT.md`](AUDIT_SNAPSHOT.md);
- manifest-first execution with terminal-report finalization:
  `audit/runner.py`, `audit/pipeline.py`, and `test_audit_runner.py`;
- efficacy, generality, locality, portability, and control-KL assembly with raw
  probe evidence: `metrics/behavioral.py`, `metrics/distributional.py`, and
  `test_audit_pipeline.py`;
- validated canonical JSON, escaped deterministic Markdown, and strict report
  comparison: `reporting/writer.py`, `reporting/comparison.py`,
  `test_report_writer.py`, and `test_report_comparison.py`.

Acceptance evidence:

- `test_cli_help_works_when_ml_imports_are_blocked` proves CLI help imports no
  Torch, Transformers, or NumPy.
- the audit pipeline persists `running`, computes metrics, validates the full
  report, writes authoritative JSON and escaped Markdown, then persists the
  same completed manifest; nested and standalone manifests are asserted equal.
- malformed/incompatible preflight input creates no misleading manifest;
  metric and report-finalization exceptions replace an existing running
  manifest with a validated failed manifest that excludes exception text.
- snapshot/case probe IDs require exact coverage, comparison context is equal,
  baseline/edited logical IDs differ, and comparable artifact hashes cannot be
  equal.
- input/output path aliasing is rejected before any write; bounded JSON,
  duplicate-key rejection, finite numbers, symbolic-link refusal, atomic
  replacement, HTML/Markdown escaping, and private-value non-echo are covered
  by unit tests.
- reports retain numeric per-probe evidence but omit AuditCase prompts,
  subjects, and targets. Fixed limitations state that snapshots are
  caller-supplied and that results are not safety certification.

The accepted end-to-end claim begins at validated evidence snapshots. A future
live-adapter workflow must produce those measurements without weakening model
state isolation, resource limits, or provenance requirements.

## Reproduction gate

From the repository root, the acceptance gate is:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m integration tests\integration\test_transformers_gpt2_adapter.py
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src\kedit_audit
.\.venv\Scripts\python.exe -m pip check
```

Observed on 2026-08-29:

- default offline gate: 193 passed, 2 integration tests deselected;
- local GPT-2 integration gate: 2 passed;
- Ruff: all checks passed;
- strict mypy: no issues in 30 source files;
- `pip check`: no broken requirements;
- online `pip-audit --skip-editable`: no known vulnerabilities in the installed
  third-party environment (the editable KEditAudit distribution is skipped).

A repository scan also found no high-confidence API-key/private-key patterns,
private absolute user paths, or tracked credential-like filenames. The only
initial dangerous-API text match was the typed/model call to `.eval()` in the
Transformers adapter, not Python's `eval` builtin. Automated scans reduce risk;
they do not prove the absence of every secret or vulnerability.

Milestone 6 remains intentionally open: public-beta packaging, contributor
workflow, CI, tutorial, release tagging, and external reproduction are not
claimed by this acceptance record.
