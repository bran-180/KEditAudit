# Roadmap

The roadmap is designed for issue-sized vibe-coding sessions. Each milestone must end with a reviewable artifact and acceptance criteria.

## Milestone 0 — Research and governance baseline

Deliverables:

- public problem statement and non-goals;
- Apache-2.0 license decision and complete license file;
- citations and dataset-license inventory;
- final project name availability check;
- threat model for checkpoints, prompts, remote code, and reports.

Acceptance:

- no README feature is described as implemented;
- every proposed metric has a definition and primary source;
- the distinction from EasyEdit is explicit.

## Milestone 1 — Versioned contracts and offline metric engine

Deliverables:

- `AuditCase`, `RunManifest`, `MetricResult`, and `AuditReport` schemas;
- JSON serialization and schema versioning;
- target-sequence, generality, locality, and control-divergence reducers operating on supplied logits;
- offline unit tests with toy arrays.

Acceptance:

- tests require no Torch, network, or GPU;
- invalid cases produce actionable validation errors;
- raw per-probe values survive JSON round trips;
- the same fixture always yields the same report.

## Milestone 2 — Tiny-model adapter and causal-tracing prototype

Deliverables:

- one pinned Transformers causal-LM adapter;
- safe module resolution;
- hook manager with guaranteed cleanup;
- clean/corrupt/restore causal tracing using a reused corruption tensor;
- heatmap data artifact, with plotting optional.

Acceptance:

- numeric module paths work;
- tuple and model-output types are preserved;
- exception paths remove every hook;
- subject token span is tokenizer-derived;
- CPU integration test completes on a small fixture.

## Milestone 3 — External editor adapters

Deliverables:

- official ROME adapter or imported fixture;
- EasyEdit adapter;
- baseline/edited lifecycle tests;
- changed-tensor inventory.

Acceptance:

- adapter revision and hyperparameters are recorded;
- baseline scores cannot be contaminated by in-place mutation;
- unsupported architecture fails closed with a clear message.

## Milestone 4 — Ripple and structural audit

Deliverables:

- versioned portability/ripple case format;
- KL/control-drift implementation;
- Frobenius and optional spectral weight-diff metrics;
- incomplete-coverage warnings.

Acceptance:

- every aggregate links back to raw probes;
- structural magnitude is never interpreted automatically as semantic harm;
- benchmark and dataset licenses are documented.

## Milestone 5 — CLI and reports

Proposed commands:

```text
kedit-audit validate-case CASE.json
kedit-audit audit --baseline BASE --edited EDITED --case CASE.json --out reports/run-id
kedit-audit compare REPORT_A.json REPORT_B.json
```

Acceptance:

- CLI help works without loading Torch;
- JSON report validates before Markdown is written;
- reports escape untrusted strings;
- errors leave a manifest describing the incomplete run.

## Milestone 6 — Public beta and maintainer workflow

Deliverables:

- `v0.1.0` release;
- reproducible tutorial;
- contribution guide and code of conduct;
- issue and PR templates;
- CI matrix and a benchmark-diff check;
- at least one external reproduction report.

Acceptance:

- a new contributor can reproduce the tiny example from a clean environment;
- at least one issue was resolved through a documented maintainer workflow;
- the Codex for OSS application uses real repository evidence, not projected adoption.

## Issue-sized backlog

- [x] 1. Decide final name, license, and citation format.
- [x] 2. Define `AuditCase` JSON schema with valid and invalid fixtures.
- [x] 3. Define `RunManifest` and artifact hashing rules.
- [x] 4. Implement sequence log-probability from supplied logits.
- [x] 5. Implement generality/locality reductions with raw evidence retention.
- [x] 6. Implement `AuditReport` round-trip validation.
- [x] 7. Build a toy-model `ModelAdapter` protocol and fake adapter.
- [x] 8. Implement and test numeric module-path resolution.
- [x] 9. Implement a deterministic `HookManager` and cleanup tests.
- [x] 10. Implement clean/corrupt/restore tracing with one fixed corruption tensor.
- [x] 11. Build a pinned GPT-2 `ModelAdapter` and local CPU integration test.
- [ ] 12. Connect GPT-2 activations to clean/corrupt/restore tracing and emit heatmap data.

## Prompt template for each Codex session

```text
Read AGENTS.md and the active milestone in docs/ROADMAP.md.

Work only on Issue #<N>: <title>.

Acceptance criteria:
- <criterion 1>
- <criterion 2>
- <criterion 3>

First inspect the repository and explain the smallest implementation plan.
Then add tests and implement only this issue. Run the focused tests and the full
local quality gate. Finish with a concise diff review and any remaining risks.
Do not add unrelated features or claim support that has not been integration-tested.
```
