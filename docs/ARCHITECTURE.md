# Architecture

## 1. Repository layout

```text
kedit-audit/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── AUDIT_CASE.md
│   ├── AUDIT_REPORT.md
│   ├── CAUSAL_TRACING.md
│   ├── GEMINI_REVIEW.md
│   ├── KNOWLEDGE_BASE.md
│   ├── METRICS.md
│   ├── MODEL_ADAPTER.md
│   ├── OSS_APPLICATION_DRAFT.md
│   ├── PROJECT_BRIEF.md
│   ├── RUN_MANIFEST.md
│   └── ROADMAP.md
├── examples/
│   ├── cases/
│   └── tiny_gpt2_audit/
├── src/kedit_audit/
│   ├── adapters/
│   │   ├── model.py
│   │   ├── rome.py
│   │   └── easyedit.py
│   ├── artifacts/
│   │   ├── audit_case.schema.json
│   │   ├── audit_snapshot.schema.json
│   │   ├── hashing.py
│   │   ├── run_manifest.schema.json
│   │   ├── schema.py
│   │   └── writer.py
│   ├── audit/
│   │   ├── runner.py
│   │   └── probes.py
│   ├── causal/
│   │   ├── hooks.py
│   │   └── tracer.py
│   ├── metrics/
│   │   ├── behavioral.py
│   │   ├── distributional.py
│   │   └── structural.py
│   ├── reporting/
│   │   └── markdown.py
│   ├── cli.py
│   └── __init__.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── AGENTS.md
├── README.md
└── pyproject.toml
```

Only directories needed by the active milestone should be created. This tree is a target architecture, not permission to generate empty modules.

## 2. Architectural boundary

The core depends on protocols rather than a specific editor implementation.

### `ModelAdapter`

Responsibilities:

- tokenize prompts and target sequences;
- return normalized next-token or sequence scores;
- expose named modules through a tested mapping;
- describe model/tokenizer revision, dtype, and device;
- provide stable baseline/edited identifiers.

The runtime protocol, immutable metadata, baseline/edited pair validation, and
deterministic offline fake are implemented and documented in
[`MODEL_ADAPTER.md`](MODEL_ADAPTER.md). A separate fail-closed adapter supports
only a preloaded CPU/float32 `GPT2LMHeadModel` under exact Torch and
Transformers versions. No external-editor compatibility is claimed.

The same adapter package provides a fail-closed dotted module resolver. It
supports numeric registered-module and sequence indices, reports the exact
resolved prefix on failure, and avoids invoking property descriptors. No
model-specific layer registry is pinned.

### `EditorArtifactAdapter`

Responsibilities:

- load or attach an externally produced edited state;
- identify changed tensors when possible;
- restore or dispose of the edited state safely;
- expose editor name, revision, hyperparameters, and source artifact provenance.

The core runner must not assume that an editor mutates a model in place.

The generic data-only contract and `EditorArtifactSession` are implemented.
Metadata records editor revision, source, architecture, state IDs, artifact
hashes, and immutable JSON hyperparameters; changed-tensor entries retain
hashes but no tensor values. Binding rejects a shared model root, and paired
scoring evaluates baseline, edited, then baseline again so an observed
in-place contamination fails closed. See [`EDITOR_ADAPTER.md`](EDITOR_ADAPTER.md).

The versioned editor manifest and `RomeArtifactAdapter` add a normalized ROME
fixture boundary. The importer verifies official source identity, immutable
revision, required hyperparameters, changed-tensor hashes, supported
architecture, and distinct preloaded roots without importing or executing the
ROME package. The current fixture is synthetic contract evidence, not a real
edited checkpoint.

`EasyEditArtifactAdapter` applies the same data-only boundary to a recorded
parameter-edit export. It requires the official source identity, exact revision,
algorithm/lifecycle hyperparameters, original-weight retention, a non-empty
changed-tensor inventory, and exact supported GPT-2 roots. No EasyEdit package
or external model is imported by default tests.

### `AuditCase`

A versioned data contract containing:

- edit subject and prompt template;
- intended target and optional original target;
- exact prompts;
- paraphrase probes;
- neighborhood/locality probes;
- portability probes with expected relationships;
- control prompts;
- dataset license and source metadata.

Version `1.0.0` is implemented as a Draft 2020-12 JSON Schema and documented in
[`AUDIT_CASE.md`](AUDIT_CASE.md). Its validator also enforces cross-category
probe-ID uniqueness and the single `{subject}` template field.

`RippleCase` version `1.0.0` is a separate portability/ripple input contract
with six normalized RippleEdits-inspired categories. It requires explicit
edited-target versus baseline-preservation expectations, raw expected answers,
global probe-ID uniqueness, and dataset/source/license revisions. Only a
fictional synthetic fixture is redistributed. See
[`RIPPLE_CASE.md`](RIPPLE_CASE.md).

### `MetricResult`

Every metric result contains:

- metric identifier and schema version;
- raw per-probe values;
- aggregate and reduction method;
- directionality and optional threshold;
- warnings and missing-data reasons;
- citations.

Version `1.0.0` is implemented as a Draft 2020-12 JSON Schema with offline
structural and semantic validation. Coverage counts must agree with the raw
probe array, missing evidence is explicit, and all numeric values must be
finite. See [`AUDIT_REPORT.md`](AUDIT_REPORT.md).

The first implemented reducer is target sequence log-probability over supplied,
already-aligned logits. It retains each token score plus sum and mean reductions
in natural-log units without importing Torch or loading a model. Its exact
contract and sign convention are documented in [`METRICS.md`](METRICS.md).

Generality and target-score locality reducers operate on paired baseline and
edited mean log-probabilities. They retain raw per-probe scores, signed and
absolute deltas, coverage, warnings, and missing reasons. Missing pairs are not
treated as zero; the exact reductions are documented in
[`METRICS.md`](METRICS.md#generality-and-locality-reductions).

The control-distribution reducer computes directed
`KL(baseline || edited)` from supplied aligned logits. It retains every
position divergence and probe mean, records temperature and coverage, and
requires identical position and vocabulary dimensions. It imports no model
library and is documented in
[`METRICS.md`](METRICS.md#control-distribution-kl-divergence).

The structural reducer joins caller-supplied flattened tensor pairs to the
ordered changed-tensor inventory. It computes coverage-aware Frobenius changes
and optional deterministic rank-two spectral estimates while retaining the
artifact hashes and omitting tensor values. Missing tensors remain explicit
and are never zero-filled. Structural evidence is descriptive and is not
automatically interpreted as semantic harm. See
[`METRICS.md`](METRICS.md#structural-weight-differences).

### `AuditReport`

The report combines the manifest, case metadata, metric results, structural evidence, and limitations. JSON is authoritative; Markdown is a deterministic rendering of JSON.

Version `1.0.0` is implemented as a composed Draft 2020-12 JSON Schema. Its
validator resolves the packaged `RunManifest` and `MetricResult` schemas,
checks nested semantic constraints, rejects duplicate metric IDs, and verifies
that report status and audit-case references agree with the manifest. The
report stores case metadata rather than duplicating prompts. Validated
canonical-JSON writing and escaped deterministic Markdown rendering are
implemented. The Milestone 5 data-only pipeline assembles complete reports from
paired validated AuditSnapshots.

### `AuditSnapshot`

`AuditSnapshot` is the dependency-light CLI boundary for caller-supplied
baseline or edited measurements. It records model, tokenizer, artifact,
environment, editor, generation, seed, and KEditAudit provenance together with
target scores and aligned control logits. Pair validation requires distinct
logical states, identical comparison context, and exact AuditCase probe
coverage. No checkpoint or external code is deserialized. See
[`AUDIT_SNAPSHOT.md`](AUDIT_SNAPSHOT.md).

## 3. Execution pipeline

```text
validate case
    -> resolve baseline and edited adapters
    -> freeze run manifest
    -> evaluate baseline probes
    -> evaluate edited probes
    -> compute behavioral metrics
    -> optionally compute weight diff
    -> optionally run causal tracing
    -> validate report schema
    -> write JSON
    -> render Markdown from JSON
```

If any mandatory probe fails, the report must show an incomplete state rather than silently dropping the probe.

The diagram is the live-adapter target pipeline. The implemented Milestone 5
CLI begins after evaluation: it validates paired data-only snapshots containing
the already-computed scores/logits, requires complete case coverage, and then
runs the metric, manifest, and report stages. Live checkpoint loading and
automatic incomplete-evidence reporting are not claimed by this command.

The implemented manifest-first runner accepts an already validated `running`
manifest and a caller-owned evaluation callback. It atomically persists the
running state, computes a proposed completed manifest for optional report
finalization, then persists a completed or failed terminal state. Failure
manifests exclude the exception message and traceback while the raised error
retains the local exception chain. The data-only CLI connects this runner to
snapshot reduction and report writing without loading a model; see
[`AUDIT_RUNNER.md`](AUDIT_RUNNER.md).

The report comparator validates both complete source contracts, requires the
same case, baseline, model/tokenizer, numeric environment, generation, and seed
context, then emits hash-linked `report_b - report_a` aggregate rows. Missing or
contract-incompatible metrics remain explicit and receive no numeric delta or
automatic conclusion. See [`REPORT_COMPARISON.md`](REPORT_COMPARISON.md).

## 4. Reproducibility manifest

Version `1.0.0` is implemented as a Draft 2020-12 JSON Schema and documented in
[`RUN_MANIFEST.md`](RUN_MANIFEST.md). Artifact fingerprints use SHA-256 and
record whether the digest covers exact raw bytes or KEditAudit canonical JSON
v1 bytes. An unavailable hash requires an explicit reason.

Record:

- KEditAudit commit;
- Python and package versions;
- model and tokenizer IDs plus immutable revisions;
- baseline and edited artifact hashes where permitted;
- device, dtype, quantization, and generation configuration;
- all random seeds;
- audit-case version and hash;
- editor adapter, source revision, and hyperparameters;
- start/end time and failure state.

## 5. Determinism and model state

- Set model evaluation mode explicitly.
- Disable dropout for deterministic evaluation.
- Separate baseline scores from edited-state lifetime.
- Reuse corruption tensors in paired causal-tracing experiments.
- Do not compare results generated with different tokenizers or incompatible generation settings.
- Make tolerances explicit for CPU/GPU and dtype-dependent differences.

The implemented `HookManager` owns forward hooks, cleans them in reverse order
on normal and exceptional exits, preserves an original model exception if
cleanup also fails, and never coerces hook outputs. Its offline contract and
limitations are documented in [`CAUSAL_TRACING.md`](CAUSAL_TRACING.md).

The clean/corrupt/restore coordinator validates module paths before execution,
derives the subject span through the adapter tokenizer, creates one corruption
object, and reuses that exact object for the corrupted and all restoration
runs. Its versioned JSON-ready evidence retains raw scores and recovery
reductions without serializing prompts or tensors. The coordinator remains
covered by dependency-free fake-adapter tests.

`GPT2CausalTraceAdapter` supplies the pinned real-model integration. It captures
detached outputs from exact `transformer.h.<index>` GPT-2 blocks, creates unit-
standard-deviation Gaussian noise with a local seeded CPU generator, adds that
fixed tensor only to the tokenizer-derived subject embedding span, and restores
the corresponding clean subject states at one block per comparison. Hook
ownership remains scoped to one model call. These diagnostic interventions do
not establish semantic causality or model safety.

## 6. Testing strategy

### Offline unit tests

Use toy tensors and a tiny local `nn.Module` to test:

- schemas and validation;
- module-path resolution, including numeric `ModuleList` indices;
- hook registration, restoration, and cleanup after exceptions;
- tuple/tensor output preservation;
- deterministic corruption reuse;
- metric reductions and missing data;
- report round-trip validation.

### Integration tests

- one in-memory two-layer GPT-2 under pinned Torch/Transformers versions;
- one pinned normalized ROME contract fixture;
- one pinned normalized EasyEdit contract fixture;
- no large model in default CI;
- downloaded-model tests opt in and cache by revision.

The GPT-2 integration is marked `integration`, runs on CPU with one Torch
intra-op thread, builds random weights and a local tokenizer in memory, and
performs no model download. It exercises scoring, two real block restorations,
deterministic heatmap evidence, local RNG isolation, and hook cleanup after an
intentional restoration error.

## 7. Optional OpenAI integration

The core audit must work without an OpenAI API key. Optional features may:

- propose candidate paraphrases or related probes for human review;
- summarize already-computed structured results;
- assist maintainers with issue triage and PR benchmark diffs.

Generated probes must record model, prompt, timestamp, review status, and license/provenance decisions. LLM output never determines the authoritative metric or safety conclusion.

## 8. Security and resource constraints

- Default to local-only artifact handling.
- Never auto-upload model weights or private prompts.
- Validate paths and avoid arbitrary code execution from model repositories when possible.
- Make `trust_remote_code` opt-in and visibly recorded.
- Fail before loading an unsupported or oversized model when resource estimates exceed configured limits.
- Treat HTML reports as untrusted-data renderings and escape content.
