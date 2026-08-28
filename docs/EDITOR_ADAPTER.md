# Editor artifact contract and model-state lifecycle

Status: generic data-only contract and ROME normalized-manifest importer
implemented; the EasyEdit importer is developed in Issue 17.

KEditAudit does not own an editing algorithm. `EditorArtifactAdapter` binds
provenance and a changed-tensor inventory to baseline and edited
`ModelAdapter` objects that the caller has already loaded as separate states.
The contract does not execute upstream editor code, deserialize checkpoints, or
download a model.

## Immutable provenance

`EditorAdapterMetadata` records:

- schema version;
- editor name and exact 40-character Git revision;
- credential-free HTTPS source repository;
- model architecture identifier;
- distinct baseline and edited logical state IDs;
- SHA-256 digests for the baseline and edited artifacts;
- a bounded, recursively copied JSON hyperparameter object.

`ChangedTensorRecord` retains each reported tensor name, shape, dtype, device,
and distinct baseline and edited SHA-256 digests. It intentionally excludes
tensor values. An inventory reports what an upstream export claims changed; it
does not independently prove that no unlisted tensor changed.

## Binding and contamination detection

`bind_editor_states` validates the normal `ModelAdapter` pair rules and also
requires different `module_root` identities. Two adapters around the same
mutated model object therefore fail before scoring.

For every `score_target_pair` request, `EditorArtifactSession` evaluates in
this order:

1. baseline score before edited evaluation;
2. edited score;
3. baseline score again.

If the second baseline result differs, the session closes and raises
`BaselineContaminationError`; no paired result is returned. The same check is
attempted when edited scoring raises. This detects mutation that affects the
evaluated prompt even when two wrapper objects hide shared underlying state.
It cannot prove that every unqueried parameter is independent, so callers must
still create truly separate model states.

The session is non-owning: `close()` invalidates the binding but does not delete
or restore caller-owned models. A future live-editor adapter that mutates state
must implement and test its own restoration before it can satisfy this
contract.

Paired score evidence contains normalized baseline and edited target-sequence
records but no prompt text. The lifecycle check is diagnostic integrity, not a
claim about edit correctness or safety.

## Versioned editor manifest

`editor_artifact.schema.json` defines the KEditAudit `1.0.0` interchange
format. It contains only artifact identity, editor provenance, bounded JSON
hyperparameters, model state IDs and hashes, and the changed-tensor inventory.
Semantic validation requires distinct state IDs and hashes, unique changed
tensor names, finite numeric metadata, and distinct before/after hashes for
every tensor reported as changed.

The schema distinguishes `external-export` from
`synthetic-contract-fixture`. This prevents test metadata from being presented
as a measured editor result. The schema contains no checkpoint path, prompt,
model output, tensor value, import target, or executable field.

## ROME normalized-manifest importer

`RomeArtifactAdapter` accepts only:

- `editor.name` equal to `ROME`;
- the official repository URL `https://github.com/kmeng01/rome`;
- an explicit 40-character producing revision;
- `GPT2LMHeadModel` artifact metadata and exact pinned Transformers GPT-2 roots
  when states are bound;
- recorded `layers` and `fact_token` hyperparameters;
- at least one changed-tensor inventory entry.

The official ROME repository documents that its editing API returns a rewritten
model and a dictionary of original values for edited weights. KEditAudit imports
a normalized hash inventory representing that boundary; it does not call the
ROME API or deserialize those weights. The committed fixture is synthetic and
pins official repository revision
`0874014cd9837e4365f3e6f3c71400ef11509e04` solely to test provenance and
fail-closed behavior. It is not evidence that KEditAudit reproduced an official
ROME edit or supports GPT-2 XL execution.
