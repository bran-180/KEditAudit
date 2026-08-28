# Editor artifact contract and model-state lifecycle

Status: generic data-only contract implemented; ROME and EasyEdit manifest
importers are developed in Issues 16 and 17.

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
