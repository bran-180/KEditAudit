# Threat model

Last reviewed: 2026-08-29

KEditAudit processes model checkpoints, prompts, tokenizer output, editor
metadata, benchmark cases, and generated reports. All of them are untrusted.
The project produces diagnostic evidence; this threat model is not a claim that
an audited model, edit, dataset, or host is safe.

## Trust boundaries and protected assets

Protected assets include local credentials, private prompts and datasets,
baseline model integrity, edited-model provenance, the host filesystem, and the
accuracy of structured audit evidence. The trusted computing base is limited to
the checked-out KEditAudit revision, its pinned or constrained dependencies,
and explicitly selected local inputs.

External model repositories, checkpoints, pickle-compatible files, tokenizer
files, editor outputs, benchmark records, report strings, symbolic links, and
optional API services remain outside that boundary.

## Threats, current controls, and residual risk

| Surface | Threat | Current control | Residual risk / required future control |
|---|---|---|---|
| Checkpoints | Pickle deserialization or a crafted model executes code | Core APIs accept preloaded objects; KEditAudit calls neither `torch.load` nor `from_pretrained`; model files are ignored by Git | The caller that loads a model owns that risk. A future loader must prefer a non-executable format, cap file size, verify provenance, and document any unsafe fallback |
| Remote code | A model repository or dependency runs custom code | The pinned GPT-2 adapters require an exact installed class and expose no `trust_remote_code` or download helper | Dependency compromise and caller-created objects remain possible; use isolated environments and verified revisions |
| Prompts and outputs | Private text is uploaded, logged, or injected into a report | Core metrics run locally; CLI errors do not echo input values; audit reports exclude case prompts/targets/subjects and escape every rendered report string | JSON remains intentionally evidence-rich; operators must protect output directories and review user-supplied provenance text |
| Audit snapshots | A crafted evidence file conflates states, omits probes, exhausts resources, or overwrites an input through an output alias | Bounded duplicate-free JSON, versioned schema limits, exact case coverage, matching context, distinct state/artifact IDs and hashes, and input/output alias refusal | Schema-valid numeric evidence and provenance can still be fabricated; the CLI does not attest how upstream measurements were produced |
| Editor artifacts | A crafted manifest hides revision or changed tensors, or asks KEditAudit to execute an editor | Normalized ROME/EasyEdit importers validate bounded data-only manifests, require immutable revisions and hashes, reject unsupported architectures, and execute no upstream editor code | A valid manifest can still misstate its provenance; operators must verify artifact hashes and source revisions outside the schema |
| Baseline state | An in-place edit contaminates the baseline comparison | Metadata requires distinct state IDs; editor sessions reject a shared model root and score baseline, edited, then baseline again to detect observed score changes | A score-level recheck cannot prove every baseline tensor is unchanged; callers must keep independently loaded states and verify artifact hashes |
| Dataset records | Oversized or malformed records exhaust memory or bypass provenance | Versioned schemas reject unknown structure and require dataset source and license fields | Schema-valid text can still be sensitive or harmful; dataset-specific access and redistribution terms remain the operator's responsibility |
| Module paths and hooks | Path traversal invokes descriptors, or failed execution leaves hooks installed | The resolver rejects private/invalid paths without invoking properties; `HookManager` cleans every owned hook on normal and exceptional exits | Preloaded model methods are executable Python and remain inside the caller-selected runtime |
| Numeric inputs | NaN, infinity, extreme shapes, or adversarial values corrupt aggregates or exhaust resources | Metric inputs require finite values; the GPT-2 adapter and AuditSnapshot schema cap text, sequence, probe, position, vocabulary, file, and output-relevant sizes | Nested schema validation and allowed maximum inputs can still consume substantial local resources; run untrusted bulk inputs under OS-level limits |
| Artifact integrity | SHA-256 is mistaken for authenticity or two artifacts use different canonicalization | Manifests record hash algorithm, coverage, and canonicalization mode | A digest does not identify a trusted publisher; signatures and trusted distribution channels are out of scope for version 0.1 |
| Reports | HTML/Markdown injection or fabricated prose changes the apparent result | JSON validates before rendering; Markdown HTML-escapes text and entity-escapes control characters; input-controlled URLs are displayed as text; atomic writers reject symbolic-link targets | Cross-file writes are not transactional and concurrent hostile filesystem mutation remains outside the local writer's guarantee; consumers must compare nested and standalone manifests |
| Optional APIs | Prompts, weights, or datasets leave the local machine | No core feature calls an external API | Any future connector requires explicit user authorization and must record the data flow and provider |
| Supply chain | A vulnerable or substituted dependency changes behavior | Exact Torch/Transformers pins for the only real-model adapter; local quality gates include dependency consistency and vulnerability review | Registries and build backends are external trust dependencies; release artifacts should add reproducible-build and signing evidence |

## Fail-closed rules

- Never execute a downloaded repository merely to inspect its metadata.
- Never infer a model, tokenizer, editor, benchmark, or dataset revision.
- Never accept the same model object as both baseline and edited state.
- Never treat missing probes or changed-tensor coverage as zero.
- Never upload model or dataset content without explicit user authorization.
- Never interpret a low audit metric as proof of safety, alignment, legal
  validity, or absence of hidden behavior.

## Incident handling

If a credential, private prompt, restricted dataset record, or executable model
artifact is committed, stop publication, preserve the relevant commit IDs for
investigation, rotate exposed credentials, remove public access where possible,
and rewrite history only through an explicitly reviewed incident procedure.
Deleting the latest file alone is not sufficient because Git history retains it.
