# Threat model

Last reviewed: 2026-08-28

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
| Prompts and outputs | Private text is uploaded, logged, or injected into a report | Core metrics run locally; causal heatmap artifacts exclude prompt, target, subject, activation, and corruption tensors | Future CLI logging and Markdown rendering must minimize retained text and escape all untrusted strings |
| Editor artifacts | A crafted manifest hides revision or changed tensors, or asks KEditAudit to execute an editor | Current work does not execute ROME or EasyEdit | Imported adapters must validate a bounded data-only manifest, require immutable revisions and hashes, and reject unsupported architectures |
| Baseline state | An in-place edit contaminates the baseline comparison | Model metadata requires distinct logical state IDs | Metadata alone cannot detect shared or shallow-copied model state; Milestone 3 must reject shared roots and detect baseline score changes across edited evaluation |
| Dataset records | Oversized or malformed records exhaust memory or bypass provenance | Versioned schemas reject unknown structure and require dataset source and license fields | Schema-valid text can still be sensitive or harmful; dataset-specific access and redistribution terms remain the operator's responsibility |
| Module paths and hooks | Path traversal invokes descriptors, or failed execution leaves hooks installed | The resolver rejects private/invalid paths without invoking properties; `HookManager` cleans every owned hook on normal and exceptional exits | Preloaded model methods are executable Python and remain inside the caller-selected runtime |
| Numeric inputs | NaN, infinity, extreme shapes, or adversarial values corrupt aggregates or exhaust resources | Metric inputs require finite values; the GPT-2 adapter caps text, sequence, target, and vocabulary sizes | Future matrix and report APIs need explicit element, nesting, and output-size limits |
| Artifact integrity | SHA-256 is mistaken for authenticity or two artifacts use different canonicalization | Manifests record hash algorithm, coverage, and canonicalization mode | A digest does not identify a trusted publisher; signatures and trusted distribution channels are out of scope for version 0.1 |
| Reports | HTML/Markdown injection or fabricated prose changes the apparent result | JSON is authoritative; no Markdown generator is implemented | Milestone 5 must validate JSON before rendering and escape untrusted content; LLM-written prose must never replace raw evidence |
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
