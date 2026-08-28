# Sources, citations, and license inventory

Inventory date: 2026-08-28

This inventory distinguishes citations from redistribution permission. A paper
or public repository can be citable without granting permission to redistribute
every checkpoint or dataset mentioned by it. KEditAudit currently redistributes
only its own synthetic fixtures.

## Implemented methods and software sources

| Component | Primary or official source | Revision / identifier | Observed license | Included content |
|---|---|---|---|---|
| KEditAudit | This repository | exact commit used for a run | Apache-2.0 | Source, schemas, synthetic fixtures |
| Target scoring and causal tracing conventions | [ROME paper/project](https://rome.baulab.info/) and [official repository](https://github.com/kmeng01/rome) | paper: NeurIPS 2022; repository observed at `0874014cd9837e4365f3e6f3c71400ef11509e04` | [MIT](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/LICENSE) for repository code | Citation and normalized metadata only; no upstream code, weights, or results copied |
| Generality, locality, portability, and editor lifecycle vocabulary | [EasyEdit system paper](https://aclanthology.org/2024.acl-demos.9/) and [official repository](https://github.com/zjunlp/EasyEdit) | repository observed at `14cea8245f06715684592ab55184939b99d70784` | [MIT](https://github.com/zjunlp/EasyEdit/blob/14cea8245f06715684592ab55184939b99d70784/LICENSE) for repository code | Citation and normalized metadata only; EasyEdit is not vendored or executed |
| KL control divergence | Kullback and Leibler, [“On Information and Sufficiency”](https://doi.org/10.1214/aoms/1177729694) | DOI `10.1214/aoms/1177729694` (1951) | Scholarly citation; no paper text redistributed | Mathematical definition only |
| Ripple evaluation categories | [RippleEdits paper](https://aclanthology.org/2024.tacl-1.16/) and [official repository](https://github.com/edenbiran/RippleEdits) | repository observed at `54f3b88af4895a3aacb580ec63ce7ae857185040` | [MIT](https://github.com/edenbiran/RippleEdits/blob/54f3b88af4895a3aacb580ec63ce7ae857185040/LICENSE.txt) for repository contents | Citation and KEditAudit-authored synthetic schema fixtures only |

The repository revisions above are observations used to make the review
reproducible; they are not promises of live-package compatibility. An imported
artifact must record the exact revision that produced it.

## Dataset and artifact inventory

| Dataset or artifact | Intended use | License status | Redistribution in KEditAudit | Decision |
|---|---|---|---|---|
| KEditAudit synthetic audit, editor, and ripple fixtures | Offline tests | Apache-2.0 as original KEditAudit content | Yes | Allowed; fixtures must contain fictional data and no model weights |
| RippleEdits benchmark | Optional future portability/ripple import | Official repository is MIT at the revision above; individual records also derive from external knowledge sources | No | Record benchmark revision, subset, source URL, and license on every imported case; operator must review upstream source terms |
| CounterFact | Optional future ROME evaluation import | No dataset-specific redistribution conclusion has been recorded by KEditAudit | No | Block redistribution and packaged fixtures until a separate license review is documented |
| EasyEdit-supported third-party datasets | User-supplied external inputs | Varies by dataset; EasyEdit's code license does not relicense datasets | No | Require explicit dataset name, revision, source, and license; never infer from adapter name |
| Hugging Face or other model checkpoints | User-preloaded model state | Model-specific | No | Caller must record model and tokenizer revisions and comply with their licenses; KEditAudit does not download or publish weights |
| ROME/EasyEdit changed-weight values | Imported editor evidence | Model- and artifact-specific | No | Store hashes and bounded metadata by default, not tensor values or checkpoints |

An unknown license is represented as unknown, not silently treated as
permissive. Schema validation can establish that a license field exists; it
cannot determine whether the stated license is legally correct.

## Citation and claim rules

- Cite the primary paper or official repository for every implemented method or
  benchmark-derived format.
- Record immutable repository revisions for imported editor artifacts.
- Cite the exact KEditAudit commit used to compute a report.
- Do not imply that MIT-licensed upstream code was copied when only an interface
  contract or citation was used.
- Do not claim compatibility with an upstream revision until a fixture or
  integration test for that exact revision passes.
- Do not describe structural magnitude, locality, or KL divergence as safety
  certification or proof of semantic harm.
