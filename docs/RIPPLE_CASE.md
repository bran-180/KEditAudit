# RippleCase contract

Version `1.0.0` is a Draft 2020-12 JSON Schema for portability and ripple
evaluation inputs. It is inspired by the six criteria published with
[RippleEdits](https://aclanthology.org/2024.tacl-1.16/) and its
[official repository](https://github.com/edenbiran/RippleEdits), observed at
revision `54f3b88af4895a3aacb580ec63ce7ae857185040` under the repository's
MIT license.

KEditAudit does not redistribute the RippleEdits benchmark. The committed
fixture is fictional KEditAudit content under Apache-2.0 and is marked
`synthetic-contract-fixture`.

## Normalized categories

`probes` always contains these six arrays:

- `relation_specificity`;
- `logical_generalization`;
- `subject_aliasing`;
- `compositionality_i`;
- `compositionality_ii`;
- `forgetfulness`.

The names normalize capitalization and the upstream README's historical
`Relation_Specifity` spelling. Empty categories remain explicit rather than
being omitted, and at least one probe must exist across the case.

Every probe records a globally unique ID, prompt, non-empty expected-answer
list, and an explicit expectation:

- `edited-target`: the probe is expected to reflect the edited fact or a
  consequence of it;
- `baseline-preservation`: the baseline behavior is expected to remain stable.

Optional conditions contain an `all` or `any` operator plus raw condition
queries and expected answers. The schema does not infer logical relationships
from natural language.

## Provenance and validation

Every case records dataset name, dataset revision, dataset license, HTTPS
source URL, and immutable 40-character source revision. An
`external-benchmark-case` additionally requires its upstream case ID. A
synthetic fixture must omit that ID so it cannot masquerade as imported data.

Validation rejects unknown fields, duplicate probe IDs across categories,
identical original and new targets, missing source/license evidence, invalid
revisions, blank or oversized text, duplicate expected answers, and a case with
zero probes.

The license field records caller-supplied provenance; schema validation cannot
decide whether that statement is legally correct. Importers must review the
specific dataset revision and upstream source terms before redistribution.

`RippleCase` defines evidence inputs only. It does not decide whether a model
answer matches an alias, aggregate results, or imply that finite ripple probes
cover every downstream consequence.
