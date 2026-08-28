# AuditReport comparison

Issue 24 implements a versioned `ReportComparison` artifact and the
`kedit-audit compare REPORT_A.json REPORT_B.json` command.

## Required shared context

Both source reports must validate independently and have identical:

- AuditCase artifact reference and canonical content hash;
- baseline artifact reference and content hash;
- model and tokenizer IDs and revisions;
- device, dtype, and quantization configuration;
- generation configuration and random seeds.

An unavailable case or baseline hash fails closed because equality cannot be
established. Edited artifacts may differ; that is the intended comparison.
Different KEditAudit commits produce a warning, while per-metric comparison is
still possible when the declared contracts match.

## Metric rows

Rows are ordered by `metric_id` and explicitly record whether a metric occurs
in both reports or only one. A metric is numerically comparable only when its
unit, direction, and reduction are identical. When both aggregates are finite,
the sole numeric comparison is:

```text
aggregate_delta_b_minus_a = report_b.aggregate - report_a.aggregate
```

Missing metrics and null aggregates remain null and carry warnings; they are
never substituted with zero. A subtraction outside the finite float range is
also represented as null. Every comparison links to canonical SHA-256 hashes
of both complete source reports.

The comparison does not reduce per-probe or structural-value differences and
does not apply directionality to pronounce a winner. Deltas are descriptive;
they do not automatically establish improvement, regression, semantic harm, or
model safety. Consumers should inspect the hash-linked authoritative JSON
reports for raw evidence.
