# JSON and Markdown reporting

Issue 23 implements `write_audit_report` and
`render_audit_report_markdown`. The JSON artifact is authoritative; Markdown is
a deterministic view derived only from a fully validated AuditReport.

The writer follows this order:

1. copy the caller mapping through canonical finite JSON;
2. validate the complete nested AuditReport contract;
3. render Markdown in memory;
4. atomically write `audit-report.json`;
5. atomically write `audit-report.md`.

Invalid reports create no output directory. If the Markdown write fails after
JSON succeeds, the validated JSON remains available and authoritative; the
writer raises an error rather than claiming both artifacts were published.

All report-provided strings are HTML-escaped and Markdown control characters
for code spans, emphasis, links, tables, backslashes, and newlines are emitted
as entities. Dataset sources and citation URLs are displayed only as text; the
renderer never creates an input-controlled link or HTML element. The Markdown
contains fixed diagnostic language and never invents a PASS/FAIL conclusion.

Writes use same-directory temporary files and atomic replacement. Existing
symbolic-link output targets are rejected. This protects normal local artifact
production but is not a sandbox against a hostile account concurrently
modifying the same filesystem directory.

The Milestone 5 data-only pipeline invokes the writer through the audit
runner's finalization callback. The report therefore embeds the exact proposed
completed manifest; only after JSON and Markdown return successfully is that
terminal manifest persisted as the standalone `run-manifest.json`. A later
standalone-manifest write failure cannot roll back an already replaced report,
so consumers must require agreement between the nested and standalone
manifests rather than inferring completion from file presence alone.
