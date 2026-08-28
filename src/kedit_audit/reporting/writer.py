"""Validated AuditReport JSON and escaped deterministic Markdown output."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kedit_audit.artifacts import (
    ArtifactWriteError,
    AuditReportValidationError,
    canonical_json_bytes,
    validate_audit_report,
    write_bytes_atomically,
)

AUDIT_REPORT_JSON_FILENAME = "audit-report.json"
AUDIT_REPORT_MARKDOWN_FILENAME = "audit-report.md"


class AuditReportWriteError(RuntimeError):
    """Raised before or during output when an AuditReport cannot be published."""


@dataclass(frozen=True)
class AuditReportWriteResult:
    """Paths to one authoritative JSON report and its Markdown rendering."""

    json_path: Path
    markdown_path: Path


def write_audit_report(
    report: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> AuditReportWriteResult:
    """Validate once, then write authoritative JSON before derived Markdown."""

    normalized = _validated_report_copy(report)
    markdown = _render_validated_report(normalized)
    directory = Path(output_directory)
    json_path = directory / AUDIT_REPORT_JSON_FILENAME
    markdown_path = directory / AUDIT_REPORT_MARKDOWN_FILENAME
    try:
        write_bytes_atomically(
            json_path,
            canonical_json_bytes(normalized) + b"\n",
        )
        write_bytes_atomically(markdown_path, markdown.encode("utf-8"))
    except ArtifactWriteError as error:
        raise AuditReportWriteError("validated audit report output could not be written") from error
    return AuditReportWriteResult(json_path=json_path, markdown_path=markdown_path)


def render_audit_report_markdown(report: Mapping[str, object]) -> str:
    """Validate and render one report as escaped deterministic Markdown."""

    return _render_validated_report(_validated_report_copy(report))


def _validated_report_copy(report: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(report, Mapping):
        raise AuditReportWriteError("report must be a mapping")
    try:
        encoded = canonical_json_bytes(report)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as error:
        raise AuditReportWriteError("report must contain finite JSON values") from error
    if not isinstance(normalized, dict):
        raise AuditReportWriteError("report must be a JSON object")
    result = cast(dict[str, object], normalized)
    try:
        validate_audit_report(result)
    except AuditReportValidationError as error:
        raise AuditReportWriteError(
            "report must satisfy the AuditReport contract before output is created"
        ) from error
    return result


def _render_validated_report(report: dict[str, object]) -> str:
    manifest = cast(Mapping[str, object], report["manifest"])
    model = cast(Mapping[str, object], manifest["model"])
    audit_case = cast(Mapping[str, object], report["audit_case"])
    dataset = cast(Mapping[str, object], audit_case["dataset"])
    lines = [
        f"# KEditAudit report: {_escape_markdown(report['report_id'])}",
        "",
        "> Diagnostic evidence only. This report does not certify model safety or the absence of harm.",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Status | {_escape_markdown(report['status'])} |",
        f"| Generated at | {_escape_markdown(report['generated_at'])} |",
        f"| Run ID | {_escape_markdown(manifest['run_id'])} |",
        f"| Model | {_escape_markdown(model['model_id'])} |",
        f"| Model revision | {_escape_markdown(model['model_revision'])} |",
        f"| Tokenizer | {_escape_markdown(model['tokenizer_id'])} |",
        f"| Tokenizer revision | {_escape_markdown(model['tokenizer_revision'])} |",
        "",
        "## Audit case",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Case ID | {_escape_markdown(audit_case['case_id'])} |",
        f"| Dataset | {_escape_markdown(dataset['name'])} |",
        f"| Dataset version | {_escape_markdown(dataset['version'])} |",
        f"| Dataset license | {_escape_markdown(dataset['license'])} |",
        f"| Dataset source | {_escape_markdown(dataset['source'])} |",
        "",
        "## Metrics",
        "",
        "| Metric | Status | Aggregate | Unit | Coverage |",
        "|---|---|---:|---|---:|",
    ]
    metrics = cast(Sequence[Mapping[str, object]], report["metrics"])
    for metric in metrics:
        coverage = cast(Mapping[str, object], metric["coverage"])
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape_markdown(metric["metric_id"]),
                    _escape_markdown(metric["status"]),
                    _escape_markdown(_display_null(metric["aggregate"])),
                    _escape_markdown(metric["unit"]),
                    _escape_markdown(coverage["fraction"]),
                )
            )
            + " |"
        )

    lines.extend(("", "## Metric warnings and citations", ""))
    for metric in metrics:
        lines.append(f"### {_escape_markdown(metric['metric_id'])}")
        lines.append("")
        _append_evidence_notes(lines, metric)

    structural = cast(Sequence[Mapping[str, object]], report["structural_evidence"])
    lines.extend(("", "## Structural evidence", ""))
    if structural:
        lines.extend(("| Evidence | Type | Status | Values |", "|---|---|---|---|"))
        for evidence in structural:
            values = json.dumps(
                evidence["values"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        _escape_markdown(evidence["evidence_id"]),
                        _escape_markdown(evidence["evidence_type"]),
                        _escape_markdown(evidence["status"]),
                        _escape_markdown(values),
                    )
                )
                + " |"
            )
        lines.extend(("", "### Structural warnings and citations", ""))
        for evidence in structural:
            lines.append(f"#### {_escape_markdown(evidence['evidence_id'])}")
            lines.append("")
            _append_evidence_notes(lines, evidence)
    else:
        lines.append("No structural evidence was recorded.")

    _append_list(
        lines,
        heading="Limitations",
        values=cast(Sequence[object], report["limitations"]),
    )
    _append_list(
        lines,
        heading="Warnings",
        values=cast(Sequence[object], report["warnings"]),
    )
    return "\n".join(lines) + "\n"


def _append_list(lines: list[str], *, heading: str, values: Sequence[object]) -> None:
    lines.extend(("", f"## {heading}", ""))
    if not values:
        lines.append("None recorded.")
        return
    lines.extend(f"- {_escape_markdown(value)}" for value in values)


def _append_evidence_notes(
    lines: list[str],
    evidence: Mapping[str, object],
) -> None:
    warnings = cast(Sequence[object], evidence["warnings"])
    citations = cast(Sequence[Mapping[str, object]], evidence["citations"])
    lines.append("Warnings:")
    if warnings:
        lines.extend(f"- {_escape_markdown(warning)}" for warning in warnings)
    else:
        lines.append("- None recorded.")
    lines.append("")
    lines.append("Citations:")
    for citation in citations:
        lines.append(
            f"- {_escape_markdown(citation['title'])} — "
            f"{_escape_markdown(citation['source'])}"
        )
    lines.append("")


def _escape_markdown(value: object) -> str:
    text = html.escape(str(value), quote=True)
    replacements = {
        "\\": "&#92;",
        "`": "&#96;",
        "*": "&#42;",
        "_": "&#95;",
        "[": "&#91;",
        "]": "&#93;",
        "|": "&#124;",
        "\r": "&#13;",
        "\n": "&#10;",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text


def _display_null(value: object) -> object:
    return "null" if value is None else value
