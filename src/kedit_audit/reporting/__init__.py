"""Validated JSON and deterministic human-readable audit reports."""

from kedit_audit.reporting.comparison import (
    MetricComparison,
    MetricSnapshot,
    ReportComparison,
    ReportComparisonError,
    compare_audit_reports,
)
from kedit_audit.reporting.writer import (
    AUDIT_REPORT_JSON_FILENAME,
    AUDIT_REPORT_MARKDOWN_FILENAME,
    AuditReportWriteError,
    AuditReportWriteResult,
    render_audit_report_markdown,
    write_audit_report,
)

__all__ = [
    "AUDIT_REPORT_JSON_FILENAME",
    "AUDIT_REPORT_MARKDOWN_FILENAME",
    "AuditReportWriteError",
    "AuditReportWriteResult",
    "MetricComparison",
    "MetricSnapshot",
    "ReportComparison",
    "ReportComparisonError",
    "compare_audit_reports",
    "render_audit_report_markdown",
    "write_audit_report",
]
