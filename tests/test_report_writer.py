from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from kedit_audit.artifacts import validate_audit_report
from kedit_audit.reporting import (
    AUDIT_REPORT_JSON_FILENAME,
    AUDIT_REPORT_MARKDOWN_FILENAME,
    AuditReportWriteError,
    render_audit_report_markdown,
    write_audit_report,
)

FIXTURE = Path(__file__).parent / "fixtures" / "audit_reports" / "valid" / "completed.json"


def _report() -> dict[str, object]:
    return cast(dict[str, object], json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_validated_json_is_authoritative_and_markdown_is_derived(tmp_path: Path) -> None:
    report = _report()

    result = write_audit_report(report, output_directory=tmp_path)

    assert result.json_path.name == AUDIT_REPORT_JSON_FILENAME
    assert result.markdown_path.name == AUDIT_REPORT_MARKDOWN_FILENAME
    persisted = json.loads(result.json_path.read_text(encoding="utf-8"))
    validate_audit_report(persisted)
    assert persisted == report
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "report-20260820-001" in markdown
    assert "generality.mean&#95;target&#95;log&#95;probability&#95;delta" in markdown
    assert "Diagnostic evidence only" in markdown


def test_invalid_report_creates_no_output_directory(tmp_path: Path) -> None:
    invalid = _report()
    invalid["status"] = "failed"
    output = tmp_path / "not-created"

    with pytest.raises(AuditReportWriteError, match="before output"):
        write_audit_report(invalid, output_directory=output)

    assert not output.exists()


def test_markdown_escapes_untrusted_html_links_tables_and_newlines(tmp_path: Path) -> None:
    malicious = '<script>alert(1)</script>|[click](javascript:alert(2))\n# injected'
    report = _report()
    cast(list[object], report["limitations"])[0] = malicious
    metric = cast(list[dict[str, object]], report["metrics"])[0]
    citations = cast(list[dict[str, object]], metric["citations"])
    citations[0]["title"] = malicious

    result = write_audit_report(report, output_directory=tmp_path)

    persisted = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert persisted["limitations"][0] == malicious
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "<script>" not in markdown
    assert "[click](javascript:" not in markdown
    assert "&#124;" in markdown
    assert "&#91;click&#93;" in markdown
    assert "&#10;# injected" in markdown
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in markdown


def test_rendering_and_files_are_deterministic(tmp_path: Path) -> None:
    report = _report()
    first = write_audit_report(report, output_directory=tmp_path / "first")
    second = write_audit_report(copy.deepcopy(report), output_directory=tmp_path / "second")

    assert first.json_path.read_bytes() == second.json_path.read_bytes()
    assert first.markdown_path.read_bytes() == second.markdown_path.read_bytes()
    assert render_audit_report_markdown(report) == first.markdown_path.read_text(
        encoding="utf-8"
    )


def test_renderer_does_not_mutate_input() -> None:
    report = _report()
    original = copy.deepcopy(report)

    render_audit_report_markdown(report)

    assert report == original
