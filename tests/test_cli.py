from __future__ import annotations

import builtins
import io
import json
from pathlib import Path
from typing import Any

import pytest

from kedit_audit.artifacts import JsonInputError, load_json_document
from kedit_audit.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "audit_cases" / "valid" / "basic.json"
SNAPSHOT_FIXTURES = Path(__file__).parent / "fixtures" / "audit_snapshots" / "valid"


def test_cli_help_works_when_ml_imports_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith(("torch.", "transformers", "numpy")):
            raise AssertionError(f"CLI help attempted ML import {name!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "validate-case" in capsys.readouterr().out


def test_validate_case_emits_only_stable_public_identity() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(["validate-case", str(FIXTURE)], stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue()) == {
        "case_id": "eiffel-tower-relocation",
        "schema_version": "1.0.0",
        "status": "valid",
    }
    assert "Eiffel" not in stdout.getvalue()


def test_validate_case_reports_paths_without_echoing_private_values(tmp_path: Path) -> None:
    private_value = "private-subject-do-not-echo"
    instance = json.loads(FIXTURE.read_text(encoding="utf-8"))
    instance["edit"]["prompt_template"] = private_value
    case_path = tmp_path / "invalid.json"
    case_path.write_text(json.dumps(instance), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(["validate-case", str(case_path)], stdout=stdout, stderr=stderr)

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert "$.edit.prompt_template" in stderr.getvalue()
    assert private_value not in stderr.getvalue()


@pytest.mark.parametrize(
    "content",
    [
        '{"schema_version":"1.0.0",',
        '{"duplicate": 1, "duplicate": 2}',
        '{"value": NaN}',
    ],
)
def test_json_loader_rejects_ambiguous_or_nonstandard_json(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(JsonInputError):
        load_json_document(path)


def test_json_loader_enforces_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_text('{"value": 1}', encoding="utf-8")

    with pytest.raises(JsonInputError, match="input limit"):
        load_json_document(path, max_bytes=4)


def test_audit_command_runs_data_only_pipeline_end_to_end(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    output = tmp_path / "audit-output"

    exit_code = main(
        [
            "audit",
            "--baseline",
            str(SNAPSHOT_FIXTURES / "baseline.json"),
            "--edited",
            str(SNAPSHOT_FIXTURES / "edited.json"),
            "--case",
            str(FIXTURE),
            "--out",
            str(output),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    public_result = json.loads(stdout.getvalue())
    assert public_result["status"] == "completed"
    assert public_result["manifest"] == "run-manifest.json"
    assert public_result["report_json"] == "audit-report.json"
    assert public_result["report_markdown"] == "audit-report.md"
    assert (output / public_result["manifest"]).is_file()
    assert (output / public_result["report_json"]).is_file()
    assert (output / public_result["report_markdown"]).is_file()


def test_audit_command_does_not_echo_private_values_on_preflight_failure(
    tmp_path: Path,
) -> None:
    private_value = "private-model-identity-do-not-echo"
    edited = json.loads((SNAPSHOT_FIXTURES / "edited.json").read_text(encoding="utf-8"))
    edited["model"]["tokenizer_id"] = private_value
    edited_path = tmp_path / "private-edited.json"
    edited_path.write_text(json.dumps(edited), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "audit",
            "--baseline",
            str(SNAPSHOT_FIXTURES / "baseline.json"),
            "--edited",
            str(edited_path),
            "--case",
            str(FIXTURE),
            "--out",
            str(tmp_path / "output"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "error: audit inputs are invalid or incompatible\n"
    assert private_value not in stderr.getvalue()


def test_audit_command_rejects_input_output_alias_before_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit-output"
    output.mkdir()
    aliased_input = output / "run-manifest.json"
    original = (SNAPSHOT_FIXTURES / "baseline.json").read_bytes()
    aliased_input.write_bytes(original)
    stderr = io.StringIO()

    exit_code = main(
        [
            "audit",
            "--baseline",
            str(aliased_input),
            "--edited",
            str(SNAPSHOT_FIXTURES / "edited.json"),
            "--case",
            str(FIXTURE),
            "--out",
            str(output),
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert stderr.getvalue() == "error: audit input and output paths must not overlap\n"
    assert aliased_input.read_bytes() == original
    assert not (output / "audit-report.json").exists()
    assert not (output / "audit-report.md").exists()


def test_audit_command_handles_unusable_output_without_traceback(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    original = b"preserve this file"
    output_file.write_bytes(original)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "audit",
            "--baseline",
            str(SNAPSHOT_FIXTURES / "baseline.json"),
            "--edited",
            str(SNAPSHOT_FIXTURES / "edited.json"),
            "--case",
            str(FIXTURE),
            "--out",
            str(output_file),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "error: audit output could not be initialized or finalized\n"
    )
    assert output_file.read_bytes() == original
