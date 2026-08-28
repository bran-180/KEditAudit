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
