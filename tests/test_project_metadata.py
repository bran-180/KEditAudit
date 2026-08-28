import tomllib
from pathlib import Path

from kedit_audit import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _top_level_cff_value(key: str) -> str:
    prefix = f"{key}:"
    for line in (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    raise AssertionError(f"Missing top-level CITATION.cff key: {key}")


def test_name_version_and_license_metadata_are_consistent() -> None:
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert pyproject["name"] == "kedit-audit"
    assert pyproject["version"] == __version__
    assert pyproject["license"] == "Apache-2.0"
    assert _top_level_cff_value("title") == "KEditAudit"
    assert _top_level_cff_value("version") == __version__
    assert _top_level_cff_value("license") == "Apache-2.0"


def test_repository_contains_complete_apache_license() -> None:
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
