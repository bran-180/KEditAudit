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


def test_governance_docs_cover_required_threats_and_license_decisions() -> None:
    threat_model = (PROJECT_ROOT / "docs" / "THREAT_MODEL.md").read_text(encoding="utf-8")
    inventory = (PROJECT_ROOT / "docs" / "SOURCES_AND_LICENSES.md").read_text(
        encoding="utf-8"
    )

    for required_surface in (
        "Checkpoints",
        "Remote code",
        "Prompts and outputs",
        "Editor artifacts",
        "Baseline state",
        "Dataset records",
        "Reports",
        "Optional APIs",
    ):
        assert required_surface in threat_model

    for required_source in (
        "github.com/kmeng01/rome",
        "github.com/zjunlp/EasyEdit",
        "github.com/edenbiran/RippleEdits",
        "10.1214/aoms/1177729694",
        "press.jhu.edu/books/title/10678/matrix-computations",
        "numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html",
        "CounterFact",
        "unknown license",
    ):
        assert required_source in inventory


def test_milestone_zero_through_four_acceptance_is_documented() -> None:
    acceptance = (PROJECT_ROOT / "docs" / "MILESTONE_ACCEPTANCE.md").read_text(
        encoding="utf-8"
    )
    roadmap = (PROJECT_ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")

    for milestone in range(5):
        assert f"## Milestone {milestone} —" in acceptance
    for issue in range(13, 21):
        assert f"- [x] {issue}." in roadmap
    assert "Milestones 5 and 6 remain intentionally open" in acceptance
