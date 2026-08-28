from kedit_audit import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.0.0"
