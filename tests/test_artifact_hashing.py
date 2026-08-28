import math
from pathlib import Path

import pytest

from kedit_audit.artifacts import (
    ArtifactHash,
    canonical_json_bytes,
    hash_bytes,
    hash_file,
    hash_json,
)


def test_raw_bytes_hash_has_stable_digest_and_size() -> None:
    result = hash_bytes(b"KEditAudit\n")

    assert result == ArtifactHash(
        algorithm="sha256",
        encoding="raw-bytes",
        digest="9c6e299b892efd51ebb9bd818d644e4a88b694d7e95621f5ce283b92dd58ca15",
        size_bytes=11,
    )
    assert result.as_dict()["digest"] == result.digest


def test_canonical_json_hash_is_key_order_independent() -> None:
    first = {"z": [3, 2, 1], "a": "café"}
    second = {"a": "café", "z": [3, 2, 1]}

    assert canonical_json_bytes(first) == b'{"a":"caf\xc3\xa9","z":[3,2,1]}'
    assert hash_json(first) == hash_json(second)
    assert hash_json(first) == ArtifactHash(
        algorithm="sha256",
        encoding="kedit-audit-canonical-json-v1",
        digest="e2e60d33b166052a2e446d8eb90e3249cb7b3714fd2713c75f89fd29160800c1",
        size_bytes=25,
    )


def test_file_hash_uses_exact_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"KEditAudit\n")

    assert hash_file(path) == hash_bytes(b"KEditAudit\n")


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"invalid": math.nan})


@pytest.mark.parametrize("value", [{1: "not-a-string-key"}, ("not", "a", "list")])
def test_canonical_json_rejects_non_json_native_values(value: object) -> None:
    with pytest.raises(TypeError):
        canonical_json_bytes(value)
