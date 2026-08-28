"""Deterministic hashes for raw and JSON KEditAudit artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

HashEncoding = Literal["raw-bytes", "kedit-audit-canonical-json-v1"]


@dataclass(frozen=True)
class ArtifactHash:
    """A SHA-256 digest plus the exact byte representation that was hashed."""

    algorithm: Literal["sha256"]
    encoding: HashEncoding
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ValueError("algorithm must be 'sha256'")
        if self.encoding not in ("raw-bytes", "kedit-audit-canonical-json-v1"):
            raise ValueError(f"unsupported hash encoding: {self.encoding!r}")
        if len(self.digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.digest
        ):
            raise ValueError("digest must contain exactly 64 lowercase hexadecimal characters")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

    def as_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable content-hash record."""

        return {
            "algorithm": self.algorithm,
            "encoding": self.encoding,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }


def canonical_json_bytes(document: object) -> bytes:
    """Encode a JSON-native value using the KEditAudit canonical JSON v1 profile."""

    _require_json_native(document)
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def hash_bytes(data: bytes) -> ArtifactHash:
    """Hash an exact byte sequence with SHA-256."""

    return _artifact_hash(data, "raw-bytes")


def hash_json(document: object) -> ArtifactHash:
    """Hash a JSON-native value after KEditAudit canonical JSON v1 encoding."""

    return _artifact_hash(canonical_json_bytes(document), "kedit-audit-canonical-json-v1")


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> ArtifactHash:
    """Stream the exact contents of a file into SHA-256 without loading it all at once."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = sha256()
    size_bytes = 0
    with Path(path).open("rb") as artifact:
        while chunk := artifact.read(chunk_size):
            digest.update(chunk)
            size_bytes += len(chunk)
    return ArtifactHash(
        algorithm="sha256",
        encoding="raw-bytes",
        digest=digest.hexdigest(),
        size_bytes=size_bytes,
    )


def _artifact_hash(data: bytes, encoding: HashEncoding) -> ArtifactHash:
    return ArtifactHash(
        algorithm="sha256",
        encoding=encoding,
        digest=sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _require_json_native(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Out of range float values are not JSON compliant")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_native(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be a string")
            _require_json_native(item, f"{path}.{key}")
        return
    raise TypeError(f"value at {path} is not JSON-native: {type(value).__name__}")
