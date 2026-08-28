"""Atomic local artifact writes with symbolic-link refusal."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class ArtifactWriteError(OSError):
    """Raised when a local artifact cannot be written safely and atomically."""


def write_bytes_atomically(target: str | Path, content: bytes) -> Path:
    """Write bytes through a same-directory temporary file and atomic replace."""

    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    target_path = Path(target)
    directory = target_path.parent
    if directory.exists() and directory.is_symlink():
        raise ArtifactWriteError("artifact output directory must not be a symbolic link")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArtifactWriteError("artifact output directory could not be created") from error
    if not directory.is_dir():
        raise ArtifactWriteError("artifact output directory must be a directory")
    if target_path.exists() and target_path.is_symlink():
        raise ArtifactWriteError("artifact target must not be a symbolic link")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=f".{target_path.name}-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target_path)
    except OSError as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ArtifactWriteError("artifact could not be written atomically") from error
    return target_path
