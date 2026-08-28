"""Bounded JSON input helpers for untrusted local artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import NoReturn

DEFAULT_MAX_JSON_BYTES = 10 * 1024 * 1024


class JsonInputError(ValueError):
    """Raised when a local JSON artifact cannot be read unambiguously."""


def load_json_document(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> object:
    """Read one finite, duplicate-key-free UTF-8 JSON document within a size cap."""

    artifact_path = Path(path)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    try:
        stat = artifact_path.stat()
    except OSError as error:
        raise JsonInputError(f"cannot access JSON artifact: {artifact_path}") from error
    if not artifact_path.is_file():
        raise JsonInputError(f"JSON artifact is not a regular file: {artifact_path}")
    if stat.st_size > max_bytes:
        raise JsonInputError(
            f"JSON artifact exceeds the {max_bytes}-byte input limit: {artifact_path}"
        )
    try:
        encoded = artifact_path.read_bytes()
    except OSError as error:
        raise JsonInputError(f"cannot read JSON artifact: {artifact_path}") from error
    if len(encoded) > max_bytes:
        raise JsonInputError(
            f"JSON artifact exceeds the {max_bytes}-byte input limit: {artifact_path}"
        )
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise JsonInputError(f"JSON artifact must use UTF-8: {artifact_path}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_finite_constant,
        )
    except JsonInputError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise JsonInputError(
            f"JSON artifact is not a valid bounded JSON document: {artifact_path}"
        ) from error


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise JsonInputError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> NoReturn:
    raise JsonInputError(f"JSON number must be finite; found {value}")
