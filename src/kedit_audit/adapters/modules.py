"""Safe dotted-path resolution for registered model modules."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TypeGuard

_NAMED_SEGMENT = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_NUMERIC_SEGMENT = re.compile(r"^(?:0|[1-9][0-9]*)$")


class ModulePathError(ValueError):
    """Raised when a module path is unsafe, ambiguous, or cannot be resolved."""

    def __init__(
        self,
        path: str,
        *,
        segment: str,
        resolved_prefix: str,
        reason: str,
    ) -> None:
        self.path = path
        self.segment = segment
        self.resolved_prefix = resolved_prefix
        location = resolved_prefix or "<root>"
        super().__init__(
            f"cannot resolve module path {path!r} at segment {segment!r} "
            f"after {location!r}: {reason}"
        )


def resolve_module_path(root: object, path: str) -> object:
    """Resolve one validated path without invoking properties or guessing layers."""

    segments = _validate_path(path)
    current = root
    resolved: list[str] = []
    for segment in segments:
        current = _resolve_segment(
            current,
            path=path,
            segment=segment,
            resolved_prefix=".".join(resolved),
        )
        resolved.append(segment)
    return current


def _validate_path(path: str) -> tuple[str, ...]:
    if not isinstance(path, str):
        raise TypeError("module path must be a string")
    if not path:
        raise ModulePathError(path, segment="", resolved_prefix="", reason="path must not be empty")
    if len(path) > 4096:
        raise ModulePathError(
            path,
            segment="",
            resolved_prefix="",
            reason="path exceeds 4096 characters",
        )

    segments = tuple(path.split("."))
    for index, segment in enumerate(segments):
        prefix = ".".join(segments[:index])
        if not segment:
            raise ModulePathError(
                path,
                segment=segment,
                resolved_prefix=prefix,
                reason="path contains an empty segment",
            )
        if len(segment) > 256:
            raise ModulePathError(
                path,
                segment=segment,
                resolved_prefix=prefix,
                reason="segment exceeds 256 characters",
            )
        if segment.startswith("_"):
            raise ModulePathError(
                path,
                segment=segment,
                resolved_prefix=prefix,
                reason="private segment names are not allowed",
            )
        if segment.isdigit() and not _NUMERIC_SEGMENT.fullmatch(segment):
            raise ModulePathError(
                path,
                segment=segment,
                resolved_prefix=prefix,
                reason="numeric segments must not contain a leading zero",
            )
        if not _NUMERIC_SEGMENT.fullmatch(segment) and not _NAMED_SEGMENT.fullmatch(segment):
            raise ModulePathError(
                path,
                segment=segment,
                resolved_prefix=prefix,
                reason="invalid segment; use a public name or a non-negative numeric index",
            )
    return segments


def _resolve_segment(
    current: object,
    *,
    path: str,
    segment: str,
    resolved_prefix: str,
) -> object:
    namespace = _instance_namespace(current)
    registered = namespace.get("_modules")
    has_registry = isinstance(registered, Mapping)
    if isinstance(registered, Mapping) and segment in registered:
        return _require_value(
            registered[segment],
            path=path,
            segment=segment,
            resolved_prefix=resolved_prefix,
            source="registered module",
        )

    if isinstance(current, Mapping) and segment in current:
        return _require_value(
            current[segment],
            path=path,
            segment=segment,
            resolved_prefix=resolved_prefix,
            source="mapping entry",
        )

    if _NUMERIC_SEGMENT.fullmatch(segment) and _is_sequence(current):
        index = int(segment)
        if index >= len(current):
            raise ModulePathError(
                path,
                segment=segment,
                resolved_prefix=resolved_prefix,
                reason=f"index {index} is outside sequence range [0, {len(current)})",
            )
        return _require_value(
            current[index],
            path=path,
            segment=segment,
            resolved_prefix=resolved_prefix,
            source="sequence entry",
        )

    if segment in namespace and segment != "_modules":
        return _require_value(
            namespace[segment],
            path=path,
            segment=segment,
            resolved_prefix=resolved_prefix,
            source="instance field",
        )

    source = "registered module mapping" if has_registry else type(current).__name__
    raise ModulePathError(
        path,
        segment=segment,
        resolved_prefix=resolved_prefix,
        reason=f"segment {segment!r} was not found in {source}",
    )


def _instance_namespace(value: object) -> Mapping[str, object]:
    try:
        namespace = vars(value)
    except TypeError:
        return {}
    return namespace


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _require_value(
    value: object,
    *,
    path: str,
    segment: str,
    resolved_prefix: str,
    source: str,
) -> object:
    if value is None:
        raise ModulePathError(
            path,
            segment=segment,
            resolved_prefix=resolved_prefix,
            reason=f"{source} resolved to null",
        )
    return value
