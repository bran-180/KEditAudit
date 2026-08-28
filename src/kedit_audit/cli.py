"""Dependency-light command-line interface for KEditAudit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn, TextIO, cast

from kedit_audit.artifacts import (
    AuditCaseValidationError,
    JsonInputError,
    load_json_document,
    validate_audit_case,
)

CommandHandler = Callable[[argparse.Namespace, TextIO, TextIO], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing any model framework."""

    parser = argparse.ArgumentParser(
        prog="kedit-audit",
        description="Validate and compare versioned knowledge-edit audit artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-case",
        help="validate an AuditCase JSON file without loading a model",
    )
    validate_parser.add_argument("case", type=Path, help="path to AuditCase JSON")
    validate_parser.set_defaults(handler=_validate_case_command)

    audit_parser = subparsers.add_parser(
        "audit",
        help="assemble an audit from already-separated baseline and edited evidence",
    )
    audit_parser.add_argument("--baseline", required=True, type=Path)
    audit_parser.add_argument("--edited", required=True, type=Path)
    audit_parser.add_argument("--case", required=True, type=Path)
    audit_parser.add_argument("--out", required=True, type=Path)
    audit_parser.set_defaults(handler=_planned_command)

    compare_parser = subparsers.add_parser(
        "compare",
        help="compare two validated AuditReport JSON files",
    )
    compare_parser.add_argument("report_a", type=Path)
    compare_parser.add_argument("report_b", type=Path)
    compare_parser.set_defaults(handler=_planned_command)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return a process exit code."""

    output = stdout if stdout is not None else sys.stdout
    error_output = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler = cast(CommandHandler, arguments.handler)
    return handler(arguments, output, error_output)


def entrypoint() -> NoReturn:
    """Console-script wrapper."""

    raise SystemExit(main())


def _validate_case_command(
    arguments: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        instance = load_json_document(cast(Path, arguments.case))
        validate_audit_case(instance)
    except JsonInputError as error:
        print(f"error: {error}", file=stderr)
        return 2
    except AuditCaseValidationError as error:
        print("error: AuditCase validation failed", file=stderr)
        for issue in error.issues:
            print(f"- {issue.path}: {_safe_validation_message(issue.message)}", file=stderr)
        return 2

    root = cast(dict[str, object], instance)
    result = {
        "case_id": root["case_id"],
        "schema_version": root["schema_version"],
        "status": "valid",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=stdout)
    return 0


def _planned_command(
    _arguments: argparse.Namespace,
    _stdout: TextIO,
    stderr: TextIO,
) -> int:
    print("error: this command is not implemented in Issue 21", file=stderr)
    return 2


def _safe_validation_message(message: str) -> str:
    lowered = message.lower()
    if "required property" in lowered:
        return message
    if "reuses" in lowered or "duplicates" in lowered:
        return "identifier must be unique across probe groups"
    if "does not match" in lowered or "{subject}" in message:
        return "value does not satisfy the required format"
    if "additional properties" in lowered:
        return "object contains unsupported properties"
    if "is not of type" in lowered:
        return "value has an invalid type"
    if "is not one of" in lowered:
        return "value is not one of the allowed options"
    if "is too long" in lowered or "is too short" in lowered:
        return "value violates the documented length constraint"
    return "value does not satisfy the AuditCase contract"
