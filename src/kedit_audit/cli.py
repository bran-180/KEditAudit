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
    canonical_json_bytes,
    load_json_document,
    validate_audit_case,
)
from kedit_audit.audit import (
    MANIFEST_FILENAME,
    AuditExecutionError,
    AuditPipelineInputError,
    AuditRunnerValidationError,
    run_audit_pipeline,
)
from kedit_audit.reporting import (
    AUDIT_REPORT_JSON_FILENAME,
    AUDIT_REPORT_MARKDOWN_FILENAME,
    ReportComparisonError,
    compare_audit_reports,
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
    audit_parser.set_defaults(handler=_audit_command)

    compare_parser = subparsers.add_parser(
        "compare",
        help="compare two validated AuditReport JSON files",
    )
    compare_parser.add_argument("report_a", type=Path)
    compare_parser.add_argument("report_b", type=Path)
    compare_parser.set_defaults(handler=_compare_command)
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


def _audit_command(
    arguments: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    input_paths = (
        cast(Path, arguments.baseline),
        cast(Path, arguments.edited),
        cast(Path, arguments.case),
    )
    if _audit_paths_overlap(input_paths, output_directory=cast(Path, arguments.out)):
        print("error: audit input and output paths must not overlap", file=stderr)
        return 2
    try:
        baseline = load_json_document(input_paths[0])
        edited = load_json_document(input_paths[1])
        audit_case = load_json_document(input_paths[2])
    except JsonInputError:
        print("error: an audit input is not a valid bounded JSON document", file=stderr)
        return 2
    try:
        result = run_audit_pipeline(
            audit_case=audit_case,
            baseline_snapshot=baseline,
            edited_snapshot=edited,
            output_directory=cast(Path, arguments.out),
        )
    except AuditPipelineInputError:
        print("error: audit inputs are invalid or incompatible", file=stderr)
        return 2
    except AuditExecutionError:
        print("error: audit failed; a failure manifest was written", file=stderr)
        return 1
    except AuditRunnerValidationError:
        print("error: audit output could not be initialized or finalized", file=stderr)
        return 1

    public_result = {
        "manifest": result.manifest_path.name,
        "report_json": result.report_json_path.name,
        "report_markdown": result.report_markdown_path.name,
        "run_id": result.run_id,
        "status": "completed",
    }
    print(canonical_json_bytes(public_result).decode("utf-8"), file=stdout)
    return 0


def _compare_command(
    arguments: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        report_a = load_json_document(cast(Path, arguments.report_a))
        report_b = load_json_document(cast(Path, arguments.report_b))
    except JsonInputError:
        print("error: a report input is not a valid bounded JSON document", file=stderr)
        return 2
    try:
        comparison = compare_audit_reports(
            cast(dict[str, object], report_a),
            cast(dict[str, object], report_b),
        )
    except ReportComparisonError as error:
        print(f"error: {error}", file=stderr)
        return 2
    print(canonical_json_bytes(comparison.as_dict()).decode("utf-8"), file=stdout)
    return 0


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


def _audit_paths_overlap(
    input_paths: Sequence[Path],
    *,
    output_directory: Path,
) -> bool:
    output_names = (
        MANIFEST_FILENAME,
        AUDIT_REPORT_JSON_FILENAME,
        AUDIT_REPORT_MARKDOWN_FILENAME,
    )
    try:
        output_targets = tuple(
            (output_directory / name).resolve(strict=False) for name in output_names
        )
    except OSError:
        return False
    for input_path in input_paths:
        try:
            resolved_input = input_path.resolve(strict=True)
        except OSError:
            continue
        for target in output_targets:
            if resolved_input == target:
                return True
            try:
                if target.exists() and resolved_input.samefile(target):
                    return True
            except OSError:
                continue
    return False
