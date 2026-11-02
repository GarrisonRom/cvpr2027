"""Command-line entry points kept thin so scripts do not duplicate validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .preflight import validate_project_configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safety-continuity project preflight")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing safety_continuity/",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat unresolved runtime resources and adapter fields as errors",
    )
    args = parser.parse_args(argv)
    issues = validate_project_configs(args.workspace_root)
    for issue in issues:
        print(f"{issue.severity.upper()} {issue.path}: {issue.message}")
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    print(f"preflight: {errors} error(s), {warnings} warning(s)")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
