"""Validate episode records for one or more explicitly named runs."""

from pathlib import Path
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    package_root = project_root / "safety_continuity" / "src"
    sys.path.insert(0, str(package_root))
    from safety_continuity.analysis.cli import validate_episodes_main

    return validate_episodes_main()


if __name__ == "__main__":
    raise SystemExit(main())
