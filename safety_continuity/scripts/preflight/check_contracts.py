"""Run repository-level configuration and interface preflight."""

from pathlib import Path
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    package_root = project_root / "safety_continuity" / "src"
    sys.path.insert(0, str(package_root))
    from safety_continuity.cli import main as package_main

    return package_main(["--workspace-root", str(project_root)])


if __name__ == "__main__":
    raise SystemExit(main())
