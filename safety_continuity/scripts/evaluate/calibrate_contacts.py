"""Calibrate live simulator geometry pairs before policy safety evaluation."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from safety_continuity.evaluation.contact_calibration import calibrate_contacts

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT.parent)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    report = calibrate_contacts(output_root=args.output_root.resolve(), repo_root=args.repo_root.resolve(), seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "tasks"}, indent=2))
    return 0 if report["contact_extraction_verified"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
