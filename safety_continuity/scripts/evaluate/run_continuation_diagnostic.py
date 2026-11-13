"""Diagnose existing checkpoint repeatability and continuation effects; no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/"src"))
from safety_continuity.evaluation.continuation_diagnostic import run, run_rollout_worker


def main() -> int:
    if "--worker-request" in sys.argv:
        parser = argparse.ArgumentParser(description="Isolated simulator worker")
        parser.add_argument("--worker-request", required=True, type=Path)
        args = parser.parse_args()
        run_rollout_worker(json.loads(args.worker_request.read_text(encoding="utf-8")))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "run"), default="plan")
    parser.add_argument("--config", type=Path, default=PROJECT/"configs/evaluation/continuation_diagnostic.yaml")
    for name in ("training-matrix", "original-checkpoint", "payload-root", "openpi-root",
                 "capability-libero-root", "capability-libero-config", "safety-libero-root", "safety-libero-config",
                 "capability-states", "human-states", "reasoning-states", "calibration-report", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT.parent)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
