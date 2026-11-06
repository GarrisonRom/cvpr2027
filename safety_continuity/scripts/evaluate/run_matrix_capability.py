"""Run the frozen LIBERO-10 capability matrix with paired states and RNG."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from safety_continuity.common.errors import ContractError
from safety_continuity.evaluation.capability_matrix import (
    create_libero10_state_manifest,
    run_matrix,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "run", "unattended"), default="plan")
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT.parent)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-manifest", type=Path, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=2048)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=24576)
    parser.add_argument("--port", type=int, default=18100)
    args = parser.parse_args()
    for name in ("training_matrix", "payload_root", "openpi_root", "libero_root", "libero_config", "repo_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.state_manifest is not None:
        args.state_manifest = args.state_manifest.expanduser().resolve()
    if args.mode == "plan":
        return run_matrix(args)
    if args.mode == "unattended":
        if args.output_root.exists():
            raise ContractError(f"refusing to overwrite capability output: {args.output_root}")
        args.output_root.mkdir(parents=True)
        env = os.environ.copy()
        env.update({
            "MUJOCO_GL": "egl",
            "LIBERO_CONFIG_PATH": str(args.libero_config),
            "PYTHONPATH": os.pathsep.join((str(args.libero_root), str(PROJECT_ROOT / "src"), str(args.openpi_root / "src"))),
        })
        state_manifest = create_libero10_state_manifest(
            args.libero_root, args.output_root / "state_manifest", args.evaluation_seed
        )
        status_path = args.output_root / "metadata" / "unattended.json"
        status = {"schema": "libero10_capability_unattended.v1", "status": "smoke_pending",
                  "phases": [], "state_manifest": str(state_manifest)}
        write_json(status_path, status)
        for phase in ("smoke", "run"):
            phase_root = args.output_root / phase
            status["status"] = f"{phase}_active"
            write_json(status_path, status)
            command = [sys.executable, str(Path(__file__).resolve()), "--mode", phase,
                       "--training-matrix", str(args.training_matrix), "--payload-root", str(args.payload_root),
                       "--openpi-root", str(args.openpi_root), "--libero-root", str(args.libero_root),
                       "--libero-config", str(args.libero_config), "--repo-root", str(args.repo_root),
                       "--output-root", str(phase_root), "--state-manifest", str(state_manifest),
                       "--evaluation-seed", str(args.evaluation_seed), "--minimum-free-gpu-mib", str(args.minimum_free_gpu_mib),
                       "--port", str(args.port)]
            log_path = args.output_root / f"{phase}.log"
            with log_path.open("x", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=args.repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT)
            status["phases"].append({"phase": phase, "return_code": result.returncode})
            if result.returncode != 0:
                status["status"] = f"{phase}_failed"
                status["ended_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                write_json(status_path, status)
                return result.returncode
        status["status"] = "capability_rollouts_complete"
        status["ended_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(status_path, status)
        return 0
    args.git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo_root,
                                     capture_output=True, text=True, check=True).stdout.strip()
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
