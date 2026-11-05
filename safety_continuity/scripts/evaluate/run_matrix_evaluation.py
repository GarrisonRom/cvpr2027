"""Evaluate explicit completed training jobs with paired states and inference RNG."""
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
from safety_continuity.common.config import sha256_file
from safety_continuity.common.errors import ContractError
from safety_continuity.common.identity import validate_run_id
from safety_continuity.evaluation.self_built import generate_libero_state_manifest, load_state_manifest

LABELS = {"Continuous training": "continuous_training", "Full-state resume": "full_state_resume",
          "Optimizer-moment reset": "optimizer_moment_reset", "Weights-only restart": "weights_only_restart"}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def select_jobs(matrix):
    if matrix.get("schema") != "mvp_training_matrix.v1" or matrix.get("status") != "training_complete":
        raise ContractError("matrix training is not complete")
    jobs = matrix.get("completed_jobs", [])
    if len(jobs) != 30 or len({j["run_id"] for j in jobs}) != 30:
        raise ContractError("expected 30 distinct completed training jobs")
    identities = set()
    for job in jobs:
        validate_run_id(job["run_id"])
        if job.get("return_code") != 0 or job["condition_label"] not in LABELS:
            raise ContractError("failed/unknown matrix training job")
        identity = (job["condition_label"], job["training_seed"], job["interruption_step"])
        if identity in identities:
            raise ContractError("duplicate matrix condition/seed/position")
        identities.add(identity)
        if job["training_seed"] not in (42, 43, 44):
            raise ContractError("unexpected training seed")
        if job["condition_label"] == "Continuous training":
            if job["interruption_step"] is not None or job["checkpoint_steps"] != [499, 999, 1499, 1999]:
                raise ContractError("continuous checkpoint schedule mismatch")
        else:
            p = job["interruption_step"]
            if p not in (500, 1000, 1500):
                raise ContractError("unexpected interruption position")
            expected = 1999 if job["condition_label"] != "Weights-only restart" else 2000-p-1
            if job["checkpoint_steps"] != [expected]:
                raise ContractError("branch final checkpoint mismatch")
    return jobs


def audit_official_resources(output_root):
    from libero.libero import benchmark
    report = {"schema": "official_resource_audit.v1", "suites": [],
              "official_safety_labels_verified": False,
              "official_metric_provider": None,
              "status": "official_metric_provider_not_integrated",
              "fallback_to_generated_states": False}
    for suite_id in ("human_safety", "reasoning_safety"):
        suite = benchmark.get_benchmark_dict()[suite_id]()
        for task in suite.get_all_tasks_by_level(1):
            state = Path(suite._get_task_file_path(1, task.level_id, "init_states", ".pruned_init"))
            report["suites"].append({"suite": suite_id, "task": task.name,
                                    "state_file": str(state), "state_available": state.is_file(),
                                    "sha256": sha256_file(state) if state.is_file() else None})
    write_json(output_root / "metadata" / "official_resource_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "run"), default="plan")
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--state-seed", type=int, default=20261105)
    parser.add_argument("--evaluation-seed", type=int, default=2048)
    args = parser.parse_args()
    for name in ("training_matrix", "calibration_report", "payload_root", "openpi_root", "libero_root", "libero_config", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    matrix = json.loads(args.training_matrix.read_text(encoding="utf-8"))
    jobs = select_jobs(matrix)
    plan = {"training_job_count": len(jobs), "suites": ["human_safety", "reasoning_safety"],
            "episodes_per_task": 10, "self_built_episode_count": 3000,
            "continuous_control_reused_across_positions": True,
            "official_results_status": "requires_external_official_metric_provider",
            "capability_status": "separate_evaluation_required"}
    if args.mode == "plan":
        print(json.dumps(plan, indent=2)); return 0
    repo = PROJECT_ROOT.parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ContractError("matrix evaluation requires a clean code checkout")
    if args.output_root.exists():
        raise ContractError("refusing to overwrite evaluation matrix output")
    # Validate every checkpoint before creating any rollout output.
    for job in jobs:
        cp = Path(job["checkpoint_root"]) / str(job["checkpoint_steps"][-1])
        if not (cp / "params").is_dir() or not (cp / "assets" / "lerobot" / "norm_stats.json").is_file():
            raise ContractError(f"incomplete inference checkpoint: {cp}")
    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    if calibration.get("schema") != "physical_contact_calibration.v2" or calibration.get("contact_extraction_verified") is not True:
        raise ContractError("missing successful live contact calibration")
    args.output_root.mkdir(parents=True)
    env = os.environ.copy()
    env.update({"MUJOCO_GL": "egl", "LIBERO_CONFIG_PATH": str(args.libero_config),
                "PYTHONPATH": os.pathsep.join((str(args.libero_root), str(PROJECT_ROOT/"src"), str(args.openpi_root/"src"))),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "SC_GIT_COMMIT": commit, "SC_GIT_DIRTY": "false",
                "OPENPI_DATA_HOME": str(args.payload_root/"cache"/"openpi"),
                "HF_HOME": str(args.payload_root/"cache"/"huggingface")})
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.splitlines()[0].split(",")
    env.update({"SC_GPU_NAME": gpu[0].strip(), "SC_GPU_MEMORY_BYTES": str(int(gpu[1].strip())*1024*1024), "SC_GPU_DRIVER": gpu[2].strip()})
    os.environ.update({"LIBERO_CONFIG_PATH": str(args.libero_config), "MUJOCO_GL": "egl"})
    sys.path.insert(0, str(args.libero_root))
    status_path = args.output_root / "metadata" / "evaluation_matrix.json"
    status = {"schema": "mvp_evaluation_matrix.v2", "status": "preparing_states", "mode": args.mode,
              "git_commit": commit, "source_training_matrix": str(args.training_matrix),
              "source_training_matrix_sha256": sha256_file(args.training_matrix),
              "calibration_report_sha256": sha256_file(args.calibration_report), "plan": plan,
              "completed_jobs": [], "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "official_safety_claim_eligible": False, "capability_equivalence_established": False}
    write_json(status_path, status)
    try:
        audit_official_resources(args.output_root)
        # The same generated state files are used by every condition/seed/position.
        states = {}
        for suite in plan["suites"]:
            states[suite] = generate_libero_state_manifest(suite_id=suite, level=1,
                output_root=args.output_root/"state_manifest", generation_seed=args.state_seed,
                states_per_task=10)
        selected_jobs = jobs[:1] if args.mode == "smoke" else jobs
        status["status"] = "running"; write_json(status_path, status)
        for suite in plan["suites"]:
            for job in selected_jobs:
                checkpoint = Path(job["checkpoint_root"]) / str(job["checkpoint_steps"][-1])
                out = args.output_root/"evaluations"/job["run_id"]/suite
                command = [str(args.openpi_root/".venv"/"bin"/"python"),
                           str(PROJECT_ROOT/"scripts"/"evaluate"/"run_self_built_mvp.py"),
                           "--mode", "full", "--suite", suite, "--condition", LABELS[job["condition_label"]],
                           "--seed", str(job["training_seed"]), "--eval-seed", str(args.evaluation_seed),
                           "--state-seed", str(args.state_seed), "--states-per-task", "10",
                           "--episodes-per-task", "1" if args.mode == "smoke" else "10",
                           "--max-steps", "10" if args.mode == "smoke" else "400",
                           "--checkpoint-name", checkpoint.parent.name, "--weights-only-step", checkpoint.name,
                           "--source-training-run-id", job["run_id"],
                           "--run-tag", f'{args.mode}{job["run_id"].split("_")[3]}',
                           "--payload-root", str(args.payload_root), "--checkpoint-root", str(args.payload_root/"checkpoints"),
                           "--openpi-root", str(args.openpi_root), "--output-root", str(out),
                           "--state-manifest", str(states[suite]), "--port", str(args.port), "--paired-inference-rng",
                           "--monitor-mode", "human_contact" if suite == "human_safety" else "harmful_completion"]
                if job["interruption_step"] is not None:
                    command.extend(("--interruption-step", str(job["interruption_step"])))
                if suite == "human_safety":
                    command.extend(("--calibration-report", str(args.calibration_report)))
                log = args.output_root/"logs"/f'{job["run_id"]}_{suite}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("x", encoding="utf-8") as handle:
                    result = subprocess.run(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    raise ContractError(f"rollout failed: {job['run_id']} {suite}; inspect {log}")
                summaries = list(out.glob("*/metrics/summary.json"))
                if len(summaries) != 1:
                    raise ContractError("missing unique evaluation summary")
                summary = json.loads(summaries[0].read_text(encoding="utf-8"))
                expected_count = len(load_state_manifest(states[suite]).tasks) * (1 if args.mode == "smoke" else 10)
                if summary["episode_count"] != expected_count or summary["interruption_step"] != job["interruption_step"]:
                    raise ContractError("evaluation count/position mismatch")
                status["completed_jobs"].append({"source_training_run_id": job["run_id"], "suite": suite,
                    "summary": str(summaries[0]), "episode_count": summary["episode_count"], "log": str(log)})
                write_json(status_path, status)
        status["status"] = "smoke_complete" if args.mode == "smoke" else "adapted_safety_rollouts_complete"
        status["ended_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(status_path, status)
        return 0
    except BaseException as exc:
        status["status"] = "failed"; status["error"] = str(exc)
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
