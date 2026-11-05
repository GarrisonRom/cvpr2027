"""Plan or launch the multi-position π0.5 training matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safety_continuity.common.config import load_yaml, sha256_file
from safety_continuity.common.errors import ContractError
from safety_continuity.common.identity import make_run_id, validate_run_id
from safety_continuity.data.io import read_update_sample_jsonl
from safety_continuity.training.mvp_plan import MvpExperimentPlan


SELECTED_EPISODES = (0, 1, 2, 6, 7, 8, 10, 12, 14, 27)


def _load_plan(workspace_root: Path) -> tuple[MvpExperimentPlan, dict[str, Any]]:
    config_root = workspace_root / "safety_continuity" / "configs"
    configs = {
        "training": load_yaml(config_root / "training" / "pi05_lora_mvp.yaml"),
        "interruption": load_yaml(config_root / "interruption" / "mvp_matrix.yaml"),
        "evaluation": load_yaml(config_root / "evaluation" / "mvp.yaml"),
        "self_built": load_yaml(config_root / "evaluation" / "self_built_mvp.yaml"),
        "data": load_yaml(config_root / "data" / "libero10_openpi_minimal.yaml"),
    }
    plan = MvpExperimentPlan.from_mapping(
        training_config=configs["training"],
        interruption_config=configs["interruption"],
        evaluation_config=configs["evaluation"],
        self_built_config=configs["self_built"],
    )
    selected = tuple(configs["data"].get("selected_episode_indices", ()))
    if selected != SELECTED_EPISODES:
        raise ContractError(
            f"frozen LIBERO update episodes changed: expected {SELECTED_EPISODES}, got {selected}"
        )
    return plan, configs


def _experiment_tag(value: str) -> str:
    tag = re.sub(r"[^a-z0-9]+", "", value.lower())
    if not tag or len(tag) > 16:
        raise ContractError("experiment tag must contain 1-16 lowercase ASCII letters or digits")
    return tag


def _run_id(run_date: dt.date, tag: str, condition: str, seed: int) -> str:
    return make_run_id(
        "mvp",
        "pi05",
        f"{tag}{condition}",
        seed,
        run_date=run_date,
    )


def _checkpoint_name(tag: str, condition: str, seed: int, position: int | None = None) -> str:
    branch = f"{condition}p{position}" if position is not None else condition
    return f"{tag}_{branch}_s{seed}"


def _roots(payload_root: Path) -> dict[str, Path]:
    return {
        "data": payload_root / "data",
        "split": payload_root / "data" / "splits",
        "model": payload_root / "models",
        "checkpoint": payload_root / "checkpoints",
        "run": payload_root / "runs",
        "cache": payload_root / "cache",
    }


def _runner_pythonpath(openpi_root: Path) -> str:
    return os.pathsep.join((str(PROJECT_ROOT / "src"), str(openpi_root / "src")))


def _schedule_paths(plan: MvpExperimentPlan, payload_root: Path) -> dict[int, Path]:
    split_root = payload_root / "data" / "splits"
    paths: dict[int, Path] = {}
    for seed in plan.training_seeds:
        path = split_root / f"mvp_update_samples_s{seed}.jsonl"
        if not path.is_file():
            raise ContractError(f"frozen update plan is missing: {path}")
        samples = read_update_sample_jsonl(path)
        if len(samples) != plan.total_steps:
            raise ContractError(
                f"seed {seed} schedule has {len(samples)} samples; expected exactly {plan.total_steps}"
            )
        if any(sample.dataset_id != plan.dataset_revision for sample in samples):
            raise ContractError(f"seed {seed} schedule uses a different dataset revision")
        paths[seed] = path
    return paths


def _checkpoint_root(payload_root: Path, checkpoint_name: str) -> Path:
    return (
        payload_root / "checkpoints" / "openpi_mvp" / "pi05_libero_mvp"
        / checkpoint_name
    )


def _job_specs(
    *,
    plan: MvpExperimentPlan,
    run_date: dt.date,
    tag: str,
    include_weights_only: bool,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in plan.training_seeds:
        a0_name = _checkpoint_name(tag, "a0", seed)
        jobs.append({
            "condition": "A0",
            "condition_label": "Continuous training",
            "seed": seed,
            "position": None,
            "run_id": _run_id(run_date, tag, "a0", seed),
            "checkpoint_name": a0_name,
            "source_checkpoint": None,
            "resume": False,
            "reset_optimizer_moments": False,
            "total_steps": plan.total_steps,
            "schedule_start": 0,
            "schedule_count": plan.total_steps,
            "optimizer_step_start": 0,
            "checkpoint_at_steps": list(plan.interruption_steps),
        })
        for position in plan.interruption_steps:
            source_step = position - 1
            for condition, label in (
                ("A1", "Full-state resume"),
                ("A2", "Optimizer-moment reset"),
                *(([("A3", "Weights-only restart")] if include_weights_only else [])),
            ):
                branch_name = _checkpoint_name(tag, condition.lower(), seed, position)
                resume = condition in {"A1", "A2"}
                jobs.append({
                    "condition": condition,
                    "condition_label": label,
                    "seed": seed,
                    "position": position,
                    "run_id": _run_id(run_date, tag, f"{condition.lower()}p{position}", seed),
                    "checkpoint_name": branch_name,
                    "source_checkpoint": {
                        "name": a0_name,
                        "step": source_step,
                    },
                    "resume": resume,
                    "reset_optimizer_moments": condition == "A2",
                    "total_steps": plan.total_steps if resume else plan.total_steps - position,
                    "schedule_start": position,
                    "schedule_count": plan.total_steps - position,
                    "optimizer_step_start": position if resume else 0,
                    "checkpoint_at_steps": [],
                })
    for job in jobs:
        validate_run_id(job["run_id"])
    return jobs


def _preflight_targets(
    *,
    jobs: list[dict[str, Any]],
    payload_root: Path,
    manifest_path: Path,
    schedules: dict[int, Path],
) -> None:
    output_root = manifest_path.parent.parent
    if output_root.exists():
        raise ContractError(f"refusing to reuse existing training matrix output: {output_root}")
    if manifest_path.exists():
        raise ContractError(f"refusing to overwrite matrix manifest: {manifest_path}")
    for job in jobs:
        run_dir = payload_root / "runs" / job["run_id"]
        checkpoint_dir = _checkpoint_root(payload_root, job["checkpoint_name"])
        if run_dir.exists():
            raise ContractError(f"refusing to overwrite training run: {run_dir}")
        if checkpoint_dir.exists():
            raise ContractError(f"refusing to overwrite checkpoint output: {checkpoint_dir}")
        staging_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
        if staging_dir.exists():
            raise ContractError(f"resume checkpoint staging path already exists: {staging_dir}")
    for seed, path in schedules.items():
        if not path.is_file():
            raise ContractError(f"schedule disappeared during preflight: {path}")


def _run_env(args: argparse.Namespace) -> dict[str, str]:
    roots = _roots(args.payload_root)
    env = os.environ.copy()
    env.update({
        "SC_DATA_ROOT": str(roots["data"]),
        "SC_SPLIT_ROOT": str(roots["split"]),
        "SC_MODEL_ROOT": str(roots["model"]),
        "SC_CHECKPOINT_ROOT": str(roots["checkpoint"]),
        "SC_RUN_ROOT": str(roots["run"]),
        "SC_CACHE_ROOT": str(roots["cache"]),
        "HF_HOME": str(roots["cache"] / "huggingface"),
        "HF_HUB_CACHE": str(roots["cache"] / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(roots["cache"] / "huggingface" / "datasets"),
        "HF_LEROBOT_HOME": str(roots["cache"] / "huggingface" / "lerobot"),
        "OPENPI_DATA_HOME": str(roots["cache"] / "openpi"),
        "PYTHONPATH": _runner_pythonpath(args.openpi_root),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "SC_GIT_COMMIT": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root, capture_output=True,
            text=True, check=True,
        ).stdout.strip(),
        "SC_GIT_DIRTY": "false",
    })
    try:
        values = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()[0].split(",")
        env["SC_GPU_NAME"] = values[0].strip()
        env["SC_GPU_MEMORY_BYTES"] = str(int(values[1].strip()) * 1024 * 1024)
        env["SC_GPU_DRIVER"] = values[2].strip()
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        env.setdefault("SC_GPU_NAME", "unknown")
        env.setdefault("SC_GPU_MEMORY_BYTES", "0")
        env.setdefault("SC_GPU_DRIVER", "unknown")
    return env


def _stage_resume_checkpoint(
    *,
    source_checkpoint: Path,
    destination_root: Path,
    step: int,
) -> Path:
    if not source_checkpoint.is_dir():
        raise ContractError(f"resume source checkpoint is missing: {source_checkpoint}")
    destination_step = destination_root / str(step)
    if destination_root.exists():
        raise ContractError(f"refusing to overwrite staged resume checkpoint: {destination_root}")
    temporary = destination_root.with_name(destination_root.name + ".tmp")
    if temporary.exists():
        raise ContractError(f"temporary staging directory already exists: {temporary}")
    temporary.mkdir(parents=True)

    def link_or_copy(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
            return destination
        except OSError:
            return shutil.copy2(source, destination)

    try:
        shutil.copytree(
            source_checkpoint,
            temporary / str(step),
            copy_function=link_or_copy,
        )
        temporary.replace(destination_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if not (destination_step / "params").is_dir():
        raise ContractError(f"staged resume checkpoint is missing params: {destination_step}")
    return destination_step


def _execute_job(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    plan: MvpExperimentPlan,
    schedule_path: Path,
    env: dict[str, str],
    log_root: Path,
) -> dict[str, Any]:
    runner = PROJECT_ROOT / "scripts" / "train" / "openpi_mvp.py"
    python = args.openpi_root / ".venv" / "bin" / "python"
    checkpoint_root = args.payload_root / "checkpoints" / "openpi_mvp" / "pi05_libero_mvp"
    staged_source: Path | None = None
    if job["condition"] in {"A1", "A2"}:
        source = job["source_checkpoint"]
        source_checkpoint = checkpoint_root / source["name"] / str(source["step"])
        staged_root = checkpoint_root / job["checkpoint_name"]
        staged_source = _stage_resume_checkpoint(
            source_checkpoint=source_checkpoint,
            destination_root=staged_root,
            step=source["step"],
        )
    elif job["condition"] == "A3":
        source = job["source_checkpoint"]
        staged_source = checkpoint_root / source["name"] / str(source["step"])

    command = [
        str(python), str(runner),
        "--openpi-root", str(args.openpi_root),
        "--payload-root", str(args.payload_root),
        "--run-id", job["run_id"],
        "--condition", job["condition"],
        "--seed", str(job["seed"]),
        "--total-steps", str(job["total_steps"]),
        "--experiment-total-steps", str(plan.total_steps),
        "--schedule-start", str(job["schedule_start"]),
        "--schedule-count", str(job["schedule_count"]),
        "--optimizer-step-start", str(job["optimizer_step_start"]),
        "--schedule-path", str(schedule_path),
        "--selected-episodes", *[str(value) for value in SELECTED_EPISODES],
        "--checkpoint-name", job["checkpoint_name"],
    ]
    if job["position"] is not None:
        command.extend(("--interruption-step", str(job["position"])))
    if job["checkpoint_at_steps"]:
        command.extend(("--checkpoint-at-steps", *[str(step) for step in job["checkpoint_at_steps"]]))
    if staged_source is not None:
        command.extend(("--source-checkpoint-dir", str(staged_source)))
    if job["resume"]:
        command.append("--resume")
    if job["reset_optimizer_moments"]:
        command.append("--reset-optimizer-moments")

    log_path = log_root / f"{job['run_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise ContractError(f"refusing to overwrite training log: {log_path}")
    with log_path.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=args.repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise ContractError(f"training stage failed ({result.returncode}): {job['run_id']}\n{tail}")
    expected_checkpoint_root = checkpoint_root / job["checkpoint_name"]
    if job["condition"] == "A0":
        expected_steps = [step - 1 for step in plan.interruption_steps] + [plan.total_steps - 1]
    else:
        expected_steps = [job["total_steps"] - 1]
    missing = [expected_checkpoint_root / str(step) for step in expected_steps
               if not (expected_checkpoint_root / str(step)).is_dir()]
    if missing:
        raise ContractError(f"training stage omitted required checkpoint(s): {missing}")
    return {
        "run_id": job["run_id"],
        "condition_label": job["condition_label"],
        "training_seed": job["seed"],
        "interruption_step": job["position"],
        "return_code": result.returncode,
        "checkpoint_root": str(expected_checkpoint_root),
        "checkpoint_steps": expected_steps,
        "log_path": str(log_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen multi-position π0.5 training matrix")
    parser.add_argument("--mode", choices=("plan", "train"), default="plan")
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--run-date", type=lambda value: dt.date.fromisoformat(value), default=dt.date.today())
    parser.add_argument("--experiment-tag", default="mpos20261105")
    parser.add_argument("--without-weights-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.payload_root = args.payload_root.expanduser().resolve()
    args.openpi_root = args.openpi_root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    tag = _experiment_tag(args.experiment_tag)
    plan, _ = _load_plan(args.repo_root)
    jobs = _job_specs(
        plan=plan,
        run_date=args.run_date,
        tag=tag,
        include_weights_only=not args.without_weights_only,
    )
    schedules = _schedule_paths(plan, args.payload_root)
    manifest_id = make_run_id("mvp", "pi05", f"{tag}matrix", 0, run_date=args.run_date)
    manifest_path = args.payload_root / "runs" / manifest_id / "metadata" / "training_matrix.json"
    _preflight_targets(
        jobs=jobs,
        payload_root=args.payload_root,
        manifest_path=manifest_path,
        schedules=schedules,
    )
    if args.mode == "plan":
        print(json.dumps({
            "schema": "mvp_training_matrix.v1",
            "status": "plan_valid",
            "manifest_id": manifest_id,
            "model_id": plan.model_id,
            "dataset_revision": plan.dataset_revision,
            "training_seeds": list(plan.training_seeds),
            "interruption_steps": list(plan.interruption_steps),
            "schedule_sha256_by_seed": {str(seed): sha256_file(path) for seed, path in schedules.items()},
            "job_count": len(jobs),
            "jobs": jobs,
        }, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    python = args.openpi_root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise ContractError(f"OpenPI Python environment not found: {python}")
    if not (args.openpi_root / "scripts" / "train.py").is_file():
        raise ContractError(f"OpenPI source checkout is incomplete: {args.openpi_root}")
    model_config = load_yaml(args.repo_root / "safety_continuity" / "configs" / "models" / "pi05_libero_safety.yaml")
    model_assets = args.payload_root / "models" / "finetuned" / "pi05_libero_safety" / model_config["checkpoint_revision"]
    if not (model_assets / "params").is_dir():
        raise ContractError(f"frozen π0.5 model params are missing: {model_assets / 'params'}")
    git_dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=args.repo_root,
        capture_output=True, text=True, check=True,
    ).stdout
    if git_dirty.strip():
        raise ContractError("training mode requires a clean Git checkout")
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError("training mode requires a visible NVIDIA GPU") from exc

    output_root = manifest_path.parent.parent
    if output_root.exists() and any(output_root.iterdir()):
        raise ContractError(f"refusing to overwrite training matrix output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    matrix_manifest = {
        "schema": "mvp_training_matrix.v1",
        "status": "running",
        "manifest_id": manifest_id,
        "experiment_tag": tag,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "training_seeds": list(plan.training_seeds),
        "interruption_steps": list(plan.interruption_steps),
        "schedule_sha256_by_seed": {str(seed): sha256_file(path) for seed, path in schedules.items()},
        "jobs": jobs,
        "completed_jobs": [],
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(matrix_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = _run_env(args)
    try:
        for job in jobs:
            result = _execute_job(
                job=job,
                args=args,
                plan=plan,
                schedule_path=schedules[job["seed"]],
                env=env,
                log_root=output_root / "logs",
            )
            matrix_manifest["completed_jobs"].append(result)
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(matrix_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(manifest_path)
        matrix_manifest["status"] = "training_complete"
        matrix_manifest["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(matrix_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    except Exception as exc:
        matrix_manifest["status"] = "failed"
        matrix_manifest["error"] = str(exc)
        matrix_manifest["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(matrix_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
        raise
    print(json.dumps({"status": matrix_manifest["status"], "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
