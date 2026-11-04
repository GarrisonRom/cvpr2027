"""Run the self-built LIBERO-Safety MVP protocol.

Examples:

    python run_self_built_mvp.py --mode smoke --output-root runs/self_built_smoke
    python run_self_built_mvp.py --mode full --suite human_safety \
        --payload-root /mnt/cvpr2027-payloads --openpi-root /mnt/openpi

Full mode starts one OpenPI websocket server for each configured training condition. The
generated labels are explicitly self-built and are never reported as official
LIBERO-Safety labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safety_continuity.common.errors import ContractError
from safety_continuity.common.identity import make_run_id
from safety_continuity.common.config import sha256_payload
from safety_continuity.common.provenance import Provenance, provenance_from_mapping
from safety_continuity.data.schemas import EpisodeRecord, StepTraceRecord, validate_episode_batch
from safety_continuity.evaluation.self_built import (
    PROTOCOL_VERSION,
    SelfBuiltRuleMonitor,
    generate_libero_state_manifest,
    generate_synthetic_manifest,
    load_state_manifest,
)
from safety_continuity.evaluation.self_built_runner import run_libero_self_built


CONDITIONS = ("A0", "A1", "A2", "A3")
CONDITION_NAMES = {
    "continuous_training": "A0",
    "full_state_resume": "A1",
    "optimizer_moment_reset": "A2",
    "weights_only_restart": "A3",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def _synthetic_smoke(output_root: Path) -> list[dict[str, Any]]:
    manifest_path = generate_synthetic_manifest(output_root / "state_manifest", seed=7)
    manifest = load_state_manifest(manifest_path)
    results: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        run_id = make_run_id("selfbuiltsmoke", "pi05", condition.lower(), 0, run_date=dt.date(2026, 11, 4))
        run_dir = output_root / run_id
        episode_rows: list[EpisodeRecord] = []
        trace_rows: list[StepTraceRecord] = []
        for task_index, task in enumerate(manifest.tasks):
            monitor = SelfBuiltRuleMonitor(
                suite_id="human_safety",
                harmful_instruction=False,
            )
            episode_id = f"{task.task_id}_e000"
            for timestep in range(3):
                dangerous = condition in {"A2", "A3"} and task_index == 1 and timestep == 1
                state = {
                    "ncon": 1 if dangerous else 0,
                    "contact_names": ["human_hand"] if dangerous else [],
                }
                result = monitor.observe(simulator_state=state, action=(0.1,) * 7)
                trace_rows.append(
                    StepTraceRecord(
                        run_id=run_id,
                        episode_id=episode_id,
                        timestep=timestep,
                        action=(0.1,) * 7,
                        active_policy_version=f"smoke_{condition.lower()}",
                        action_chunk_id=f"{episode_id}_chunk_0000",
                        monitor_margins=dict(result.margins),
                        simulator_state_ref=f"{manifest.manifest_sha256()}:{task.task_id}",
                    )
                )
            success = task_index == 0
            final_result = monitor.finalize(success=success)
            episode_rows.append(
                EpisodeRecord(
                    run_id=run_id,
                    model_id="pi05_libero_safety",
                    condition_id=condition,
                    benchmark_id="libero_safety_self_built",
                    suite_id="human_safety",
                    task_id=task.task_id,
                    scene_id=task.task_id,
                    instruction_id=task.task_id,
                    instruction_type="neutral",
                    initial_state_seed=manifest.generation_seed,
                    training_seed=0,
                    model_version=f"smoke_{condition.lower()}",
                    success=success,
                    safe_outcome=final_result.outcome,
                    violation_type=final_result.violation_type,
                    safety_cost=final_result.safety_cost,
                    exposure_duration_s=None,
                    completion_time_s=None,
                    termination_reason="success" if success else "timeout",
                    evaluation_seed=0,
                    metadata={
                        "protocol": PROTOCOL_VERSION,
                        "official_labels": False,
                        "smoke": True,
                    },
                )
            )
        validate_episode_batch(episode_rows)
        _write_jsonl(run_dir / "episodes" / "episodes.jsonl", [row.to_dict() for row in episode_rows])
        _write_jsonl(run_dir / "episodes" / "step_traces" / "steps.jsonl", [row.to_dict() for row in trace_rows])
        summary = {
            "schema": "self_built_smoke_summary.v1",
            "protocol": PROTOCOL_VERSION,
            "run_id": run_id,
            "condition_id": condition,
            "episode_count": len(episode_rows),
            "success_count": sum(row.success is True for row in episode_rows),
            "unsafe_count": sum(row.safe_outcome == "unsafe" for row in episode_rows),
            "official_labels": False,
            "state_manifest": str(manifest_path),
        }
        _write_json(run_dir / "metrics" / "summary.json", summary)
        results.append(summary)
    _write_json(output_root / "smoke_summary.json", {"schema": "self_built_smoke_index.v1", "runs": results})
    return results


def _wait_for_server(process: subprocess.Popen[str], log_path: Path, timeout_s: int = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if "server listening" in log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else False:
            return
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
            raise ContractError(f"policy server exited with code {process.returncode}: {tail}")
        time.sleep(2)
    raise ContractError(f"policy server did not become ready within {timeout_s}s: {log_path}")


def _checkpoint_path(
    checkpoint_root: Path,
    condition: str,
    seed: int,
    weights_only_step: int,
    checkpoint_prefix: str = "",
) -> Path:
    step = weights_only_step if condition == "A3" else 1999
    name = f"{checkpoint_prefix}_{condition.lower()}_s{seed}" if checkpoint_prefix else f"{condition.lower()}_s{seed}"
    path = checkpoint_root / "openpi_mvp" / "pi05_libero_mvp" / name / str(step)
    if not path.is_dir():
        raise ContractError(f"missing {condition} checkpoint: {path}")
    return path


def _full(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.payload_root or not args.openpi_root:
        raise ContractError("full mode requires --payload-root and --openpi-root")
    if args.suite == "all":
        suites = list(("human_safety", "reasoning_safety"))
    else:
        suites = [args.suite]
    if args.condition == "all":
        conditions = CONDITIONS
    else:
        condition = CONDITION_NAMES.get(args.condition, args.condition)
        if condition not in CONDITIONS:
            raise ContractError(f"unsupported condition name: {args.condition}")
        conditions = [condition]
    output_root = args.output_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    state_root = output_root / "state_manifest"
    summaries: list[dict[str, Any]] = []
    for suite in suites:
        if args.state_manifest is not None:
            manifest_path = args.state_manifest.resolve()
            manifest = load_state_manifest(manifest_path)
            if manifest.suite_id != suite or manifest.level != args.level:
                raise ContractError("provided state manifest does not match suite and level")
            if manifest.states_per_task != args.states_per_task:
                raise ContractError("provided state manifest has a different states_per_task")
        else:
            manifest_path = generate_libero_state_manifest(
                suite_id=suite,
                level=args.level,
                output_root=state_root,
                generation_seed=args.state_seed,
                states_per_task=args.states_per_task,
                camera_size=args.camera_size,
                overwrite=args.overwrite_states,
            )
            manifest = load_state_manifest(manifest_path)
        for condition in conditions:
            checkpoint = _checkpoint_path(
                checkpoint_root,
                condition,
                args.seed,
                args.weights_only_step,
                args.checkpoint_prefix,
            )
            run_id = make_run_id(
                f"selfbuilt{suite.split('_')[0]}",
                "pi05",
                condition.lower(),
                args.seed,
                run_date=args.run_date,
            )
            run_dir = output_root / run_id
            if run_dir.exists() and any(run_dir.iterdir()):
                raise ContractError(f"refusing to overwrite existing evaluation run: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
            resolved = {
                "schema": "self_built_resolved_config.v1",
                "protocol": PROTOCOL_VERSION,
                "run_id": run_id,
                "condition": condition,
                "suite": suite,
                "level": args.level,
                "seed": args.seed,
                "evaluation_seed": args.eval_seed,
                "source_training_run_id": args.source_training_run_id,
                "checkpoint": str(checkpoint),
                "checkpoint_prefix": args.checkpoint_prefix,
                "state_manifest": str(manifest_path),
                "state_manifest_sha256": manifest.manifest_sha256(),
                "states_per_task": manifest.states_per_task,
                "episodes_per_task": args.episodes_per_task,
                "replan_steps": args.replan_steps,
                "num_steps_wait": args.num_steps_wait,
                "max_steps": args.max_steps,
                "official_labels": False,
            }
            metadata_dir = run_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            _write_json(metadata_dir / "resolved_config.json", resolved)
            import yaml

            (metadata_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=True, allow_unicode=True),
                encoding="utf-8",
            )
            started_at = dt.datetime.now(dt.UTC).isoformat()
            server_log = run_dir / "logs" / "policy_server.log"
            server_log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["OPENPI_DATA_HOME"] = str(args.payload_root / "cache" / "openpi")
            server_cmd = [
                str(args.openpi_root / ".venv" / "bin" / "python"),
                str(args.openpi_root / "scripts" / "serve_policy.py"),
                "--env",
                "LIBERO",
                "--port",
                str(args.port),
                "policy:checkpoint",
                f"--policy.config={args.policy_config}",
                f"--policy.dir={checkpoint}",
            ]
            with server_log.open("w", encoding="utf-8") as log_handle:
                server = subprocess.Popen(server_cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            try:
                _wait_for_server(server, server_log)
                summary = run_libero_self_built(
                    manifest_path=manifest_path,
                    run_id=run_id,
                    condition_id=condition,
                    model_version=str(checkpoint),
                    training_seed=args.seed,
                    evaluation_seed=args.eval_seed,
                    policy_host="127.0.0.1",
                    policy_port=args.port,
                    episodes_per_task=args.episodes_per_task,
                    output_episode_path=run_dir / "episodes" / "episodes.jsonl",
                    output_trace_path=run_dir / "episodes" / "step_traces" / "steps.jsonl",
                    video_dir=run_dir / "videos" if args.save_videos else None,
                    replan_steps=args.replan_steps,
                    num_steps_wait=args.num_steps_wait,
                    max_steps=args.max_steps,
                )
                _write_json(run_dir / "metrics" / "summary.json", summary)
                training_run_id = args.source_training_run_id
                training_provenance = None
                if training_run_id:
                    training_provenance_path = (
                        args.payload_root / "runs" / training_run_id
                        / "metadata" / "provenance.json"
                    )
                    training_provenance = provenance_from_mapping(
                        json.loads(training_provenance_path.read_text(encoding="utf-8"))
                    )
                try:
                    mujoco_version = importlib.metadata.version("mujoco")
                except importlib.metadata.PackageNotFoundError:
                    mujoco_version = "unavailable"
                try:
                    torch_version = importlib.metadata.version("torch")
                except importlib.metadata.PackageNotFoundError:
                    torch_version = "unavailable"
                ended_at = dt.datetime.now(dt.UTC).isoformat()
                provenance = Provenance(
                    run_id=run_id,
                    git_commit=os.environ.get("SC_GIT_COMMIT", "unknown"),
                    git_dirty=os.environ.get("SC_GIT_DIRTY", "false").lower() == "true",
                    config_hash=sha256_payload(resolved),
                    model_id="pi05_libero_safety",
                    model_revision=(
                        training_provenance.model_revision if training_provenance
                        else "not_recorded"
                    ),
                    processor_revision=(
                        training_provenance.processor_revision if training_provenance
                        else "not_recorded"
                    ),
                    dataset_manifest=f"{suite}_L{args.level}_self_built@{manifest.manifest_sha256()}",
                    split_id=f"{suite}_L{args.level}_self_built_states",
                    model_seed=args.seed,
                    evaluation_seed=args.eval_seed,
                    gpu_name=os.environ.get("SC_GPU_NAME", "unknown"),
                    visible_memory_bytes=int(os.environ.get("SC_GPU_MEMORY_BYTES", "0")),
                    driver=os.environ.get("SC_GPU_DRIVER", "unknown"),
                    python_version=platform.python_version(),
                    framework_versions={
                        "torch": torch_version,
                        "mujoco": mujoco_version,
                        "monitor": PROTOCOL_VERSION,
                    },
                    simulator_version=f"mujoco-{mujoco_version}; libero_version_unrecorded",
                    started_at_utc=started_at,
                    ended_at_utc=ended_at,
                    extra={
                        "checkpoint": str(checkpoint),
                        "source_training_run_id": training_run_id,
                        "source_training_git_commit": (
                            training_provenance.git_commit if training_provenance else None
                        ),
                        "official_labels": False,
                        "provenance_scope": "self_built_evaluation_run",
                    },
                )
                _write_json(metadata_dir / "provenance.json", provenance.to_dict())
                summaries.append(summary)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    index_path = output_root / (
        f"full_summary_{args.run_date:%Y%m%d}_{args.suite}_{args.condition.lower()}_s{args.seed}.json"
    )
    if index_path.exists():
        raise ContractError(f"refusing to overwrite existing full-run index: {index_path}")
    _write_json(index_path, {"schema": "self_built_full_index.v1", "runs": summaries})
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-built safety-continuity MVP")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("human_safety", "reasoning_safety", "all"), default="human_safety")
    parser.add_argument(
        "--condition",
        default="all",
        help="continuous_training, full_state_resume, optimizer_moment_reset, weights_only_restart, or all",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eval-seed", type=int, default=23)
    parser.add_argument("--state-seed", type=int, default=20261104)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--states-per-task", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--payload-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--openpi-root", type=Path, default=None)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--weights-only-step", dest="weights_only_step", type=int, default=999)
    parser.add_argument("--checkpoint-prefix", default="")
    parser.add_argument("--state-manifest", type=Path, default=None)
    parser.add_argument("--source-training-run-id", default=None)
    parser.add_argument("--overwrite-states", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--run-date", type=lambda value: dt.date.fromisoformat(value), default=dt.date.today())
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "smoke":
        results = _synthetic_smoke(args.output_root.resolve())
    else:
        if args.payload_root is None:
            args.payload_root = Path(os.environ.get("SC_DATA_ROOT", ".")).resolve().parent
        if args.checkpoint_root is None:
            args.checkpoint_root = Path(os.environ.get("SC_CHECKPOINT_ROOT", args.payload_root / "checkpoints"))
        if args.openpi_root is None:
            raise ContractError("full mode requires --openpi-root")
        results = _full(args)
    print(json.dumps({"schema": "self_built_mvp_index.v1", "runs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
