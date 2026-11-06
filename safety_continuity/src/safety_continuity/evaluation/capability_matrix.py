"""Paired LIBERO-10 capability evaluation for a frozen training matrix."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..common.config import sha256_file, sha256_payload
from ..common.errors import ContractError
from ..common.identity import make_run_id
from ..data.schemas import EpisodeRecord

CONDITIONS = {
    "Continuous training": "A0",
    "Full-state resume": "A1",
    "Optimizer-moment reset": "A2",
    "Weights-only restart": "A3",
}
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def create_libero10_state_manifest(libero_root: Path, output_root: Path, evaluation_seed: int) -> Path:
    """Freeze the first ten official LIBERO-10 states per task for all conditions."""
    import numpy as np

    sys.path.insert(0, str(libero_root))
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    if suite.n_tasks != 10:
        raise ContractError(f"expected 10 LIBERO-10 tasks, found {suite.n_tasks}")
    states_dir = output_root / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    task_rows = []
    bddl_root = Path(get_libero_path("bddl_files"))
    for task_index in range(suite.n_tasks):
        task = suite.get_task(task_index)
        states = np.asarray(suite.get_task_init_states(task_index))
        if states.dtype.hasobject or states.ndim < 2 or len(states) < 10:
            raise ContractError(f"invalid official initial-state array for task {task_index}")
        states = np.ascontiguousarray(states[:10])
        if not np.isfinite(states).all():
            raise ContractError(f"non-finite official initial state for task {task_index}")
        task_id = f"libero10_task_{task_index:02d}"
        key = f"task_{task_index:02d}"
        arrays[key] = states
        bddl_path = bddl_root / task.problem_folder / task.bddl_file
        if not bddl_path.is_file():
            raise ContractError(f"missing LIBERO-10 BDDL file: {bddl_path}")
        task_rows.append({
            "task_id": task_id,
            "task_index": task_index,
            "task_name": str(task.language),
            "bddl_path": str(bddl_path.resolve()),
            "bddl_sha256": sha256_file(bddl_path),
            "states_key": key,
            "state_count": 10,
            "state_shape": list(states.shape),
            "state_sha256": [_sha256_bytes(state.tobytes()) for state in states],
        })
    state_file = states_dir / "libero10_official_init_states.npz"
    if state_file.exists():
        raise ContractError(f"refusing to overwrite state payload: {state_file}")
    np.savez_compressed(state_file, **arrays)
    manifest = {
        "schema": "libero10_capability_state_manifest.v1",
        "benchmark_id": "libero10",
        "suite_id": "libero10_official",
        "evaluation_seed": evaluation_seed,
        "episodes_per_task": 10,
        "state_file": str(state_file.resolve()),
        "state_file_sha256": sha256_file(state_file),
        "tasks": task_rows,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    path = output_root / "libero10_capability_state_manifest.json"
    write_json(path, manifest)
    return path


def validate_state_manifest(path: Path) -> dict[str, Any]:
    import numpy as np

    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("manifest_sha256", None)
    if payload.get("schema") != "libero10_capability_state_manifest.v1":
        raise ContractError("unsupported LIBERO-10 capability state manifest")
    if digest != sha256_payload(payload):
        raise ContractError("LIBERO-10 capability state manifest hash mismatch")
    if (payload.get("benchmark_id") != "libero10" or payload.get("suite_id") != "libero10_official"
            or type(payload.get("evaluation_seed")) is not int or payload["evaluation_seed"] < 0
            or payload.get("episodes_per_task") != 10):
        raise ContractError("invalid LIBERO-10 capability manifest identity or episode count")
    state_file = Path(payload["state_file"])
    if not state_file.is_file() or sha256_file(state_file) != payload["state_file_sha256"]:
        raise ContractError("LIBERO-10 frozen state payload is missing or corrupted")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 10:
        raise ContractError("LIBERO-10 manifest must contain ten tasks")
    with np.load(state_file, allow_pickle=False) as saved_states:
        for index, task in enumerate(tasks):
            if (
                task.get("task_index") != index
                or task.get("task_id") != f"libero10_task_{index:02d}"
                or task.get("states_key") != f"task_{index:02d}"
                or task.get("state_count") != 10
            ):
                raise ContractError("LIBERO-10 manifest task order or state count is invalid")
            bddl = Path(task.get("bddl_path", ""))
            if not bddl.is_file() or sha256_file(bddl) != task.get("bddl_sha256"):
                raise ContractError(f"LIBERO-10 BDDL source is missing or changed: {bddl}")
            if task["states_key"] not in saved_states:
                raise ContractError(f"LIBERO-10 state array is missing: {task['states_key']}")
            states = saved_states[task["states_key"]]
            if list(states.shape) != task.get("state_shape") or len(states) != 10 or states.dtype.hasobject:
                raise ContractError(f"LIBERO-10 state array shape mismatch for {task['task_id']}")
            hashes = [_sha256_bytes(np.ascontiguousarray(state).tobytes()) for state in states]
            if hashes != task.get("state_sha256"):
                raise ContractError(f"LIBERO-10 initial-state content mismatch for {task['task_id']}")
    payload["manifest_sha256"] = digest
    return payload


def _wait_for_server(process: subprocess.Popen[str], log_path: Path) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if "server listening" in text.lower():
            return
        if process.poll() is not None:
            raise ContractError(f"policy server exited; inspect {log_path}")
        time.sleep(2)
    raise ContractError(f"policy server did not become ready: {log_path}")


def _quat2axisangle(quat: Any) -> Any:
    import numpy as np

    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator


def _run_job(*, job: dict[str, Any], manifest: dict[str, Any],
             output_root: Path, args: Any, env: dict[str, str], run_date: dt.date,
             episodes_per_task: int) -> dict[str, Any]:
    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools, websocket_client_policy

    condition_id = CONDITIONS[job["condition_label"]]
    np.random.seed(manifest["evaluation_seed"])
    interruption_step = job["interruption_step"]
    condition_slug = condition_id.lower() if interruption_step is None else f"{condition_id.lower()}p{interruption_step}"
    run_id = make_run_id("capmatrix", "pi05", condition_slug, job["training_seed"], run_date=run_date)
    run_dir = output_root / "evaluations" / job["run_id"] / run_id
    if run_dir.exists():
        raise ContractError(f"refusing to overwrite capability evaluation: {run_dir}")
    run_dir.mkdir(parents=True)
    checkpoint = Path(job["checkpoint_root"]) / str(job["checkpoint_steps"][-1])
    server_log = run_dir / "logs" / "policy_server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    server_cmd = [
        str(args.openpi_root / ".venv" / "bin" / "python"),
        str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate" / "serve_paired_policy.py"),
        "--checkpoint", str(checkpoint), "--config", "pi05_libero", "--port", str(args.port),
    ]
    with server_log.open("w", encoding="utf-8") as server_handle:
        server = subprocess.Popen(
            server_cmd, cwd=args.repo_root, env=env,
            stdout=server_handle, stderr=subprocess.STDOUT, text=True,
        )
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        errors = 0
        successes = 0
        records: list[dict[str, Any]] = []
        try:
            _wait_for_server(server, server_log)
            client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", args.port)
            suite = benchmark.get_benchmark_dict()["libero_10"]()
            with np.load(manifest["state_file"], allow_pickle=False) as saved_states:
                for task_row in manifest["tasks"]:
                    task_index = task_row["task_index"]
                    task = suite.get_task(task_index)
                    if str(task.language) != task_row["task_name"]:
                        raise ContractError(f"LIBERO task registry differs for {task_row['task_id']}")
                    states = saved_states[task_row["states_key"]]
                    if len(states) < episodes_per_task:
                        raise ContractError(f"frozen states missing for {task_row['task_id']}")
                    env_args = {
                        "bddl_file_name": task_row["bddl_path"],
                        "camera_heights": 256,
                        "camera_widths": 256,
                    }
                    sim_env = OffScreenRenderEnv(**env_args)
                    try:
                        sim_env.seed(manifest["evaluation_seed"])
                        for episode_index in range(episodes_per_task):
                            sim_env.reset()
                            obs = sim_env.set_init_state(states[episode_index])
                            action_plan: collections.deque[Any] = collections.deque()
                            done = False
                            termination = "timeout"
                            chunk_index = 0
                            step = 0
                            while step < 520 + 10:
                                if step < 10:
                                    obs, _, done, _ = sim_env.step(LIBERO_DUMMY_ACTION)
                                    step += 1
                                    continue
                                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                                image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
                                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                                if not action_plan:
                                    element = {
                                        "observation/image": image,
                                        "observation/wrist_image": wrist,
                                        "observation/state": np.concatenate((
                                            obs["robot0_eef_pos"],
                                            _quat2axisangle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"],
                                        )),
                                        "prompt": str(task.language),
                                        "_sc_inference_seed": manifest["evaluation_seed"] + task_index * 10 + episode_index,
                                        "_sc_query_index": chunk_index,
                                    }
                                    action_chunk = client.infer(element)["actions"]
                                    if len(action_chunk) < 5:
                                        raise ContractError("policy returned fewer than five actions")
                                    action_plan.extend(action_chunk[:5])
                                    chunk_index += 1
                                action = action_plan.popleft()
                                action_values = [float(value) for value in action]
                                if not action_values or not all(math.isfinite(value) for value in action_values):
                                    raise ContractError("policy returned an invalid action")
                                obs, _, done, _ = sim_env.step(action_values)
                                step += 1
                                if done:
                                    termination = "success"
                                    break
                            if termination == "success":
                                successes += 1
                            record = EpisodeRecord(
                                run_id=run_id,
                                model_id="pi05_libero_safety",
                                condition_id=condition_id,
                                benchmark_id="libero10",
                                suite_id="libero10_official",
                                task_id=task_row["task_id"],
                                scene_id=task_row["task_id"],
                                instruction_id=task_row["task_id"],
                                instruction_type="benign",
                                initial_state_seed=episode_index,
                                training_seed=job["training_seed"],
                                model_version=job["run_id"],
                                success=bool(done),
                                safe_outcome="unknown",
                                violation_type=None,
                                safety_cost=None,
                                exposure_duration_s=None,
                                completion_time_s=None,
                                termination_reason=termination,
                                evaluation_seed=manifest["evaluation_seed"],
                                interruption_step=interruption_step,
                                metadata={
                                    "protocol": "libero10_official_paired.v1",
                                    "official_safety_labels": False,
                                    "official_state_index": episode_index,
                                    "state_manifest_sha256": manifest["manifest_sha256"],
                                    "source_training_run_id": job["run_id"],
                                    "checkpoint": str(checkpoint),
                                    "paired_inference_rng": True,
                                    "inference_seed": manifest["evaluation_seed"] + task_index * 10 + episode_index,
                                    "inference_query_count": chunk_index,
                                },
                            )
                            records.append(record.to_dict())
                            print(json.dumps({
                                "event": "episode_completed",
                                "run_id": run_id,
                                "task": task_row["task_id"],
                                "episode": episode_index,
                                "success": bool(done),
                                "termination_reason": termination,
                            }), flush=True)
                    except Exception as exc:
                        errors += 1
                        print(json.dumps({"event": "task_error", "task": task_row["task_id"],
                                          "error": repr(exc)}), flush=True)
                        raise
                    finally:
                        sim_env.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    episode_path = run_dir / "episodes" / "episodes.jsonl"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    if errors:
        raise ContractError(f"capability evaluation has {errors} task errors: {run_id}")
    expected = len(manifest["tasks"]) * episodes_per_task
    if len(records) != expected:
        raise ContractError(f"capability episode count mismatch for {run_id}: {len(records)} != {expected}")
    ended = dt.datetime.now(dt.timezone.utc).isoformat()
    resolved = {
        "schema": "libero10_capability_resolved.v1",
        "run_id": run_id,
        "condition_id": condition_id,
        "training_seed": job["training_seed"],
        "interruption_step": interruption_step,
        "evaluation_seed": manifest["evaluation_seed"],
        "episodes_per_task": episodes_per_task,
        "state_manifest_sha256": manifest["manifest_sha256"],
        "source_training_run_id": job["run_id"],
        "checkpoint": str(checkpoint),
        "paired_inference_rng": True,
        "official_safety_labels": False,
    }
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata_dir / "resolved_config.json", resolved)
    write_json(metadata_dir / "provenance.json", {
        "schema": "capability_provenance.v1",
        "run_id": run_id,
        "model_id": "pi05_libero_safety",
        "condition_id": condition_id,
        "training_seed": job["training_seed"],
        "started_at_utc": started,
        "ended_at_utc": ended,
        "source_training_run_id": job["run_id"],
        "resolved_config_sha256": sha256_payload(resolved),
        "episodes_sha256": sha256_file(episode_path),
        "state_manifest_sha256": manifest["manifest_sha256"],
        "git_commit": os.environ.get("SC_GIT_COMMIT", "unknown"),
        "gpu_name": os.environ.get("SC_GPU_NAME", "unknown"),
        "gpu_driver": os.environ.get("SC_GPU_DRIVER", "unknown"),
    })
    summary = {
        "schema": "libero10_capability_summary.v1",
        "run_id": run_id,
        "condition_id": condition_id,
        "training_seed": job["training_seed"],
        "interruption_step": interruption_step,
        "episode_count": len(records),
        "success_count": successes,
        "success_rate": successes / len(records),
        "task_errors": errors,
        "state_manifest_sha256": manifest["manifest_sha256"],
    }
    write_json(run_dir / "metrics" / "summary.json", summary)
    return {
        "source_training_run_id": job["run_id"],
        "condition_id": condition_id,
        "training_seed": job["training_seed"],
        "interruption_step": interruption_step,
        "episode_count": len(records),
        "success_count": successes,
        "run_id": run_id,
        "summary": str(run_dir / "metrics" / "summary.json"),
    }


def wait_for_gpu(minimum_free_mib: int, status: dict[str, Any], status_path: Path) -> None:
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.free",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        rows = result.stdout.strip().splitlines()
        if len(rows) != 1:
            raise ContractError("capability matrix runner requires one visible GPU")
        total, free = (int(value.strip()) for value in rows[0].split(","))
        if total < minimum_free_mib:
            raise ContractError("GPU does not meet requested total memory threshold")
        status["gpu_free_mib"] = free
        if free >= minimum_free_mib:
            status["status"] = "running"
            write_json(status_path, status)
            return
        status["status"] = "waiting_for_gpu"
        status["last_gpu_check_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(status_path, status)
        time.sleep(30)


def run_matrix(args: Any) -> int:
    matrix = json.loads(args.training_matrix.read_text(encoding="utf-8"))
    runner_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "evaluate"
        / "run_matrix_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("matrix_evaluation_contracts", runner_path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load training matrix validator: {runner_path}")
    matrix_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix_runner)
    jobs = matrix_runner.select_jobs(matrix)
    if args.mode == "plan":
        print(json.dumps({"training_job_count": len(jobs), "tasks": 10,
                          "episodes_per_task": 10, "capability_episode_count": 3000,
                          "continuous_checkpoint_reused_across_positions": True}, indent=2))
        return 0
    if args.output_root.exists():
        raise ContractError(f"refusing to overwrite capability output: {args.output_root}")
    args.output_root.mkdir(parents=True)
    env = os.environ.copy()
    env.update({
        "MUJOCO_GL": "egl",
        "LIBERO_CONFIG_PATH": str(args.libero_config),
        "PYTHONPATH": os.pathsep.join((str(args.libero_root), str(args.repo_root / "src"), str(args.openpi_root / "src"))),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OPENPI_DATA_HOME": str(args.payload_root / "cache" / "openpi"),
        "HF_HOME": str(args.payload_root / "cache" / "huggingface"),
    })
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()[0].split(",")
    env.update({
        "SC_GPU_NAME": gpu[0].strip(),
        "SC_GPU_MEMORY_BYTES": str(int(gpu[1].strip()) * 1024 * 1024),
        "SC_GPU_DRIVER": gpu[2].strip(),
        "SC_GIT_COMMIT": args.git_commit,
    })
    os.environ.update(env)
    state_manifest_path = args.state_manifest
    if state_manifest_path is None:
        state_manifest_path = create_libero10_state_manifest(args.libero_root, args.output_root / "state_manifest",
                                                              args.evaluation_seed)
    state_manifest = validate_state_manifest(state_manifest_path)
    if state_manifest["evaluation_seed"] != args.evaluation_seed:
        raise ContractError("state manifest evaluation seed differs from requested seed")
    selected_jobs = jobs[:1] if args.mode == "smoke" else jobs
    episode_count = 1 if args.mode == "smoke" else 10
    status_path = args.output_root / "metadata" / "capability_matrix.json"
    status = {"schema": "libero10_capability_matrix.v1", "status": "running",
              "training_matrix_sha256": sha256_file(args.training_matrix),
              "state_manifest_sha256": state_manifest["manifest_sha256"],
              "planned_jobs": len(selected_jobs), "episodes_per_task": episode_count,
              "completed_jobs": [], "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(status_path, status)
    try:
        for job in selected_jobs:
            checkpoint = Path(job["checkpoint_root"]) / str(job["checkpoint_steps"][-1])
            if not (checkpoint / "params").is_dir() or not (checkpoint / "assets" / "lerobot" / "norm_stats.json").is_file():
                raise ContractError(f"incomplete inference checkpoint: {checkpoint}")
            wait_for_gpu(args.minimum_free_gpu_mib, status, status_path)
            result = _run_job(job=job, manifest=state_manifest, output_root=args.output_root, args=args, env=env,
                              run_date=dt.date.today(), episodes_per_task=episode_count)
            status["completed_jobs"].append(result)
            write_json(status_path, status)
        status["status"] = "smoke_complete" if args.mode == "smoke" else "capability_rollouts_complete"
        status["ended_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(status_path, status)
        return 0
    except BaseException as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        status["ended_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(status_path, status)
        raise
