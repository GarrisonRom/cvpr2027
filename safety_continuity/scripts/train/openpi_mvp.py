"""Executable OpenPI MVP runner with deterministic data and resume audit.

This launcher intentionally keeps the native parameter subset used by the
released safety checkpoint. OpenPI's LoRA config adds parameters absent from
that checkpoint, so the MVP trains only ``action_out_proj``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from safety_continuity.common.config import sha256_file, sha256_payload
from safety_continuity.common.errors import ContractError
from safety_continuity.common.provenance import Provenance
from safety_continuity.common.roots import RuntimeRoots
from safety_continuity.data.schemas import (
    BatchConsumptionRecord,
    UpdateSampleRef,
)
from safety_continuity.training.contracts import BatchManifestRef
from safety_continuity.training.mvp_plan import MvpTrainingSlice


LIBERO10_TASKS = (
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the yellow and white mug in the microwave and close it",
    "put both moka pots on the stove",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "pick up the book and place it in the back compartment of the caddy",
)

LIBERO_DATASET_REVISION = "v2.0"


def _load_train_script(openpi_root: Path):
    script_path = openpi_root / "scripts" / "train.py"
    if not script_path.is_file():
        raise ContractError(f"OpenPI train script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("safety_continuity_openpi_train", script_path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load OpenPI train script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_openpi_modules(openpi_root: Path):
    source = str(openpi_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from flax import nnx
        from openpi.models import pi0_config
        from openpi.training import config as config_lib
        from openpi.training import data_loader
        from openpi.training import weight_loaders
        from openpi.training.config import AssetsConfig
        from openpi.shared import nnx_utils
    except ImportError as exc:
        raise ContractError(
            "OpenPI dependencies are not installed in the selected environment"
        ) from exc
    return nnx, nnx_utils, pi0_config, config_lib, data_loader, weight_loaders, AssetsConfig


def _official_episode_indices(repo_id: str) -> tuple[int, ...]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id, revision=LIBERO_DATASET_REVISION)
    first_by_task: dict[str, int] = {}
    for episode_index, episode in sorted(metadata.episodes.items()):
        for task in episode.get("tasks", []):
            first_by_task.setdefault(task.strip(), int(episode_index))
    missing = [task for task in LIBERO10_TASKS if task not in first_by_task]
    if missing:
        raise ContractError(f"LIBERO-10 tasks missing from {repo_id}: {missing}")
    return tuple(first_by_task[task] for task in LIBERO10_TASKS)


def _write_schedule(
    *,
    repo_id: str,
    schedule_path: Path,
    total_samples: int,
    selected_episode_indices: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id, revision=LIBERO_DATASET_REVISION)
    revision = str(metadata.revision)
    episode_lengths = {
        int(index): int(metadata.episodes[int(index)]["length"])
        for index in selected_episode_indices
    }
    schedule: list[UpdateSampleRef] = []
    for ordinal in range(total_samples):
        episode_index = selected_episode_indices[ordinal % len(selected_episode_indices)]
        episode_length = episode_lengths[episode_index]
        frame_index = (ordinal // len(selected_episode_indices)) % episode_length
        episode_tasks = metadata.episodes[episode_index].get("tasks", [])
        task_id = str(episode_tasks[0] if episode_tasks else f"episode_{episode_index}")
        schedule.append(UpdateSampleRef(
            sample_id=(
                f"{repo_id}@{revision}/episode-{episode_index}/frame-{frame_index}"
            ),
            dataset_id=f"{repo_id}@{revision}",
            task_id=task_id,
            episode_id=f"episode-{episode_index}",
            augmentation_seed=seed + ordinal,
            ordinal=ordinal,
        ))
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = schedule_path.with_suffix(schedule_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in schedule:
            handle.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")
    temporary.replace(schedule_path)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "selected_episode_indices": list(selected_episode_indices),
        "sample_count": len(schedule),
        "sha256": sha256_file(schedule_path),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _configure_tracking_dataset(
    data_loader: Any,
    *,
    selected_episode_indices: tuple[int, ...],
    schedule_samples: tuple[UpdateSampleRef, ...],
    schedule_start_ordinal: int,
    schedule_count: int,
    run_id: str,
    condition_id: str,
    training_seed: int,
    consumption_path: Path,
    optimizer_step_start: int,
) -> None:
    original_dataset = data_loader.lerobot_dataset.LeRobotDataset
    tracking_lock = threading.Lock()
    tracking_sequence = 0

    class TrackingDataset(original_dataset):
        def __init__(self, repo_id: str, *args: Any, **kwargs: Any) -> None:
            # LeRobot v2.0 indexes episode_data_index by the contiguous local
            # prefix, even when a non-contiguous episode subset is requested.
            # Load that prefix and map each frozen sample back to its episode
            # offset explicitly below.
            kwargs["episodes"] = list(range(max(selected_episode_indices) + 1))
            kwargs["download_videos"] = False
            kwargs["revision"] = LIBERO_DATASET_REVISION
            super().__init__(repo_id, *args, **kwargs)

        def __len__(self) -> int:
            # OpenPI fetches the next batch after the final optimizer update;
            # provide one non-consumed sentinel batch for that final fetch.
            return schedule_count + 1

        def __getitem__(self, index: int) -> Any:
            nonlocal tracking_sequence
            consumed = tracking_sequence < schedule_count
            global_ordinal = schedule_start_ordinal + min(tracking_sequence, schedule_count - 1)
            base_length = super().__len__()
            sample = schedule_samples[global_ordinal]
            try:
                episode_index = int(sample.episode_id.rsplit("-", 1)[1])
                frame_index = int(sample.sample_id.rsplit("/frame-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ContractError(f"cannot decode schedule sample id: {sample.sample_id}") from exc
            if episode_index >= len(self.meta.episodes):
                raise ContractError(f"schedule episode is absent from loaded dataset: {episode_index}")
            base_index = sum(
                int(self.meta.episodes[index]["length"])
                for index in range(episode_index)
            ) + frame_index
            raw = super().__getitem__(base_index % base_length)
            if not consumed:
                return raw
            record = BatchConsumptionRecord(
                run_id=run_id,
                condition_id=condition_id,
                training_seed=training_seed,
                sequence_index=tracking_sequence,
                optimizer_step=optimizer_step_start + tracking_sequence + 1,
                microbatch_index=0,
                sample_position=0,
                sample=sample,
                dataset_revision=sample.dataset_id,
            )
            with tracking_lock:
                consumption_path.parent.mkdir(parents=True, exist_ok=True)
                with consumption_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                tracking_sequence += 1
            return raw

    data_loader.lerobot_dataset.LeRobotDataset = TrackingDataset
    original_create = data_loader.create_torch_data_loader

    def create_fixed_loader(*args: Any, **kwargs: Any) -> Any:
        kwargs["shuffle"] = False
        kwargs["num_workers"] = 0
        kwargs["num_batches"] = schedule_count + 1
        return original_create(*args, **kwargs)

    data_loader.create_torch_data_loader = create_fixed_loader


def _reset_adam_moments(train_lib: Any) -> None:
    original_restore = train_lib._checkpoints.restore_state
    import jax

    def zero_tree(value: Any) -> Any:
        """Zero a parameter-shaped PyTree while preserving its container type."""

        def zero_leaf(leaf: Any) -> Any:
            if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                return jax.numpy.zeros_like(leaf)
            return leaf

        return jax.tree_util.tree_map(zero_leaf, value)

    def restore_with_moments_reset(checkpoint_manager: Any, state: Any, data_loader: Any,
                                   step: int | None = None) -> Any:
        restored = original_restore(checkpoint_manager, state, data_loader, step)
        opt_state = restored.opt_state
        # OpenPI's Adam state is [adamax? optional, scale_by_adam, ...]; reset
        # only moment tensors while preserving count and the schedule state.
        def reset_moments(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: (zero_tree(child) if key in {"mu", "nu"} else child)
                    for key, child in value.items()
                }
            # Optax states are NamedTuple PyTrees (for example,
            # ScaleByAdamState(count, mu, nu)); ``type(value) is tuple`` does
            # not match those types, so handle their named fields explicitly.
            if hasattr(value, "_fields") and hasattr(value, "_replace"):
                updates = {}
                for key in value._fields:
                    child = getattr(value, key)
                    updates[key] = zero_tree(child) if key in {"mu", "nu"} else reset_moments(child)
                return value._replace(**updates)
            if type(value) is list:
                return [reset_moments(child) for child in value]
            if type(value) is tuple:
                return tuple(reset_moments(child) for child in value)
            return value
        return dataclasses.replace(restored, opt_state=reset_moments(opt_state))

    train_lib._checkpoints.restore_state = restore_with_moments_reset


def _checkpoint_loop_steps(total_steps: int, optimizer_checkpoint_steps: tuple[int, ...]) -> tuple[int, ...]:
    """Map optimizer update counts to OpenPI loop indices and retain the final state."""

    if type(total_steps) is not int or total_steps <= 0:
        raise ContractError("total_steps must be a positive integer")
    if len(optimizer_checkpoint_steps) != len(set(optimizer_checkpoint_steps)):
        raise ContractError("checkpoint-at-steps must not contain duplicates")
    if any(type(step) is not int or not 0 < step <= total_steps for step in optimizer_checkpoint_steps):
        raise ContractError("checkpoint-at-steps must satisfy 0 < step <= total_steps")
    return tuple(sorted({*(step - 1 for step in optimizer_checkpoint_steps), total_steps - 1}))


def _install_checkpoint_filter(train_lib: Any, loop_steps: tuple[int, ...]) -> None:
    """Keep selected prefix states while OpenPI's loop calls save_state per step."""

    original_save_state = train_lib._checkpoints.save_state
    selected = frozenset(loop_steps)

    def save_selected(checkpoint_manager: Any, state: Any, data_loader: Any, step: Any) -> None:
        if int(step) in selected:
            original_save_state(checkpoint_manager, state, data_loader, int(step))

    train_lib._checkpoints.save_state = save_selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one controlled OpenPI MVP branch")
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", choices=["PRE", "A0", "A1", "A2", "A3"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument(
        "--experiment-total-steps",
        type=int,
        default=None,
        help="full schedule horizon; defaults to --total-steps for legacy runs",
    )
    parser.add_argument("--schedule-start", type=int, default=0)
    parser.add_argument("--schedule-count", type=int, required=True)
    parser.add_argument("--optimizer-step-start", type=int, default=0)
    parser.add_argument(
        "--interruption-step",
        type=int,
        default=None,
        help="validated multi-position plan boundary; omitted for legacy runs",
    )
    parser.add_argument(
        "--checkpoint-at-steps",
        type=int,
        nargs="*",
        default=[],
        help="optimizer update counts whose prefix checkpoints must be retained",
    )
    parser.add_argument("--schedule-path", type=Path, required=True)
    parser.add_argument("--selected-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--source-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-optimizer-moments", action="store_true")
    parser.add_argument("--checkpoint-name", required=True)
    return parser.parse_args()


def _validate_runner_args(args: argparse.Namespace) -> MvpTrainingSlice | None:
    if type(args.total_steps) is not int or args.total_steps <= 0:
        raise ContractError("total_steps must be a positive integer")
    experiment_total_steps = (
        args.total_steps
        if args.experiment_total_steps is None
        else args.experiment_total_steps
    )
    if type(experiment_total_steps) is not int or experiment_total_steps <= 0:
        raise ContractError("experiment_total_steps must be a positive integer")
    if args.schedule_start < 0 or args.schedule_count <= 0:
        raise ContractError("invalid schedule slice")
    if args.schedule_start + args.schedule_count > experiment_total_steps:
        raise ContractError("schedule slice exceeds the experiment horizon")
    if len(args.checkpoint_at_steps) != len(set(args.checkpoint_at_steps)):
        raise ContractError("checkpoint-at-steps must not contain duplicates")
    if any(type(step) is not int or not 0 < step <= experiment_total_steps for step in args.checkpoint_at_steps):
        raise ContractError("checkpoint-at-steps must be within the experiment horizon")
    if args.checkpoint_at_steps and args.condition != "A0":
        raise ContractError("multi-position prefix checkpoints can only be requested on continuous training")
    if any(step > args.total_steps for step in args.checkpoint_at_steps):
        raise ContractError("checkpoint-at-steps must be within this trainer invocation")
    if args.interruption_step is None:
        if args.condition == "A0" and args.experiment_total_steps is not None:
            planned_slice = MvpTrainingSlice(
                condition_id="A0",
                training_seed=args.seed,
                interruption_step=None,
                total_steps=experiment_total_steps,
                trainer_total_steps=args.total_steps,
                schedule_start=args.schedule_start,
                schedule_count=args.schedule_count,
                optimizer_step_start=args.optimizer_step_start,
            )
            planned_slice.validate()
            return planned_slice
        if args.schedule_start + args.schedule_count > args.total_steps and args.condition != "A3":
            raise ContractError("legacy schedule slice must fit within total_steps")
        return None
    if args.condition is None:
        raise ContractError("interruption_step requires a condition")
    planned_slice = MvpTrainingSlice(
        condition_id=args.condition,
        training_seed=args.seed,
        interruption_step=args.interruption_step,
        total_steps=experiment_total_steps,
        trainer_total_steps=args.total_steps,
        schedule_start=args.schedule_start,
        schedule_count=args.schedule_count,
        optimizer_step_start=args.optimizer_step_start,
    )
    planned_slice.validate()
    return planned_slice


def main() -> int:
    args = _parse_args()
    _validate_runner_args(args)
    experiment_total_steps = (
        args.total_steps
        if args.experiment_total_steps is None
        else args.experiment_total_steps
    )
    retained_checkpoint_steps = (
        _checkpoint_loop_steps(args.total_steps, tuple(args.checkpoint_at_steps))
        if args.checkpoint_at_steps else ()
    )
    roots = RuntimeRoots.from_env(args.openpi_root.parent)
    schedule_path = args.schedule_path.expanduser().resolve()
    if not schedule_path.is_file():
        if args.interruption_step is not None:
            raise ContractError(
                "multi-position plan requires its existing frozen schedule; implicit schedule creation is disabled"
            )
        selected = tuple(args.selected_episodes)
        _write_schedule(
            repo_id="physical-intelligence/libero",
            schedule_path=schedule_path,
            total_samples=experiment_total_steps,
            selected_episode_indices=selected,
            seed=args.seed,
        )
    import json as _json
    schedule_rows = []
    with schedule_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                from safety_continuity.data.schemas import update_sample_from_mapping
                schedule_rows.append(update_sample_from_mapping(_json.loads(line)))
    schedule_samples = tuple(schedule_rows)
    if len(schedule_samples) < experiment_total_steps:
        raise ContractError("schedule file is shorter than the full experiment horizon")

    run_dir = roots.run_root / args.run_id
    if run_dir.exists():
        raise ContractError(f"refusing to overwrite existing training run: {run_dir}")
    for name in ("metadata", "events", "episodes", "metrics", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    resolved = {
        "runner": "openpi_mvp.v1",
        "run_id": args.run_id,
        "condition": args.condition,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "experiment_total_steps": experiment_total_steps,
        "schedule_start": args.schedule_start,
        "schedule_count": args.schedule_count,
        "interruption_step": args.interruption_step,
        "checkpoint_at_steps": list(args.checkpoint_at_steps),
        "checkpoint_loop_steps": list(retained_checkpoint_steps),
        "schedule_path": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
        "selected_episodes": list(args.selected_episodes),
        "trainable_modules": ["action_out_proj"],
        "source_checkpoint_dir": str(args.source_checkpoint_dir) if args.source_checkpoint_dir else None,
    }
    resolved_path = run_dir / "metadata" / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    provenance = Provenance(
        run_id=args.run_id,
        git_commit=os.environ.get("SC_GIT_COMMIT", "unknown"),
        git_dirty=False,
        config_hash=sha256_payload(resolved),
        model_id="pi05_libero_safety",
        model_revision="e66d84bfec17770384550593e39975f7e21e3463",
        processor_revision="e66d84bfec17770384550593e39975f7e21e3463",
        dataset_manifest="physical_intelligence_libero@v2.0",
        split_id="mvp_update_samples",
        model_seed=args.seed,
        evaluation_seed=args.seed,
        gpu_name=os.environ.get("SC_GPU_NAME", "unknown"),
        visible_memory_bytes=int(os.environ.get("SC_GPU_MEMORY_BYTES", "0")),
        driver=os.environ.get("SC_GPU_DRIVER", "unknown"),
        python_version=sys.version.split()[0],
        framework_versions={"openpi": "pinned_external_checkout", "runner": "openpi_mvp.v1"},
        simulator_version="not_run_in_training",
        started_at_utc=now,
        extra={
            "condition": args.condition,
            "schedule_start": args.schedule_start,
            "interruption_step": args.interruption_step,
            "experiment_total_steps": experiment_total_steps,
        },
    )
    _write_json(run_dir / "metadata" / "provenance.json", provenance.to_dict())
    ref = BatchManifestRef(
        run_id=args.run_id,
        condition_id=args.condition or "PRE",
        training_seed=args.seed,
        manifest_id="mvp_openpi_libero_v1",
        dataset_revision="physical-intelligence/libero@v2.0",
        root_env="SC_SPLIT_ROOT",
        relative_path=schedule_path.name,
        sha256=sha256_file(schedule_path),
        sample_count=len(schedule_samples),
        expected_optimizer_steps=args.schedule_count,
        schedule_start_ordinal=args.schedule_start,
    )
    _write_json(run_dir / "metadata" / "batch_manifest.json", ref.to_dict())
    consumption_path = run_dir / "events" / "batch_consumption.jsonl"
    if consumption_path.exists():
        raise ContractError(f"refusing to overwrite existing batch trace: {consumption_path}")

    nnx, nnx_utils, pi0_config, config_lib, data_loader, weight_loaders, AssetsConfig = \
        _load_openpi_modules(args.openpi_root)
    train_lib = _load_train_script(args.openpi_root)
    if retained_checkpoint_steps:
        _install_checkpoint_filter(train_lib, retained_checkpoint_steps)
    _configure_tracking_dataset(
        data_loader,
        selected_episode_indices=tuple(args.selected_episodes),
        schedule_samples=schedule_samples,
        schedule_start_ordinal=args.schedule_start,
        schedule_count=args.schedule_count,
        run_id=args.run_id,
        condition_id=args.condition or "PRE",
        training_seed=args.seed,
        consumption_path=consumption_path,
        optimizer_step_start=args.optimizer_step_start,
    )
    if args.reset_optimizer_moments:
        _reset_adam_moments(train_lib)
        _write_json(
            run_dir / "events" / "optimizer_moments_reset.json",
            {
                "schema": "optimizer_moments_reset.v1",
                "run_id": args.run_id,
                "condition_id": args.condition,
                "source_checkpoint_dir": str(args.source_checkpoint_dir) if args.source_checkpoint_dir else None,
                "reset_state_keys": ["optimizer_moments"],
                "preserved_state_keys": [
                    "optimizer_structure",
                    "optimizer_step",
                    "scheduler_and_global_step",
                    "rng",
                    "batch_cursor",
                    "normalization",
                    "gradient_accumulation",
                ],
            },
        )
    from openpi.models import pi0_config as _pi0_config
    model = config_lib.get_config("pi05_libero").model
    freeze_filter = nnx.All(nnx.Not(nnx_utils.PathRegex(".*action_out_proj.*")))
    pi_root = Path(os.environ["SC_MODEL_ROOT"]) / "finetuned/pi05_libero_safety" / "e66d84bfec17770384550593e39975f7e21e3463"
    base = config_lib.get_config("pi05_libero")
    data = dataclasses.replace(base.data, assets=AssetsConfig(assets_dir=str(pi_root / "assets"), asset_id="lerobot"))
    checkpoint_base = Path(os.environ["SC_CHECKPOINT_ROOT"]) / "openpi_mvp"
    config = dataclasses.replace(
        base,
        name="pi05_libero_mvp",
        exp_name=args.checkpoint_name,
        model=model,
        freeze_filter=freeze_filter,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(args.source_checkpoint_dir / "params") if args.source_checkpoint_dir and not args.resume
            else str(pi_root / "params")
        ),
        assets_base_dir=str(Path(os.environ["SC_CACHE_ROOT"]) / "openpi/assets"),
        checkpoint_base_dir=str(checkpoint_base),
        seed=args.seed,
        batch_size=1,
        num_workers=0,
        num_train_steps=args.total_steps,
        log_interval=10,
        save_interval=1 if args.checkpoint_at_steps else args.total_steps,
        keep_period=1 if args.checkpoint_at_steps else None,
        overwrite=not args.resume,
        resume=args.resume,
        wandb_enabled=False,
        ema_decay=None,
        fsdp_devices=1,
    )
    train_lib.main(config)
    checkpoint_run_root = checkpoint_base / config.name / config.exp_name
    expected_checkpoints = retained_checkpoint_steps or (args.total_steps - 1,)
    missing_checkpoints = [
        checkpoint_run_root / str(step)
        for step in expected_checkpoints
        if not (checkpoint_run_root / str(step)).is_dir()
    ]
    if missing_checkpoints:
        raise ContractError(f"OpenPI did not retain required checkpoints: {missing_checkpoints}")
    ended = datetime.now(timezone.utc).isoformat()
    payload = json.loads((run_dir / "metadata" / "provenance.json").read_text(encoding="utf-8"))
    payload["ended_at_utc"] = ended
    _write_json(run_dir / "metadata" / "provenance.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
