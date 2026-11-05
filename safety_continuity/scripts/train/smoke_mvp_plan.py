"""CPU smoke test for the multi-position MVP training/evaluation contracts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import date
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the multi-position MVP plan")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    package_root = workspace_root / "safety_continuity" / "src"
    sys.path.insert(0, str(package_root))

    from safety_continuity.common.config import load_yaml
    from safety_continuity.common.errors import ContractError
    from safety_continuity.common.identity import make_run_id
    from safety_continuity.data.io import read_update_sample_jsonl
    from safety_continuity.data.schemas import EpisodeRecord, validate_episode_batch
    from safety_continuity.training.mvp_plan import MvpExperimentPlan
    from safety_continuity.training.contracts import BatchManifest

    runner_path = workspace_root / "safety_continuity" / "scripts" / "train" / "openpi_mvp.py"
    runner_spec = importlib.util.spec_from_file_location("mvp_train_runner_smoke", runner_path)
    if runner_spec is None or runner_spec.loader is None:
        raise ContractError(f"cannot load native runner contract: {runner_path}")
    runner_module = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner_module)

    matrix_path = workspace_root / "safety_continuity" / "scripts" / "train" / "run_mvp_matrix.py"
    matrix_spec = importlib.util.spec_from_file_location("mvp_matrix_runner_smoke", matrix_path)
    if matrix_spec is None or matrix_spec.loader is None:
        raise ContractError(f"cannot load training matrix runner: {matrix_path}")
    matrix_module = importlib.util.module_from_spec(matrix_spec)
    matrix_spec.loader.exec_module(matrix_module)

    config_root = workspace_root / "safety_continuity" / "configs"
    plan = MvpExperimentPlan.from_mapping(
        training_config=load_yaml(config_root / "training" / "pi05_lora_mvp.yaml"),
        interruption_config=load_yaml(config_root / "interruption" / "mvp_matrix.yaml"),
        evaluation_config=load_yaml(config_root / "evaluation" / "mvp.yaml"),
        self_built_config=load_yaml(config_root / "evaluation" / "self_built_mvp.yaml"),
    )

    def must_reject(name: str, operation) -> None:
        try:
            operation()
        except ContractError:
            return
        raise ContractError(f"invalid contract case was accepted: {name}")

    training_config = load_yaml(config_root / "training" / "pi05_lora_mvp.yaml")
    interruption_config = load_yaml(config_root / "interruption" / "mvp_matrix.yaml")
    evaluation_config = load_yaml(config_root / "evaluation" / "mvp.yaml")
    self_built_config = load_yaml(config_root / "evaluation" / "self_built_mvp.yaml")
    bad_seed_config = {**training_config, "seeds": [11, 23, 37]}
    must_reject(
        "historical seeds cannot become primary seeds",
        lambda: MvpExperimentPlan.from_mapping(
            training_config=bad_seed_config,
            interruption_config=interruption_config,
            evaluation_config=evaluation_config,
            self_built_config=self_built_config,
        ),
    )
    bad_position_config = {
        **interruption_config,
        "experiment": {**interruption_config["experiment"], "positions": [500, 2000]},
    }
    must_reject(
        "interruption at the training horizon is invalid",
        lambda: MvpExperimentPlan.from_mapping(
            training_config=training_config,
            interruption_config=bad_position_config,
            evaluation_config=evaluation_config,
            self_built_config=self_built_config,
        ),
    )
    bad_episode_config = {
        **evaluation_config,
        "self_built_safety": {
            **evaluation_config["self_built_safety"],
            "episodes_per_task": evaluation_config["self_built_safety"]["episodes_per_task"] + 1,
        },
    }
    must_reject(
        "official and self-built episode counts must match",
        lambda: MvpExperimentPlan.from_mapping(
            training_config=training_config,
            interruption_config=interruption_config,
            evaluation_config=bad_episode_config,
            self_built_config=self_built_config,
        ),
    )
    mismatched_model_config = {
        **evaluation_config,
        "mvp": {**evaluation_config["mvp"], "model_id": "different_model"},
    }
    must_reject(
        "training and evaluation use the same frozen model",
        lambda: MvpExperimentPlan.from_mapping(
            training_config=training_config,
            interruption_config=interruption_config,
            evaluation_config=mismatched_model_config,
            self_built_config=self_built_config,
        ),
    )
    duplicate_benchmark_config = {
        **evaluation_config,
        "self_built_safety": {
            **evaluation_config["self_built_safety"],
            "benchmark_id": evaluation_config["official_safety"]["benchmark_id"],
        },
    }
    must_reject(
        "official and self-built benchmark IDs stay separate",
        lambda: MvpExperimentPlan.from_mapping(
            training_config=training_config,
            interruption_config=interruption_config,
            evaluation_config=duplicate_benchmark_config,
            self_built_config=self_built_config,
        ),
    )
    uncalibrated_monitor_config = {
        **self_built_config,
        "calibration": {
            **self_built_config["calibration"],
            "physical_contact_calibration_required": False,
        },
    }
    must_reject(
        "physical contact calibration cannot be disabled",
        lambda: MvpExperimentPlan.from_mapping(
            training_config=training_config,
            interruption_config=interruption_config,
            evaluation_config=evaluation_config,
            self_built_config=uncalibrated_monitor_config,
        ),
    )
    incorrectly_official_self_built = {
        **self_built_config,
        "official_labels": True,
    }
    must_reject(
        "self-built labels cannot claim official provenance",
        lambda: MvpExperimentPlan.from_mapping(
            training_config=training_config,
            interruption_config=interruption_config,
            evaluation_config=evaluation_config,
            self_built_config=incorrectly_official_self_built,
        ),
    )

    split_root = workspace_root / "data" / "splits"
    manifest_by_seed = {}
    for seed in plan.training_seeds:
        manifest_path = split_root / f"mvp_update_samples_s{seed}.jsonl"
        samples = read_update_sample_jsonl(manifest_path)
        manifest = BatchManifest(
            manifest_id="mvp_openpi_libero_v1",
            dataset_revision=plan.dataset_revision,
            samples=samples,
        )
        manifest.validate()
        if len(samples) < plan.total_steps:
            raise ContractError(f"schedule is shorter than total_steps: {manifest_path}")
        manifest_by_seed[seed] = samples

    comparison_slices = plan.all_slices()
    execution_slices = plan.execution_slices()
    jobs = matrix_module._job_specs(
        plan=plan,
        run_date=date(2026, 11, 5),
        tag="smokematrix",
        include_weights_only=True,
    )
    job_ids = [job["run_id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ContractError("training matrix generated duplicate run IDs")
    baseline_jobs = [job for job in jobs if job["condition"] == "A0"]
    if len(baseline_jobs) != len(plan.training_seeds):
        raise ContractError("matrix must create exactly one continuous baseline per seed")
    if any(job["checkpoint_at_steps"] != list(plan.interruption_steps) for job in baseline_jobs):
        raise ContractError("continuous runs do not retain every configured prefix checkpoint")
    branch_jobs = [job for job in jobs if job["condition"] in {"A1", "A2", "A3"}]
    if len(branch_jobs) != len(plan.training_seeds) * len(plan.interruption_steps) * 3:
        raise ContractError("matrix branch job count does not match position x seed x condition")
    if len({job["run_id"] for job in jobs}) != len(jobs):
        raise ContractError("matrix plan contains duplicate immutable run IDs")
    expected_package_root = workspace_root / "safety_continuity" / "src"
    if Path(matrix_module._runner_pythonpath(workspace_root / "openpi").split(os.pathsep)[0]) != expected_package_root:
        raise ContractError("training executor PYTHONPATH does not include the package source root")

    with tempfile.TemporaryDirectory(prefix="mvp_checkpoint_stage_smoke_") as temporary_root:
        root = Path(temporary_root)
        source_checkpoint = root / "source" / "499"
        (source_checkpoint / "params").mkdir(parents=True)
        (source_checkpoint / "params" / "weights.json").write_text('{"weight": 1}\n', encoding="utf-8")
        staged = matrix_module._stage_resume_checkpoint(
            source_checkpoint=source_checkpoint,
            destination_root=root / "staged_branch",
            step=499,
        )
        if not (staged / "params" / "weights.json").is_file():
            raise ContractError("resume checkpoint staging did not preserve the checkpoint tree")
        if not (source_checkpoint / "params" / "weights.json").is_file():
            raise ContractError("resume checkpoint staging modified the prefix checkpoint")
        must_reject(
            "staged branch checkpoint cannot be overwritten",
            lambda: matrix_module._stage_resume_checkpoint(
                source_checkpoint=source_checkpoint,
                destination_root=root / "staged_branch",
                step=499,
            ),
        )
    for item in plan.all_slices():
        runner_args = argparse.Namespace(
            condition=item.condition_id,
            seed=item.training_seed,
            total_steps=item.trainer_total_steps,
            experiment_total_steps=item.total_steps,
            schedule_start=item.schedule_start,
            schedule_count=item.schedule_count,
            optimizer_step_start=item.optimizer_step_start,
            interruption_step=item.interruption_step,
            checkpoint_at_steps=(
                list(plan.interruption_steps) if item.condition_id == "A0" else []
            ),
        )
        if runner_module._validate_runner_args(runner_args) != item:
            raise ContractError(f"native runner reinterpreted plan slice: {item}")
        if item.condition_id == "A0":
            expected_loop_steps = tuple(sorted({step - 1 for step in plan.interruption_steps} | {plan.total_steps - 1}))
            actual_loop_steps = runner_module._checkpoint_loop_steps(
                item.trainer_total_steps,
                tuple(runner_args.checkpoint_at_steps),
            )
            if actual_loop_steps != expected_loop_steps:
                raise ContractError(
                    f"continuous checkpoint retention mismatch: {actual_loop_steps} != {expected_loop_steps}"
                )
    saved_loop_steps: list[int] = []
    fake_train_lib = SimpleNamespace(
        _checkpoints=SimpleNamespace(
            save_state=lambda manager, state, loader, step: saved_loop_steps.append(step),
        ),
    )
    expected_loop_steps = runner_module._checkpoint_loop_steps(
        plan.total_steps,
        plan.interruption_steps,
    )
    runner_module._install_checkpoint_filter(fake_train_lib, expected_loop_steps)
    for loop_step in range(plan.total_steps):
        fake_train_lib._checkpoints.save_state(None, None, None, loop_step)
    if tuple(saved_loop_steps) != expected_loop_steps:
        raise ContractError(
            f"OpenPI checkpoint filter retained {saved_loop_steps}, expected {expected_loop_steps}"
        )
    invalid_a3 = argparse.Namespace(
        condition="A3",
        seed=42,
        total_steps=plan.total_steps - plan.interruption_steps[0],
        experiment_total_steps=plan.total_steps,
        schedule_start=plan.interruption_steps[0],
        schedule_count=plan.total_steps - plan.interruption_steps[0],
        optimizer_step_start=plan.interruption_steps[0],
        interruption_step=plan.interruption_steps[0],
        checkpoint_at_steps=[],
    )
    must_reject(
        "weights-only restart starts with a fresh optimizer",
        lambda: runner_module._validate_runner_args(invalid_a3),
    )
    for point in plan.interruption_steps:
        for seed in plan.training_seeds:
            samples = manifest_by_seed[seed]
            prefix = samples[:point]
            suffix = samples[point:plan.total_steps]
            point_slices = [
                item for item in comparison_slices
                if item.training_seed == seed
                and item.interruption_step == point
            ]
            by_condition = {item.condition_id: item for item in point_slices}
            if by_condition["PRE"].schedule_count != len(prefix):
                raise ContractError("PRE slice does not cover the interruption prefix")
            for condition in ("A1", "A2", "A3"):
                item = by_condition[condition]
                if item.schedule_start != point or item.schedule_count != len(suffix):
                    raise ContractError(f"{condition} slice does not cover the same suffix")
            if samples[:plan.total_steps] != prefix + suffix:
                raise ContractError("prefix/suffix reconstruction changed sample order")

    smoke_records: list[EpisodeRecord] = []
    for point in plan.interruption_steps:
        for condition in ("A1", "A2"):
            for seed in plan.training_seeds:
                run_id = make_run_id(
                    "smoke",
                    "pi05",
                    f"{condition.lower()}p{point}",
                    seed,
                    run_date=date(2026, 11, 5),
                )
                for task_index in range(2):
                    smoke_records.append(EpisodeRecord(
                        run_id=run_id,
                        model_id=plan.model_id,
                        condition_id=condition,
                        benchmark_id="libero_safety_self_built",
                        suite_id="human_safety",
                        task_id=f"smoke_task_{task_index}",
                        scene_id=f"smoke_scene_{task_index}",
                        instruction_id=f"smoke_instruction_{task_index}",
                        instruction_type="benign",
                        initial_state_seed=100 + task_index,
                        training_seed=seed,
                        model_version=f"{condition.lower()}_p{point}",
                        success=True,
                        safe_outcome="safe",
                        violation_type=None,
                        safety_cost=0.0,
                        exposure_duration_s=0.0,
                        completion_time_s=1.0,
                        termination_reason="success",
                        evaluation_seed=seed,
                        interruption_step=point,
                        metadata={
                            "protocol": "self_built_safety.v1",
                            "official_labels": False,
                            "smoke": True,
                        },
                    ))
                    smoke_records.append(EpisodeRecord(
                        run_id=run_id,
                        model_id=plan.model_id,
                        condition_id=condition,
                        benchmark_id="libero_safety_official",
                        suite_id="human_safety",
                        task_id=f"smoke_task_{task_index}",
                        scene_id=f"smoke_scene_{task_index}",
                        instruction_id=f"smoke_instruction_{task_index}",
                        instruction_type="benign",
                        initial_state_seed=100 + task_index,
                        training_seed=seed,
                        model_version=f"{condition.lower()}_p{point}",
                        success=True,
                        safe_outcome="unknown",
                        violation_type=None,
                        safety_cost=None,
                        exposure_duration_s=None,
                        completion_time_s=1.0,
                        termination_reason="success",
                        evaluation_seed=seed,
                        interruption_step=point,
                        metadata={"official_labels": True, "smoke": True},
                    ))
    validate_episode_batch(smoke_records)
    if {record.benchmark_id for record in smoke_records} != {
        "libero_safety_official", "libero_safety_self_built",
    }:
        raise ContractError("official and self-built safety records were mixed or omitted")

    print(json.dumps({
        "schema": "mvp_plan_smoke_report.v1",
        "status": "ok",
        "model_id": plan.model_id,
        "dataset_revision": plan.dataset_revision,
        "training_seeds": list(plan.training_seeds),
        "interruption_steps": list(plan.interruption_steps),
        "schedule_count_by_seed": {str(seed): len(samples) for seed, samples in manifest_by_seed.items()},
        "comparison_slice_count": len(comparison_slices),
        "execution_slice_count": len(execution_slices),
        "matrix_job_count": len(jobs),
        "continuous_baseline_count": len(baseline_jobs),
        "position_branch_job_count": len(branch_jobs),
        "retained_prefix_checkpoint_loop_steps": list(
            expected_loop_steps
        ),
        "synthetic_episode_count": len(smoke_records),
        "rejected_invalid_case_count": 9,
        "official_and_self_built_tracks_separate": True,
        "checkpoint_stage_smoke": "ok",
        "official_safety_enabled": plan.official_safety_enabled,
        "self_built_safety_enabled": plan.self_built_safety_enabled,
        "calibration_required": plan.calibration_required,
    }, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"mvp plan smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
