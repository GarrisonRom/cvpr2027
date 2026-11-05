"""Print the validated multi-position MVP training matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and expand the multi-position MVP plan")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--without-weights-only", action="store_true")
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    package_root = workspace_root / "safety_continuity" / "src"
    sys.path.insert(0, str(package_root))
    from safety_continuity.common.config import load_yaml
    from safety_continuity.training.mvp_plan import MvpExperimentPlan

    config_root = workspace_root / "safety_continuity" / "configs"
    plan = MvpExperimentPlan.from_mapping(
        training_config=load_yaml(config_root / "training" / "pi05_lora_mvp.yaml"),
        interruption_config=load_yaml(config_root / "interruption" / "mvp_matrix.yaml"),
        evaluation_config=load_yaml(config_root / "evaluation" / "mvp.yaml"),
        self_built_config=load_yaml(config_root / "evaluation" / "self_built_mvp.yaml"),
    )
    slices = plan.all_slices(include_weights_only=not args.without_weights_only)
    execution_slices = plan.execution_slices(include_weights_only=not args.without_weights_only)
    print(json.dumps({
        "schema": "mvp_plan_expansion.v1",
        "model_id": plan.model_id,
        "dataset_revision": plan.dataset_revision,
        "total_steps": plan.total_steps,
        "interruption_steps": list(plan.interruption_steps),
        "training_seeds": list(plan.training_seeds),
        "episodes_per_task": plan.episodes_per_task,
        "official_safety_enabled": plan.official_safety_enabled,
        "self_built_safety_enabled": plan.self_built_safety_enabled,
        "calibration_required": plan.calibration_required,
        "slice_count": len(slices),
        "execution_slice_count": len(execution_slices),
        "checkpoint_schedules": list(plan.checkpoint_schedules()),
        "slices": [
            {
                "condition_id": item.condition_id,
                "training_seed": item.training_seed,
                "interruption_step": item.interruption_step,
                "total_steps": item.total_steps,
                "schedule_start": item.schedule_start,
                "schedule_count": item.schedule_count,
                "optimizer_step_start": item.optimizer_step_start,
            }
            for item in slices
        ],
    }, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
