"""Configuration-only preflight; model loading belongs to later adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import CONTRACT_VERSION
from .common.config import load_yaml, require_schema_version
from .common.errors import ConfigurationError, ContractError
from .common.roots import RuntimeRoots
from .checkpointing.contracts import STATE_KEYS
from .training.mvp_plan import MvpExperimentPlan


@dataclass(frozen=True)
class PreflightIssue:
    severity: str
    path: str
    message: str


CONFIG_PATHS = (
    "configs/defaults.yaml",
    "configs/data/libero10_no_noops.yaml",
    "configs/data/libero_safety_eval.yaml",
    "configs/models/pi05_libero_safety.yaml",
    "configs/training/pi05_lora_mvp.yaml",
    "configs/interruption/mvp_matrix.yaml",
    "configs/evaluation/mvp.yaml",
    "configs/evaluation/self_built_mvp.yaml",
    "configs/analysis/mvp_primary.yaml",
    "configs/plotting/mvp_figures.yaml",
)


def _issue(severity: str, path: Path, message: str) -> PreflightIssue:
    return PreflightIssue(severity=severity, path=str(path), message=message)


def validate_project_configs(workspace_root: str | Path) -> list[PreflightIssue]:
    roots = RuntimeRoots.from_env(workspace_root)
    issues: list[PreflightIssue] = []
    loaded: dict[str, dict[str, Any]] = {}

    for relative in CONFIG_PATHS:
        path = roots.resolve_repo(f"safety_continuity/{relative}")
        try:
            payload = load_yaml(path)
            require_schema_version(payload)
            if payload.get("contract_version") != CONTRACT_VERSION:
                raise ConfigurationError(
                    f"expected contract_version={CONTRACT_VERSION}, "
                    f"got {payload.get('contract_version')!r}"
                )
            loaded[relative] = payload
        except ConfigurationError as exc:
            issues.append(_issue("error", path, str(exc)))

    for relative, payload in loaded.items():
        for key in ("manifest", "asset_manifest", "model_config", "update_dataset_config", "config"):
            reference = payload.get(key)
            if not isinstance(reference, str) or not reference:
                continue
            try:
                reference_path = roots.resolve_repo(reference)
            except ConfigurationError as exc:
                issues.append(_issue("error", roots.resolve_repo(f"safety_continuity/{relative}"),
                                     str(exc)))
                continue
            if not reference_path.is_file():
                issues.append(_issue("error", roots.resolve_repo(f"safety_continuity/{relative}"),
                                     f"referenced tracked file does not exist: {reference}"))
                continue
            if reference_path.suffix in {".yaml", ".yml"}:
                try:
                    referenced = load_yaml(reference_path)
                    require_schema_version(referenced)
                    if referenced.get("contract_version") != CONTRACT_VERSION:
                        issues.append(_issue(
                            "error",
                            reference_path,
                            f"expected contract_version={CONTRACT_VERSION}, "
                            f"got {referenced.get('contract_version')!r}",
                        ))
                except ConfigurationError as exc:
                    issues.append(_issue("error", reference_path, str(exc)))

    model = loaded.get("configs/models/pi05_libero_safety.yaml")
    if model:
        model_manifest_path = roots.resolve_repo(model["manifest"])
        if model_manifest_path.is_file():
            try:
                model_manifest = load_yaml(model_manifest_path)
            except ConfigurationError as exc:
                issues.append(_issue("error", model_manifest_path, str(exc)))
                model_manifest = {}
            if model_manifest.get("resource_id") != model.get("model_id"):
                issues.append(_issue(
                    "error", model_manifest_path,
                    "model manifest resource_id does not match model config model_id",
                ))
            if model_manifest.get("adapter_id") != model.get("adapter_id"):
                issues.append(_issue(
                    "error", model_manifest_path,
                    "model manifest adapter_id does not match model config adapter_id",
                ))
            if model_manifest.get("action_schema_id") != model.get("action", {}).get("schema_id"):
                issues.append(_issue(
                    "error", model_manifest_path,
                    "model manifest action_schema_id does not match model config action.schema_id",
                ))

    if model:
        action = model.get("action", {})
        for key in ("dimension", "chunk_length"):
            if action.get(key) is None:
                issues.append(_issue(
                    "warning",
                    roots.resolve_repo("safety_continuity/configs/models/pi05_libero_safety.yaml"),
                    f"model action.{key} is unresolved; model preflight must fill it before loading",
                ))
        observation = model.get("observation", {})
        unresolved_observation = {
            "camera_mapping": not observation.get("camera_mapping"),
            "language_key": not observation.get("language_key"),
            "proprioception_key": not observation.get("proprioception_key"),
        }
        for key, unresolved in unresolved_observation.items():
            if unresolved:
                issues.append(_issue(
                    "warning",
                    roots.resolve_repo("safety_continuity/configs/models/pi05_libero_safety.yaml"),
                    f"model observation.{key} is unresolved; adapter preflight must fill it",
                ))
        if (
            model.get("adapter", {}).get("enabled")
            and model.get("adapter", {}).get("type") == "lora"
            and not model.get("adapter", {}).get("target_modules")
        ):
            issues.append(_issue(
                "warning",
                roots.resolve_repo("safety_continuity/configs/models/pi05_libero_safety.yaml"),
                "LoRA adapter.target_modules is unresolved",
            ))

    training = loaded.get("configs/training/pi05_lora_mvp.yaml")
    if training:
        unresolved_training = []
        if training.get("optimizer", {}).get("learning_rate") is None:
            unresolved_training.append("optimizer.learning_rate")
        for key in ("gradient_accumulation_steps", "effective_batch_size"):
            if training.get("batch", {}).get(key) is None:
                unresolved_training.append(f"batch.{key}")
        for field_name in unresolved_training:
            issues.append(_issue(
                "warning",
                roots.resolve_repo("safety_continuity/configs/training/pi05_lora_mvp.yaml"),
                f"training {field_name} is unresolved; freeze it before a run",
            ))

    interruption = loaded.get("configs/interruption/mvp_matrix.yaml")
    if interruption:
        conditions = interruption.get("conditions", [])
        condition_ids = [condition.get("id") for condition in conditions]
        if (
            any(not isinstance(condition_id, str) or not condition_id for condition_id in condition_ids)
            or len(condition_ids) != len(set(condition_ids))
        ):
            issues.append(_issue(
                "error",
                roots.resolve_repo("safety_continuity/configs/interruption/mvp_matrix.yaml"),
                "interruption condition ids must be unique",
            ))
        known_states = set(STATE_KEYS)
        for condition in conditions:
            plan = condition.get("restore_plan")
            if plan is None:
                continue
            raw_preserve = plan.get("preserve", [])
            raw_reset = plan.get("reset", [])
            raw_required = plan.get("required", [])
            plan_lists = (raw_preserve, raw_reset, raw_required)
            if any(
                not isinstance(values, list)
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
                for values in plan_lists
            ):
                issues.append(_issue(
                    "error",
                    roots.resolve_repo("safety_continuity/configs/interruption/mvp_matrix.yaml"),
                    f"condition {condition.get('id')} restore plan fields must be unique string lists",
                ))
                continue
            preserve = set(raw_preserve)
            reset = set(raw_reset)
            required = set(raw_required)
            if (preserve | reset | required) - known_states:
                issues.append(_issue(
                    "error",
                    roots.resolve_repo("safety_continuity/configs/interruption/mvp_matrix.yaml"),
                    f"condition {condition.get('id')} uses unknown checkpoint state keys",
                ))
            elif (
                preserve & reset
                or (preserve | reset) != known_states
                or not required <= known_states
            ):
                issues.append(_issue(
                    "error",
                    roots.resolve_repo("safety_continuity/configs/interruption/mvp_matrix.yaml"),
                    f"condition {condition.get('id')} restore plan must classify every state exactly once",
                ))

    training = loaded.get("configs/training/pi05_lora_mvp.yaml")
    interruption = loaded.get("configs/interruption/mvp_matrix.yaml")
    evaluation = loaded.get("configs/evaluation/mvp.yaml")
    self_built = loaded.get("configs/evaluation/self_built_mvp.yaml")
    if training and interruption and evaluation and self_built:
        try:
            training_steps = training.get("schedule", {}).get("total_optimizer_steps")
            interruption_steps = interruption.get("experiment", {}).get("total_optimizer_steps")
            if training_steps != interruption_steps:
                raise ContractError(
                    "training.schedule.total_optimizer_steps and "
                    "interruption.experiment.total_optimizer_steps must match"
                )
            MvpExperimentPlan.from_mapping(
                training_config=training,
                interruption_config=interruption,
                evaluation_config=evaluation,
                self_built_config=self_built,
            )
        except (ConfigurationError, ContractError, TypeError, ValueError) as exc:
            issues.append(_issue(
                "error",
                roots.resolve_repo("safety_continuity/configs/interruption/mvp_matrix.yaml"),
                f"invalid multi-position MVP plan: {exc}",
            ))

    data = loaded.get("configs/data/libero10_no_noops.yaml")
    if data:
        split_config = data.get("splits", {})
        split = split_config.get("update_manifest")
        split_template = split_config.get("update_manifest_template")
        split_root_env = split_config.get("update_manifest_root_env", "SC_SPLIT_ROOT")
        split_references: list[str] = []
        if isinstance(split, str) and split:
            split_references.append(split)
        elif isinstance(split_template, str) and split_template:
            split_references.extend(
                split_template.format(seed=seed)
                for seed in (42, 43, 44)
            )
        for split_reference in split_references:
            split_path = roots.resolve_payload(split_root_env, split_reference)
            if not split_path.is_file():
                issues.append(_issue(
                    "warning",
                    roots.resolve_repo("safety_continuity/configs/data/libero10_no_noops.yaml"),
                    f"runtime split manifest is not present yet: {split_path}",
                ))

    return issues
