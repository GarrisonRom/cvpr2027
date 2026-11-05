"""Stable JSONL schemas shared by evaluation, analysis and reporting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..common.errors import ContractError
from ..common.identity import validate_run_id

SAFE_OUTCOMES = {"safe", "unsafe", "refusal", "false_refusal", "unknown"}
INSTRUCTION_TYPES = {"benign", "harmful", "neutral", "other", "unknown"}


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value


def _require_finite_number(value: Any, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise ContractError(f"{field_name} must be a finite number")


@dataclass(frozen=True)
class UpdateSampleRef:
    """One deterministic training sample in the frozen update schedule."""

    sample_id: str
    dataset_id: str
    task_id: str
    episode_id: str
    augmentation_seed: int
    ordinal: int

    def __post_init__(self) -> None:
        _required_text(self.sample_id, "sample_id")
        _required_text(self.dataset_id, "dataset_id")
        _required_text(self.task_id, "task_id")
        _required_text(self.episode_id, "episode_id")
        if type(self.ordinal) is not int or type(self.augmentation_seed) is not int:
            raise ContractError("ordinal and augmentation_seed must be integers")
        if self.ordinal < 0 or self.augmentation_seed < 0:
            raise ContractError("ordinal and augmentation_seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "update_sample.v1",
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "augmentation_seed": self.augmentation_seed,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class BatchConsumptionRecord:
    """One example actually consumed by one optimizer update."""

    run_id: str
    condition_id: str
    training_seed: int
    sequence_index: int
    optimizer_step: int
    microbatch_index: int
    sample_position: int
    sample: UpdateSampleRef
    dataset_revision: str | None = None

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        _required_text(self.condition_id, "condition_id")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ContractError("training_seed must be a non-negative integer")
        for name in ("sequence_index", "microbatch_index", "sample_position"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")
        if type(self.optimizer_step) is not int or self.optimizer_step <= 0:
            raise ContractError("optimizer_step must be a positive integer")
        if not isinstance(self.sample, UpdateSampleRef):
            raise ContractError("consumed sample must be an UpdateSampleRef")
        if self.dataset_revision is not None and (
            not isinstance(self.dataset_revision, str) or not self.dataset_revision.strip()
        ):
            raise ContractError("dataset_revision must be non-empty text when present")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "batch_consumption.v1",
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "training_seed": self.training_seed,
            "sequence_index": self.sequence_index,
            "optimizer_step": self.optimizer_step,
            "microbatch_index": self.microbatch_index,
            "sample_position": self.sample_position,
            "sample": self.sample.to_dict(),
            "dataset_revision": self.dataset_revision,
        }

@dataclass(frozen=True)
class EpisodeRecord:
    """One row in episodes/episodes.jsonl; episode is the analysis observation."""

    run_id: str
    model_id: str
    condition_id: str
    benchmark_id: str
    suite_id: str
    task_id: str
    scene_id: str
    instruction_id: str
    instruction_type: str
    initial_state_seed: int
    training_seed: int
    model_version: str
    success: bool | None
    safe_outcome: str
    violation_type: str | None
    safety_cost: float | None
    exposure_duration_s: float | None
    completion_time_s: float | None
    termination_reason: str
    evaluation_seed: int | None = None
    interruption_step: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        for name in (
            "run_id",
            "model_id",
            "condition_id",
            "benchmark_id",
            "suite_id",
            "task_id",
            "scene_id",
            "instruction_id",
            "instruction_type",
            "model_version",
            "safe_outcome",
            "termination_reason",
        ):
            _required_text(getattr(self, name), name)
        if type(self.initial_state_seed) is not int or type(self.training_seed) is not int:
            raise ContractError("initial_state_seed and training_seed must be integers")
        if self.initial_state_seed < 0:
            raise ContractError("initial_state_seed must be non-negative")
        if self.training_seed < 0:
            raise ContractError("training_seed must be non-negative")
        if self.evaluation_seed is not None and (
            type(self.evaluation_seed) is not int or self.evaluation_seed < 0
        ):
            raise ContractError("evaluation_seed must be a non-negative integer")
        if self.interruption_step is not None and (
            type(self.interruption_step) is not int or self.interruption_step <= 0
        ):
            raise ContractError("interruption_step must be a positive integer when present")
        if self.success is not None and type(self.success) is not bool:
            raise ContractError("success must be a boolean or null")
        if self.safe_outcome not in SAFE_OUTCOMES:
            raise ContractError(f"unknown safe_outcome: {self.safe_outcome}")
        if self.instruction_type not in INSTRUCTION_TYPES:
            raise ContractError(f"unknown instruction_type: {self.instruction_type}")
        if self.safe_outcome == "false_refusal" and self.instruction_type != "benign":
            raise ContractError("false_refusal outcome requires a benign instruction")
        if self.safe_outcome == "refusal" and self.instruction_type == "benign":
            raise ContractError("refusal on a benign instruction must be labeled false_refusal")
        for name in ("safety_cost", "exposure_duration_s", "completion_time_s"):
            value = getattr(self, name)
            if value is not None:
                _require_finite_number(value, name)
                if value < 0:
                    raise ContractError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = "episode.v1"
        return payload


@dataclass(frozen=True)
class StepTraceRecord:
    """One optional per-step trace, aligned to the active policy version."""

    run_id: str
    episode_id: str
    timestep: int
    action: tuple[float, ...]
    active_policy_version: str
    action_chunk_id: str
    monitor_margins: Mapping[str, float]
    simulator_state_ref: str

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if isinstance(self.monitor_margins, Mapping):
            object.__setattr__(
                self,
                "monitor_margins",
                MappingProxyType(dict(self.monitor_margins)),
            )
        _required_text(self.run_id, "run_id")
        _required_text(self.episode_id, "episode_id")
        _required_text(self.active_policy_version, "active_policy_version")
        _required_text(self.action_chunk_id, "action_chunk_id")
        _required_text(self.simulator_state_ref, "simulator_state_ref")
        if type(self.timestep) is not int or self.timestep < 0:
            raise ContractError("step trace requires a non-negative timestep and action")
        if not isinstance(self.action, tuple) or not self.action:
            raise ContractError("step trace action must be a non-empty tuple")
        for value in self.action:
            _require_finite_number(value, "step trace action value")
        if not isinstance(self.monitor_margins, Mapping):
            raise ContractError("monitor_margins must be a mapping")
        for name, value in self.monitor_margins.items():
            _required_text(name, "monitor margin name")
            _require_finite_number(value, f"monitor margin {name}")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "schema": "step_trace.v1",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "timestep": self.timestep,
            "action": list(self.action),
            "active_policy_version": self.active_policy_version,
            "action_chunk_id": self.action_chunk_id,
            "monitor_margins": dict(self.monitor_margins),
            "simulator_state_ref": self.simulator_state_ref,
        }


def episode_from_mapping(payload: Mapping[str, Any]) -> EpisodeRecord:
    data = dict(payload)
    data.pop("schema", None)
    return EpisodeRecord(**data)


def update_sample_from_mapping(payload: Mapping[str, Any]) -> UpdateSampleRef:
    if payload.get("schema") != "update_sample.v1":
        raise ContractError(f"expected update sample schema, got {payload.get('schema')!r}")
    data = dict(payload)
    data.pop("schema", None)
    return UpdateSampleRef(**data)


def batch_consumption_from_mapping(payload: Mapping[str, Any]) -> BatchConsumptionRecord:
    if payload.get("schema") != "batch_consumption.v1":
        raise ContractError(f"expected batch consumption schema, got {payload.get('schema')!r}")
    data = dict(payload)
    data.pop("schema", None)
    sample_payload = data.get("sample")
    if not isinstance(sample_payload, Mapping):
        raise ContractError("batch consumption sample must be an object")
    data["sample"] = update_sample_from_mapping(sample_payload)
    return BatchConsumptionRecord(**data)


def step_trace_from_mapping(payload: Mapping[str, Any]) -> StepTraceRecord:
    data = dict(payload)
    data.pop("schema", None)
    if isinstance(data.get("action"), list):
        data["action"] = tuple(float(value) for value in data["action"])
    return StepTraceRecord(**data)


def validate_episode_batch(records: Sequence[EpisodeRecord]) -> None:
    """Reject duplicate pairing keys before aggregation can hide them."""
    seen: set[tuple[object, ...]] = set()
    for record in records:
        key = (
            record.model_id,
            record.benchmark_id,
            record.suite_id,
            record.task_id,
            record.scene_id,
            record.instruction_id,
            record.instruction_type,
            record.condition_id,
            record.initial_state_seed,
            record.evaluation_seed,
            record.interruption_step,
            record.training_seed,
        )
        if key in seen:
            raise ContractError(f"duplicate episode pairing key: {key}")
        seen.add(key)
