"""Validated multi-position MVP experiment plan.

The plan layer owns the experiment matrix. Native training and simulator
adapters consume its slices but do not infer interruption positions or safety
track selection from directory names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..common.errors import ContractError


PRIMARY_TRAINING_SEEDS = (42, 43, 44)
CONDITION_IDS = ("A0", "A1", "A2", "A3")
RESUME_CONDITION_IDS = ("A1", "A2", "A3")


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ContractError(f"{field_name} must be a positive integer")
    return value


def _unique_ints(values: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ContractError(f"{field_name} must be a non-empty list")
    if any(type(value) is not int or value < 0 for value in values):
        raise ContractError(f"{field_name} must contain non-negative integers")
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ContractError(f"{field_name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class InterruptionPoint:
    """One legal interruption boundary inside a fixed training horizon."""

    step: int
    total_steps: int

    def __post_init__(self) -> None:
        _positive_int(self.total_steps, "total_steps")
        if type(self.step) is not int or not 0 < self.step < self.total_steps:
            raise ContractError(
                f"interruption step must satisfy 0 < step < total_steps; "
                f"got {self.step!r} for total_steps={self.total_steps}"
            )

    @property
    def prefix_steps(self) -> int:
        return self.step

    @property
    def suffix_steps(self) -> int:
        return self.total_steps - self.step


@dataclass(frozen=True)
class MvpTrainingSlice:
    """A deterministic schedule slice consumed by one training branch."""

    condition_id: str
    training_seed: int
    interruption_step: int | None
    total_steps: int
    trainer_total_steps: int
    schedule_start: int
    schedule_count: int
    optimizer_step_start: int

    def validate(self) -> None:
        if self.condition_id not in (*CONDITION_IDS, "PRE"):
            raise ContractError(f"unsupported MVP condition id: {self.condition_id!r}")
        if type(self.training_seed) is not int or self.training_seed not in PRIMARY_TRAINING_SEEDS:
            raise ContractError(
                f"MVP training seed must be one of {PRIMARY_TRAINING_SEEDS}; "
                f"got {self.training_seed!r}"
            )
        _positive_int(self.total_steps, "total_steps")
        _positive_int(self.trainer_total_steps, "trainer_total_steps")
        if type(self.schedule_start) is not int or self.schedule_start < 0:
            raise ContractError("schedule_start must be a non-negative integer")
        _positive_int(self.schedule_count, "schedule_count")
        if type(self.optimizer_step_start) is not int or self.optimizer_step_start < 0:
            raise ContractError("optimizer_step_start must be a non-negative integer")
        if self.schedule_start + self.schedule_count > self.total_steps:
            raise ContractError("training schedule slice exceeds total_steps")
        if self.interruption_step is not None:
            InterruptionPoint(self.interruption_step, self.total_steps)
        if self.condition_id == "A0":
            if self.interruption_step is not None or self.schedule_start != 0:
                raise ContractError("continuous training must consume the full schedule from zero")
            if (
                self.schedule_count != self.total_steps
                or self.trainer_total_steps != self.total_steps
                or self.optimizer_step_start != 0
            ):
                raise ContractError("continuous training slice must cover the full horizon")
        elif self.condition_id == "PRE":
            if (
                self.interruption_step is None
                or self.schedule_count != self.interruption_step
                or self.trainer_total_steps != self.interruption_step
            ):
                raise ContractError("interruption checkpoint slice must end at its interruption step")
            if self.schedule_start != 0 or self.optimizer_step_start != 0:
                raise ContractError("interruption checkpoint must start at optimizer step zero")
        elif self.condition_id in {"A1", "A2"}:
            if self.interruption_step is None:
                raise ContractError(f"{self.condition_id} requires an interruption step")
            point = InterruptionPoint(self.interruption_step, self.total_steps)
            if (
                self.schedule_start != point.step
                or self.schedule_count != point.suffix_steps
                or self.trainer_total_steps != self.total_steps
            ):
                raise ContractError(f"{self.condition_id} must consume the post-interruption suffix")
            if self.optimizer_step_start != point.step:
                raise ContractError(f"{self.condition_id} must preserve the optimizer step at resume")
        else:
            if self.interruption_step is None:
                raise ContractError("A3 requires an interruption step")
            point = InterruptionPoint(self.interruption_step, self.total_steps)
            if (
                self.schedule_start != point.step
                or self.schedule_count != point.suffix_steps
                or self.trainer_total_steps != point.suffix_steps
            ):
                raise ContractError("A3 must train for the suffix length on the frozen post-interruption schedule")
            if self.optimizer_step_start != 0:
                raise ContractError("A3 starts with a fresh optimizer at step zero")


@dataclass(frozen=True)
class MvpExperimentPlan:
    """The frozen single-model, single-update-plan MVP matrix."""

    model_id: str
    dataset_revision: str
    total_steps: int
    interruption_steps: tuple[int, ...]
    training_seeds: tuple[int, ...]
    episodes_per_task: int
    official_safety_enabled: bool
    self_built_safety_enabled: bool
    calibration_required: bool

    def validate(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ContractError("MVP model_id is required")
        if not isinstance(self.dataset_revision, str) or not self.dataset_revision.strip():
            raise ContractError("MVP dataset_revision is required")
        _positive_int(self.total_steps, "total_steps")
        if not self.interruption_steps:
            raise ContractError("MVP requires at least one interruption step")
        points = [InterruptionPoint(step, self.total_steps) for step in self.interruption_steps]
        if tuple(sorted(point.step for point in points)) != self.interruption_steps:
            raise ContractError("MVP interruption_steps must be strictly increasing")
        if self.training_seeds != PRIMARY_TRAINING_SEEDS:
            raise ContractError(
                f"MVP training seeds are fixed to {PRIMARY_TRAINING_SEEDS}; "
                f"got {self.training_seeds!r}"
            )
        _positive_int(self.episodes_per_task, "episodes_per_task")
        if type(self.official_safety_enabled) is not bool:
            raise ContractError("official_safety_enabled must be boolean")
        if type(self.self_built_safety_enabled) is not bool:
            raise ContractError("self_built_safety_enabled must be boolean")
        if not self.official_safety_enabled or not self.self_built_safety_enabled:
            raise ContractError("MVP requires both official and self-built safety tracks")
        if type(self.calibration_required) is not bool:
            raise ContractError("calibration_required must be boolean")
        if self.calibration_required is not True:
            raise ContractError("MVP requires monitor calibration before safety claims")

    @classmethod
    def from_mapping(
        cls,
        *,
        training_config: Mapping[str, Any],
        interruption_config: Mapping[str, Any],
        evaluation_config: Mapping[str, Any],
        self_built_config: Mapping[str, Any],
    ) -> "MvpExperimentPlan":
        training_schedule = training_config.get("schedule", {})
        interruption = interruption_config.get("experiment", {})
        evaluation = evaluation_config
        mvp = evaluation_config.get("mvp", {})
        official = evaluation_config.get("official_safety", {})
        self_built = evaluation_config.get("self_built_safety", {})
        total_steps = _positive_int(
            training_schedule.get("total_optimizer_steps"),
            "schedule.total_optimizer_steps",
        )
        configured_total_steps = interruption.get("total_optimizer_steps")
        if configured_total_steps is not None and configured_total_steps != total_steps:
            raise ContractError(
                "interruption.experiment.total_optimizer_steps must match training schedule"
            )
        interruption_steps = _unique_ints(
            interruption.get("positions"),
            "experiment.positions",
        )
        training_seeds = tuple(
            _unique_ints(training_config.get("seeds"), "training.seeds")
        )
        official_episodes = _positive_int(
            official.get("episodes_per_task_condition_seed"),
            "official_safety.episodes_per_task_condition_seed",
        )
        self_built_episodes = _positive_int(
            self_built.get("episodes_per_task"),
            "self_built_safety.episodes_per_task",
        )
        configured_self_built_episodes = self_built_config.get("episodes_per_task")
        if configured_self_built_episodes is not None and configured_self_built_episodes != self_built_episodes:
            raise ContractError(
                "evaluation/self_built_mvp.yaml episodes_per_task must match evaluation/mvp.yaml"
            )
        if official_episodes != self_built_episodes:
            raise ContractError(
                "official and self-built tracks must use the same episodes_per_task"
            )
        if evaluation.get("training_seeds") != list(PRIMARY_TRAINING_SEEDS):
            raise ContractError("evaluation.training_seeds must equal [42, 43, 44]")
        if self_built_config.get("training_seeds") != list(PRIMARY_TRAINING_SEEDS):
            raise ContractError("self-built training_seeds must equal [42, 43, 44]")
        if evaluation_config.get("report", {}).get("use_official_safety_labels") is not True:
            raise ContractError("official safety track must use official safety labels")
        official_benchmark_id = official.get("benchmark_id")
        self_built_benchmark_id = self_built.get("benchmark_id")
        if (
            not isinstance(official_benchmark_id, str)
            or not official_benchmark_id.strip()
            or not isinstance(self_built_benchmark_id, str)
            or not self_built_benchmark_id.strip()
        ):
            raise ContractError("official and self-built safety tracks need benchmark ids")
        if official_benchmark_id == self_built_benchmark_id:
            raise ContractError("official and self-built safety tracks must use distinct benchmark ids")
        if self_built_config.get("official_labels") is not False:
            raise ContractError("self-built safety track must set official_labels=false")
        if self_built_config.get("outputs", {}).get("official_safety_metric") != "prohibited":
            raise ContractError("self-built track must prohibit official safety metrics")
        calibration_config = self_built_config.get("calibration", {})
        if calibration_config.get("physical_contact_calibration_required") is not True:
            raise ContractError("physical contact calibration is required before safety interpretation")
        fixed_update_plan = mvp.get("fixed_update_plan")
        dataset_revision = str(
            training_config.get("dataset_revision") or "physical-intelligence/libero@v2.0"
        )
        if fixed_update_plan is not None and fixed_update_plan != dataset_revision:
            raise ContractError("evaluation.mvp.fixed_update_plan must match training.dataset_revision")
        expected_model_id = "pi05_libero_safety"
        if (
            training_config.get("model_id") != expected_model_id
            or mvp.get("model_id") != expected_model_id
        ):
            raise ContractError(
                f"the frozen MVP model id must be {expected_model_id!r} in training and evaluation configs"
            )
        plan = cls(
            model_id=expected_model_id,
            dataset_revision=dataset_revision,
            total_steps=total_steps,
            interruption_steps=interruption_steps,
            training_seeds=training_seeds,
            episodes_per_task=official_episodes,
            official_safety_enabled=official.get("enabled"),
            self_built_safety_enabled=self_built.get("enabled"),
            calibration_required=calibration_config.get("required"),
        )
        plan.validate()
        if interruption.get("primary_conditions") != ["A1", "A2"]:
            raise ContractError("primary_conditions must be full-state resume and optimizer reset")
        return plan

    def slices_for_position(self, step: int, *, include_weights_only: bool = True) -> tuple[MvpTrainingSlice, ...]:
        point = InterruptionPoint(step, self.total_steps)
        condition_ids = ("PRE", "A0", "A1", "A2", "A3") if include_weights_only else ("PRE", "A0", "A1", "A2")
        slices: list[MvpTrainingSlice] = []
        for seed in self.training_seeds:
            for condition_id in condition_ids:
                if condition_id == "A0":
                    item = MvpTrainingSlice(
                        condition_id, seed, None, self.total_steps, self.total_steps,
                        0, self.total_steps, 0,
                    )
                elif condition_id == "PRE":
                    item = MvpTrainingSlice(
                        condition_id, seed, point.step, self.total_steps, point.prefix_steps,
                        0, point.prefix_steps, 0,
                    )
                elif condition_id == "A3":
                    item = MvpTrainingSlice(
                        condition_id, seed, point.step, self.total_steps, point.suffix_steps,
                        point.step, point.suffix_steps, 0,
                    )
                else:
                    item = MvpTrainingSlice(
                        condition_id, seed, point.step, self.total_steps, self.total_steps,
                        point.step, point.suffix_steps, point.step,
                    )
                item.validate()
                slices.append(item)
        return tuple(slices)

    def all_slices(self, *, include_weights_only: bool = True) -> tuple[MvpTrainingSlice, ...]:
        self.validate()
        return tuple(
            item
            for step in self.interruption_steps
            for item in self.slices_for_position(step, include_weights_only=include_weights_only)
        )

    def checkpoint_schedules(self) -> tuple[dict[str, Any], ...]:
        """Return one continuous prefix schedule per seed.

        A single continuous run produces all interruption checkpoints for its
        seed. Prefixes must not be trained independently for each position.
        """

        self.validate()
        return tuple({
            "training_seed": seed,
            "total_steps": self.total_steps,
            "checkpoint_steps": list(self.interruption_steps),
        } for seed in self.training_seeds)

    def execution_slices(self, *, include_weights_only: bool = True) -> tuple[MvpTrainingSlice, ...]:
        """Return unique trainer invocations after prefix checkpoints exist."""

        self.validate()
        slices: list[MvpTrainingSlice] = []
        resume_conditions = ("A1", "A2", "A3") if include_weights_only else ("A1", "A2")
        for seed in self.training_seeds:
            continuous = MvpTrainingSlice(
                "A0", seed, None, self.total_steps, self.total_steps,
                0, self.total_steps, 0,
            )
            continuous.validate()
            slices.append(continuous)
            for step in self.interruption_steps:
                point_slices = self.slices_for_position(
                    step,
                    include_weights_only=include_weights_only,
                )
                slices.extend(
                    item for item in point_slices
                    if item.condition_id in resume_conditions
                    and item.training_seed == seed
                )
        return tuple(slices)
