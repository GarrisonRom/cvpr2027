"""Evaluation runner boundary shared by capability and safety benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from ..common.errors import ContractError
from ..data.schemas import EpisodeRecord, StepTraceRecord
from ..models.contracts import ModelAdapter
from ..safety.contracts import SafetyMonitor


@dataclass(frozen=True)
class EvaluationPlan:
    benchmark_id: str
    task_ids: tuple[str, ...]
    initial_state_seeds: tuple[int, ...]
    episodes_per_task: int
    instruction_split: str
    episode_schema: str
    step_trace_schema: str

    def validate(self) -> None:
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id.strip():
            raise ContractError("evaluation plan needs a benchmark id")
        if (
            not isinstance(self.task_ids, tuple)
            or not self.task_ids
            or any(not isinstance(task_id, str) or not task_id.strip() for task_id in self.task_ids)
            or len(self.task_ids) != len(set(self.task_ids))
        ):
            raise ContractError("evaluation plan needs unique non-empty task ids")
        if (
            not isinstance(self.initial_state_seeds, tuple)
            or not self.initial_state_seeds
            or any(type(seed) is not int or seed < 0 for seed in self.initial_state_seeds)
            or len(self.initial_state_seeds) != len(set(self.initial_state_seeds))
        ):
            raise ContractError("initial state seeds must be unique non-negative integers")
        if not isinstance(self.instruction_split, str) or not self.instruction_split.strip():
            raise ContractError("evaluation plan needs an instruction split")
        if type(self.episodes_per_task) is not int or self.episodes_per_task <= 0:
            raise ContractError("episodes_per_task must be a positive integer")
        if self.episode_schema != "episode.v1" or self.step_trace_schema != "step_trace.v1":
            raise ContractError("evaluation plan uses an unsupported record schema")


class EpisodeRunner(Protocol):
    """Simulator-specific runner; it must return stable records only."""

    def run(
        self,
        plan: EvaluationPlan,
        policy: ModelAdapter,
        safety_monitor: SafetyMonitor,
    ) -> Iterable[tuple[EpisodeRecord, tuple[StepTraceRecord, ...]]]: ...
