"""Safety monitor output kept separate from capability success."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from ..common.errors import ContractError

SAFETY_OUTCOMES = {"safe", "unsafe", "refusal", "false_refusal", "unknown"}


@dataclass(frozen=True)
class SafetyMonitorResult:
    outcome: str
    violation_type: str | None
    safety_cost: float | None
    margins: Mapping[str, float]
    monitor_version: str

    def __post_init__(self) -> None:
        if isinstance(self.margins, Mapping):
            object.__setattr__(self, "margins", MappingProxyType(dict(self.margins)))
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.outcome, str) or self.outcome not in SAFETY_OUTCOMES:
            raise ContractError(f"unknown safety outcome: {self.outcome}")
        if self.violation_type is not None and (
            not isinstance(self.violation_type, str) or not self.violation_type.strip()
        ):
            raise ContractError("violation_type must be non-empty text when present")
        if self.safety_cost is not None:
            if (
                isinstance(self.safety_cost, bool)
                or not isinstance(self.safety_cost, (int, float))
                or (
                    isinstance(self.safety_cost, float)
                    and not math.isfinite(self.safety_cost)
                )
            ):
                raise ContractError("safety cost must be a finite number")
            if self.safety_cost < 0:
                raise ContractError("safety cost must be non-negative")
        if not isinstance(self.margins, Mapping):
            raise ContractError("safety monitor margins must be a mapping")
        for name, value in self.margins.items():
            if not isinstance(name, str) or not name.strip():
                raise ContractError("safety monitor margin names must be non-empty text")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                raise ContractError(f"safety monitor margin {name!r} must be finite")
        if not isinstance(self.monitor_version, str) or not self.monitor_version.strip():
            raise ContractError("safety monitor version is required")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": "safety_monitor.v1",
            "outcome": self.outcome,
            "violation_type": self.violation_type,
            "safety_cost": self.safety_cost,
            "margins": dict(self.margins),
            "monitor_version": self.monitor_version,
        }


class SafetyMonitor(Protocol):
    version: str

    def observe(self, *, simulator_state: object, action: object) -> SafetyMonitorResult: ...
