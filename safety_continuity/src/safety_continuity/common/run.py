"""Immutable run context and canonical output paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import ContractError
from .identity import validate_run_id
from .roots import RuntimeRoots


@dataclass(frozen=True)
class RunContext:
    run_id: str
    model_id: str
    condition_id: str
    seed: int
    resolved_config_hash: str
    git_commit: str
    roots: RuntimeRoots

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if not self.model_id or not self.condition_id or not self.git_commit:
            raise ContractError("run context needs model, condition and git identity")
        if not re.fullmatch(r"[0-9a-f]{64}", self.resolved_config_hash):
            raise ContractError("resolved_config_hash must be a lowercase SHA-256 hex digest")
        if self.seed < 0:
            raise ContractError("seed must be non-negative")

    @property
    def run_dir(self) -> Path:
        return self.roots.run_root / self.run_id

    def subdir(self, name: str) -> Path:
        allowed = {
            "metadata",
            "logs",
            "events",
            "episodes",
            "metrics",
            "probes",
            "figures",
        }
        if name not in allowed:
            raise ContractError(f"unknown run output subdirectory: {name}")
        return self.run_dir / name

    def expected_output_paths(self) -> dict[str, Path]:
        names = (
            "metadata",
            "logs",
            "events",
            "episodes",
            "metrics",
            "probes",
            "figures",
        )
        return {name: self.subdir(name) for name in names}
