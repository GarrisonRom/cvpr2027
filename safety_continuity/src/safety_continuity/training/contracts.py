"""Training boundaries that make interruption conditions comparable."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Protocol

from ..checkpointing.contracts import CheckpointManifest, RestorePlan
from ..common.errors import ContractError
from ..common.identity import validate_run_id
from ..data.schemas import UpdateSampleRef
from ..models.contracts import ModelAdapter


@dataclass(frozen=True)
class BatchManifestRef:
    run_id: str
    condition_id: str
    training_seed: int
    manifest_id: str
    dataset_revision: str
    root_env: str
    relative_path: str
    sha256: str
    sample_count: int
    expected_optimizer_steps: int

    def validate(self) -> None:
        validate_run_id(self.run_id)
        if not isinstance(self.condition_id, str) or not self.condition_id.strip():
            raise ContractError("batch manifest reference condition_id is required")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ContractError("batch manifest reference training_seed must be non-negative")
        if not isinstance(self.manifest_id, str) or not self.manifest_id.strip():
            raise ContractError("batch manifest id is required")
        if not isinstance(self.dataset_revision, str) or not self.dataset_revision.strip():
            raise ContractError("batch manifest dataset revision is required")
        if self.root_env != "SC_SPLIT_ROOT":
            raise ContractError("batch manifests must be rooted at SC_SPLIT_ROOT")
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ContractError("batch manifest relative path is required")
        relative = PurePosixPath(self.relative_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or "\\" in self.relative_path
            or ":" in self.relative_path
        ):
            raise ContractError("batch manifest path must be a safe relative POSIX path")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ContractError("batch manifest needs a lowercase SHA-256 digest")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ContractError("batch manifest sample_count must be a positive integer")
        if type(self.expected_optimizer_steps) is not int or self.expected_optimizer_steps <= 0:
            raise ContractError("expected_optimizer_steps must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "batch_manifest_ref.v1",
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "training_seed": self.training_seed,
            "manifest_id": self.manifest_id,
            "dataset_revision": self.dataset_revision,
            "root_env": self.root_env,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "sample_count": self.sample_count,
            "expected_optimizer_steps": self.expected_optimizer_steps,
        }


def batch_manifest_ref_from_mapping(payload: Mapping[str, Any]) -> BatchManifestRef:
    if not isinstance(payload, Mapping) or payload.get("schema") != "batch_manifest_ref.v1":
        raise ContractError("expected batch manifest reference schema batch_manifest_ref.v1")
    fields = dict(payload)
    fields.pop("schema", None)
    try:
        reference = BatchManifestRef(**fields)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid batch manifest reference: {exc}") from exc
    reference.validate()
    return reference


@dataclass(frozen=True)
class BatchManifest:
    """Immutable ordered sample schedule shared by all interruption branches."""

    manifest_id: str
    dataset_revision: str
    samples: tuple[UpdateSampleRef, ...]

    def validate(self) -> None:
        if (
            not isinstance(self.manifest_id, str)
            or not self.manifest_id.strip()
            or not isinstance(self.dataset_revision, str)
            or not self.dataset_revision.strip()
        ):
            raise ContractError("batch manifest needs an id, dataset revision and samples")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ContractError("batch manifest samples must be a non-empty immutable tuple")
        if any(not isinstance(sample, UpdateSampleRef) for sample in self.samples):
            raise ContractError("batch manifest samples must be UpdateSampleRef records")
        ordinals = [sample.ordinal for sample in self.samples]
        if ordinals != list(range(len(ordinals))):
            raise ContractError("batch manifest ordinals must be contiguous from zero")


@dataclass(frozen=True)
class TrainingRunSpec:
    run_id: str
    model_id: str
    condition_id: str
    seed: int
    total_optimizer_steps: int
    interruption_step: int | None
    batch_manifest_id: str

    def validate(self) -> None:
        validate_run_id(self.run_id)
        for name in ("model_id", "condition_id", "batch_manifest_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"training run {name} is required")
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError("training seed must be a non-negative integer")
        if type(self.total_optimizer_steps) is not int or self.total_optimizer_steps <= 0:
            raise ContractError("total_optimizer_steps must be positive")
        if self.interruption_step is not None:
            if type(self.interruption_step) is not int or not (
                0 < self.interruption_step < self.total_optimizer_steps
            ):
                raise ContractError("interruption_step must be inside the training horizon")


class TrainerAdapter(Protocol):
    """Native training loops implement this protocol behind the OpenPI adapter."""

    def train_until(
        self,
        spec: TrainingRunSpec,
        batch_manifest: BatchManifest,
        *,
        stop_step: int,
    ) -> CheckpointManifest: ...

    def resume(
        self,
        spec: TrainingRunSpec,
        batch_manifest: BatchManifest,
        checkpoint: CheckpointManifest,
        restore_plan: RestorePlan,
        *,
        stop_step: int,
    ) -> CheckpointManifest: ...


def assert_same_schedule(branches: Iterable[BatchManifest]) -> None:
    """Ensure branches consume the same revision and complete ordered sample refs."""
    manifests = list(branches)
    if not manifests:
        raise ContractError("at least one batch manifest is required")
    for manifest in manifests:
        manifest.validate()
    reference = (manifests[0].manifest_id, manifests[0].dataset_revision, manifests[0].samples)
    for manifest in manifests[1:]:
        current = (manifest.manifest_id, manifest.dataset_revision, manifest.samples)
        if current != reference:
            raise ContractError(
                "interruption branches consume different manifests, dataset revisions, "
                "or ordered sample references"
            )
