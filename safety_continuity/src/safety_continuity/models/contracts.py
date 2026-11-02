"""Model adapter contract independent of OpenPI, PyTorch or simulator code."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from ..common.errors import ContractError


@dataclass(frozen=True)
class ObservationSpec:
    schema_id: str
    required_modalities: tuple[str, ...]
    camera_mapping: Mapping[str, str]
    language_key: str | None
    proprioception_key: str | None

    def __post_init__(self) -> None:
        if isinstance(self.camera_mapping, Mapping):
            object.__setattr__(self, "camera_mapping", MappingProxyType(dict(self.camera_mapping)))
        self.validate_resolved()

    def validate_resolved(self) -> None:
        if not isinstance(self.schema_id, str) or not self.schema_id.strip():
            raise ContractError("observation schema id is required")
        if (
            not isinstance(self.required_modalities, tuple)
            or not self.required_modalities
            or any(not isinstance(modality, str) or not modality.strip()
                   for modality in self.required_modalities)
        ):
            raise ContractError("observation schema and modalities are required")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ContractError("observation modalities must be unique")
        if not isinstance(self.camera_mapping, Mapping):
            raise ContractError("camera_mapping must be a mapping")
        if any(
            not isinstance(name, str) or not name.strip()
            or not isinstance(key, str) or not key.strip()
            for name, key in self.camera_mapping.items()
        ):
            raise ContractError("camera mappings must map non-empty names to non-empty keys")
        if "rgb" in self.required_modalities and not self.camera_mapping:
            raise ContractError("RGB observation requires an explicit camera mapping")
        if "language" in self.required_modalities and not self.language_key:
            raise ContractError("language observation requires an explicit field key")
        if "proprioception" in self.required_modalities and not self.proprioception_key:
            raise ContractError("proprioception requires an explicit field key")


@dataclass(frozen=True)
class ActionSpec:
    schema_id: str
    dimension: int | None
    chunk_length: int | None
    dtype: str
    normalization_id: str

    def validate_resolved(self) -> None:
        if not isinstance(self.schema_id, str) or not self.schema_id.strip():
            raise ContractError("action schema id is required")
        if self.dimension is None or type(self.dimension) is not int or self.dimension <= 0:
            raise ContractError("action dimension must be resolved before model loading")
        if self.chunk_length is None or type(self.chunk_length) is not int or self.chunk_length <= 0:
            raise ContractError("action chunk length must be resolved before model loading")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ContractError("action dtype is required")
        if not isinstance(self.normalization_id, str) or not self.normalization_id.strip():
            raise ContractError("action normalization id is required")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    adapter_id: str
    checkpoint_revision: str
    processor_revision: str
    embodiment: str
    observation: ObservationSpec
    action: ActionSpec
    checkpoint_format: str
    supported_modes: tuple[str, ...]

    def validate_resolved(self) -> None:
        for name in ("model_id", "adapter_id", "checkpoint_revision", "processor_revision",
                     "embodiment", "checkpoint_format"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"model {name} is required")
        if not self.model_id or not self.adapter_id or not self.checkpoint_revision:
            raise ContractError("model identity and checkpoint revision are required")
        if (
            not isinstance(self.supported_modes, tuple)
            or not self.supported_modes
            or any(not isinstance(mode, str) or not mode.strip() for mode in self.supported_modes)
            or len(self.supported_modes) != len(set(self.supported_modes))
        ):
            raise ContractError("supported_modes must be a non-empty tuple of unique names")
        self.observation.validate_resolved()
        self.action.validate_resolved()


@dataclass(frozen=True)
class ActionChunk:
    """Canonical post-processed action representation at the deployment boundary."""

    model_version: str
    chunk_id: str
    values: tuple[tuple[float, ...], ...]
    normalization_id: str

    def validate_against(self, spec: ActionSpec) -> None:
        spec.validate_resolved()
        if len(self.values) != spec.chunk_length:
            raise ContractError(
                f"action chunk has {len(self.values)} steps, expected {spec.chunk_length}"
            )
        if any(len(step) != spec.dimension for step in self.values):
            raise ContractError("action dimension does not match model contract")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            for step in self.values
            for value in step
        ):
            raise ContractError("action chunk contains a non-finite value")
        if self.normalization_id != spec.normalization_id:
            raise ContractError("action normalization does not match model contract")


class ModelAdapter(Protocol):
    """Every model implementation must expose the same boundaries."""

    @property
    def spec(self) -> ModelSpec: ...

    def load(self, *, device: str, mode: str) -> None: ...

    def preprocess(self, observation: Mapping[str, Any]) -> Any: ...

    def predict(self, model_input: Any) -> ActionChunk: ...

    def save_inference_export(self, destination: str) -> Mapping[str, Any]: ...


def assert_model_compatible(left: ModelSpec, right: ModelSpec) -> None:
    """Check complete serving-interface compatibility for comparison or switching."""
    left.validate_resolved()
    right.validate_resolved()
    if left.model_id != right.model_id:
        raise ContractError("policies use different model ids")
    if left.adapter_id != right.adapter_id:
        raise ContractError("policies use different model adapters")
    if left.embodiment != right.embodiment:
        raise ContractError("policies use different embodiments")
    if left.processor_revision != right.processor_revision:
        raise ContractError("policies use different processor revisions")
    if left.observation != right.observation:
        raise ContractError("policies use different observation fields or camera mappings")
    if left.action != right.action:
        raise ContractError("policies use different action schemas or normalization")
    if left.checkpoint_format != right.checkpoint_format:
        raise ContractError("policies use different checkpoint formats")
    if set(left.supported_modes) != set(right.supported_modes):
        raise ContractError("policies declare different supported modes")
