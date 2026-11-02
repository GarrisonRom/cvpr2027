"""Translate a resolved model YAML into the shared ModelSpec."""

from __future__ import annotations

from typing import Any, Mapping

from ..common.errors import ConfigurationError
from .contracts import ActionSpec, ModelSpec, ObservationSpec


def model_spec_from_config(payload: Mapping[str, Any]) -> ModelSpec:
    try:
        processor = payload["processor"]
        observation = payload["observation"]
        action = payload["action"]
        embodiment = payload["embodiment"]["platform"]
        checkpoint = payload["checkpoint"]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(f"model config is missing a contract section: {exc}") from exc

    spec = ModelSpec(
        model_id=str(payload["model_id"]),
        adapter_id=str(payload["adapter_id"]),
        checkpoint_revision=str(payload.get("checkpoint_revision", processor["revision"])),
        processor_revision=str(processor["revision"]),
        embodiment=str(embodiment),
        observation=ObservationSpec(
            schema_id=str(observation["schema_id"]),
            required_modalities=tuple(str(value) for value in observation["required_modalities"]),
            camera_mapping={str(k): str(v) for k, v in observation["camera_mapping"].items()},
            language_key=observation.get("language_key"),
            proprioception_key=observation.get("proprioception_key"),
        ),
        action=ActionSpec(
            schema_id=str(action["schema_id"]),
            dimension=action.get("dimension"),
            chunk_length=action.get("chunk_length"),
            dtype=str(action["dtype"]),
            normalization_id=str(action["normalization_id"]),
        ),
        checkpoint_format=str(checkpoint["format"]),
        supported_modes=tuple(str(value) for value in checkpoint["mode_contracts"]),
    )
    spec.validate_resolved()
    return spec
