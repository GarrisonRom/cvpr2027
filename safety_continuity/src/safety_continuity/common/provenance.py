"""Machine-readable provenance record required for every completed run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from .errors import ContractError
from .identity import validate_run_id


def _parse_utc_timestamp(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"provenance field {field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(
            f"provenance field {field_name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"provenance field {field_name} must include a UTC offset")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"provenance field {field_name} must be expressed in UTC")
    return parsed


@dataclass(frozen=True)
class Provenance:
    run_id: str
    git_commit: str
    git_dirty: bool
    config_hash: str
    model_id: str
    model_revision: str
    processor_revision: str
    dataset_manifest: str
    split_id: str
    model_seed: int
    evaluation_seed: int
    gpu_name: str
    visible_memory_bytes: int
    driver: str
    python_version: str
    framework_versions: Mapping[str, str]
    simulator_version: str
    started_at_utc: str
    ended_at_utc: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        validate_run_id(self.run_id)
        for name in (
            "run_id",
            "git_commit",
            "config_hash",
            "model_id",
            "model_revision",
            "processor_revision",
            "dataset_manifest",
            "split_id",
            "gpu_name",
            "driver",
            "python_version",
            "simulator_version",
            "started_at_utc",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"provenance field {name} is required")
        if len(self.config_hash) != 64:
            raise ContractError("provenance config_hash must be SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", self.config_hash):
            raise ContractError("provenance config_hash must be lowercase hexadecimal")
        if type(self.git_dirty) is not bool:
            raise ContractError("provenance git_dirty must be a boolean")
        if type(self.model_seed) is not int or type(self.evaluation_seed) is not int:
            raise ContractError("provenance seeds must be integers")
        if self.model_seed < 0 or self.evaluation_seed < 0:
            raise ContractError("provenance seeds must be non-negative")
        if type(self.visible_memory_bytes) is not int:
            raise ContractError("visible_memory_bytes must be an integer")
        if self.visible_memory_bytes < 0:
            raise ContractError("visible_memory_bytes must be non-negative")
        if self.ended_at_utc is not None and (
            not isinstance(self.ended_at_utc, str) or not self.ended_at_utc.strip()
        ):
            raise ContractError("ended_at_utc must be non-empty when present")
        started = _parse_utc_timestamp(self.started_at_utc, "started_at_utc")
        if self.ended_at_utc is not None:
            ended = _parse_utc_timestamp(self.ended_at_utc, "ended_at_utc")
            if ended < started:
                raise ContractError("ended_at_utc precedes started_at_utc")
        if not isinstance(self.framework_versions, Mapping):
            raise ContractError("framework_versions must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["schema"] = "provenance.v1"
        return payload


def provenance_from_mapping(payload: Mapping[str, Any]) -> Provenance:
    if not isinstance(payload, Mapping) or payload.get("schema") != "provenance.v1":
        raise ContractError("expected provenance schema provenance.v1")
    fields = dict(payload)
    fields.pop("schema", None)
    try:
        provenance = Provenance(**fields)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid provenance fields: {exc}") from exc
    provenance.validate()
    return provenance
