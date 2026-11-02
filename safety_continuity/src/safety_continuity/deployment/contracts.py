"""Hot-update records that make switch timing and runtime state auditable."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from ..common.errors import ContractError
from ..common.identity import validate_run_id
from ..models.contracts import ActionChunk, ModelAdapter


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_utc_timestamp(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{field_name} must include a UTC offset")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{field_name} must be expressed in UTC")
    return parsed


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class PolicyVersion:
    version_id: str
    model_id: str
    artifact_sha256: str
    adapter_sha256: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.version_id, str) or not self.version_id.strip():
            raise ContractError("policy version_id is required")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ContractError("policy model_id is required")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if self.adapter_sha256 is not None:
            _validate_sha256(self.adapter_sha256, "adapter_sha256")


@dataclass(frozen=True)
class HotUpdateRequest:
    run_id: str
    episode_id: str
    old_version_id: str
    new_version_id: str
    boundary: str
    action_queue_policy: str
    runtime_state_policy: str
    requested_at_utc: str
    allow_rollback: bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_run_id(self.run_id)
        for name in (
            "episode_id",
            "old_version_id",
            "new_version_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} is required")
        if not isinstance(self.boundary, str) or self.boundary not in {
            "episode_start", "action_chunk_boundary"
        }:
            raise ContractError(f"unsupported switch boundary: {self.boundary}")
        if not isinstance(self.action_queue_policy, str) or self.action_queue_policy not in {
            "finish_current_chunk", "flush"
        }:
            raise ContractError(f"unsupported action queue policy: {self.action_queue_policy}")
        if not isinstance(self.runtime_state_policy, str) or self.runtime_state_policy not in {
            "preserve", "reset"
        }:
            raise ContractError(f"unsupported runtime state policy: {self.runtime_state_policy}")
        if self.old_version_id == self.new_version_id:
            raise ContractError("a hot update must identify old and new versions")
        _validate_utc_timestamp(self.requested_at_utc, "requested_at_utc")
        if type(self.allow_rollback) is not bool:
            raise ContractError("allow_rollback must be a boolean")


@dataclass(frozen=True)
class HotUpdateEvent:
    run_id: str
    episode_id: str
    old_version_id: str
    new_version_id: str
    old_artifact_sha256: str
    new_artifact_sha256: str
    old_adapter_sha256: str | None
    new_adapter_sha256: str | None
    requested_at_utc: str
    actual_switch_at_utc: str | None
    boundary: str
    action_queue_policy: str
    queued_actions_before: int
    queued_actions_after: int
    no_op_control: bool
    success: bool
    rollback_reason: str | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_run_id(self.run_id)
        for name in ("episode_id", "old_version_id", "new_version_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} is required")
        if self.old_version_id == self.new_version_id:
            raise ContractError("hot update event must identify distinct policy versions")
        _validate_sha256(self.old_artifact_sha256, "old_artifact_sha256")
        _validate_sha256(self.new_artifact_sha256, "new_artifact_sha256")
        for name in ("old_adapter_sha256", "new_adapter_sha256"):
            value = getattr(self, name)
            if value is not None:
                _validate_sha256(value, name)
        requested = _validate_utc_timestamp(self.requested_at_utc, "requested_at_utc")
        if self.actual_switch_at_utc is not None:
            switched = _validate_utc_timestamp(
                self.actual_switch_at_utc,
                "actual_switch_at_utc",
            )
            if switched < requested:
                raise ContractError("actual switch time precedes request time")
        if not isinstance(self.boundary, str) or self.boundary not in {
            "episode_start", "action_chunk_boundary"
        }:
            raise ContractError(f"unsupported switch boundary: {self.boundary}")
        if not isinstance(self.action_queue_policy, str) or self.action_queue_policy not in {
            "finish_current_chunk", "flush"
        }:
            raise ContractError(f"unsupported action queue policy: {self.action_queue_policy}")
        for name in ("queued_actions_before", "queued_actions_after"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")
        if type(self.no_op_control) is not bool or type(self.success) is not bool:
            raise ContractError("no_op_control and success must be booleans")
        if self.no_op_control and (
            self.old_artifact_sha256 != self.new_artifact_sha256
            or self.old_adapter_sha256 != self.new_adapter_sha256
        ):
            raise ContractError("no-op control requires identical model and adapter hashes")
        if self.success and self.actual_switch_at_utc is None:
            raise ContractError("successful hot update requires an actual switch timestamp")
        if self.success and (self.rollback_reason is not None or self.failure_reason is not None):
            raise ContractError("successful hot update cannot have failure or rollback reasons")
        if not self.success and (
            not isinstance(self.failure_reason, str) or not self.failure_reason.strip()
        ):
            raise ContractError("failed hot update requires a failure_reason")
        if self.rollback_reason is not None and (
            not isinstance(self.rollback_reason, str) or not self.rollback_reason.strip()
        ):
            raise ContractError("rollback_reason must be non-empty text when present")
        if self.rollback_reason is not None and self.actual_switch_at_utc is None:
            raise ContractError("rollback_reason requires an actual switch timestamp")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": "hot_update_event.v1",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "old_version_id": self.old_version_id,
            "new_version_id": self.new_version_id,
            "old_artifact_sha256": self.old_artifact_sha256,
            "new_artifact_sha256": self.new_artifact_sha256,
            "old_adapter_sha256": self.old_adapter_sha256,
            "new_adapter_sha256": self.new_adapter_sha256,
            "requested_at_utc": self.requested_at_utc,
            "actual_switch_at_utc": self.actual_switch_at_utc,
            "boundary": self.boundary,
            "action_queue_policy": self.action_queue_policy,
            "queued_actions_before": self.queued_actions_before,
            "queued_actions_after": self.queued_actions_after,
            "no_op_control": self.no_op_control,
            "success": self.success,
            "rollback_reason": self.rollback_reason,
            "failure_reason": self.failure_reason,
        }


class PolicyServer(Protocol):
    """Serving implementations must expose an atomic chunk-level switch."""

    def publish(self, version: PolicyVersion, adapter: ModelAdapter) -> None: ...

    def request_update(self, request: HotUpdateRequest) -> HotUpdateEvent: ...

    def predict(self, observation: object) -> ActionChunk: ...
