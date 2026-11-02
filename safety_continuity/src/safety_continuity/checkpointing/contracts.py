"""Explicit checkpoint state inventory and interruption restore contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..common.config import sha256_file
from ..common.errors import ContractError
from ..common.identity import validate_run_id

STATE_KEYS = (
    "model",
    "optimizer",
    "optimizer_moments",
    "optimizer_step",
    "scheduler_and_global_step",
    "rng",
    "batch_cursor",
    "normalization",
    "gradient_accumulation",
    "amp_scaler",
    "ema",
    "distributed_layout",
)


@dataclass(frozen=True)
class CheckpointFile:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ContractError("checkpoint file relative_path must be non-empty text")
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
            or ":" in self.relative_path
            or not path.parts
        ):
            raise ContractError("checkpoint file path must be a safe relative POSIX path")
        object.__setattr__(self, "relative_path", path.as_posix())
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ContractError("checkpoint file size must be a non-negative integer")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ContractError("checkpoint file needs a lowercase SHA-256 digest")


@dataclass(frozen=True)
class StateEntry:
    key: str
    included: bool
    restore_compatible: bool
    files: tuple[CheckpointFile, ...] = ()
    reset_reason: str | None = None
    logical_locator: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or self.key not in STATE_KEYS:
            raise ContractError(f"unknown checkpoint state key: {self.key!r}")
        if type(self.included) is not bool or type(self.restore_compatible) is not bool:
            raise ContractError(f"state flags must be booleans: {self.key}")
        if not isinstance(self.files, tuple) or any(
            not isinstance(file, CheckpointFile) for file in self.files
        ):
            raise ContractError(f"state files must be a tuple of CheckpointFile: {self.key}")
        file_paths = [file.relative_path for file in self.files]
        if len(file_paths) != len(set(file_paths)):
            raise ContractError(f"state {self.key} lists duplicate checkpoint files")
        if self.included and not self.files:
            raise ContractError(f"included state {self.key} needs at least one file")
        if self.included and (
            not isinstance(self.logical_locator, str) or not self.logical_locator.strip()
        ):
            raise ContractError(f"included state {self.key} needs a logical_locator")
        if not self.included and self.files:
            raise ContractError(f"excluded state {self.key} cannot list checkpoint files")
        if not self.included and (
            not isinstance(self.reset_reason, str) or not self.reset_reason.strip()
        ):
            raise ContractError(f"absent state {self.key} needs an explicit reset_reason")


@dataclass(frozen=True)
class CheckpointManifest:
    run_id: str
    global_step: int
    checkpoint_kind: str
    entries: tuple[StateEntry, ...]
    save_completed: bool
    manifest_sha256: str | None = None
    schema_version: int = 1

    def validate(self) -> None:
        if type(self.schema_version) is not int:
            raise ContractError("schema_version must be an integer")
        if self.schema_version != 1:
            raise ContractError("unsupported checkpoint manifest schema")
        if type(self.global_step) is not int:
            raise ContractError("global_step must be an integer")
        if not isinstance(self.run_id, str):
            raise ContractError("checkpoint run_id must be text")
        validate_run_id(self.run_id)
        if type(self.save_completed) is not bool:
            raise ContractError("save_completed must be a boolean")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, StateEntry) for entry in self.entries
        ):
            raise ContractError("checkpoint entries must be an immutable StateEntry tuple")
        if self.manifest_sha256 is not None and (
            not isinstance(self.manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256)
        ):
            raise ContractError("manifest_sha256 must be a SHA-256 digest")
        if self.global_step < 0:
            raise ContractError("global_step must be non-negative")
        if not isinstance(self.checkpoint_kind, str) or self.checkpoint_kind not in {
            "full_training_state", "weights_only", "inference_export"
        }:
            raise ContractError(f"unknown checkpoint kind: {self.checkpoint_kind}")
        keys = [entry.key for entry in self.entries]
        if set(keys) != set(STATE_KEYS) or len(keys) != len(STATE_KEYS):
            raise ContractError("checkpoint manifest must explicitly list every state key")
        file_paths = [
            file
            for entry in self.entries
            for file in entry.files
        ]
        files_by_path: dict[str, CheckpointFile] = {}
        for file in file_paths:
            previous = files_by_path.setdefault(file.relative_path, file)
            if previous != file:
                raise ContractError(
                    f"checkpoint path has inconsistent size/hash across state entries: "
                    f"{file.relative_path}"
                )
        if not self.save_completed:
            raise ContractError("an incomplete checkpoint cannot be used for resume")
        inventory = {entry.key: entry for entry in self.entries}
        if self.checkpoint_kind == "full_training_state" and not all(
            entry.included and entry.restore_compatible for entry in inventory.values()
        ):
            raise ContractError("full_training_state checkpoint has missing or incompatible state")
        if (
            self.checkpoint_kind in {"weights_only", "inference_export"}
            and not inventory["model"].included
        ):
            raise ContractError(f"{self.checkpoint_kind} checkpoint must include model state")

    def by_key(self) -> dict[str, StateEntry]:
        self.validate()
        return {entry.key: entry for entry in self.entries}


@dataclass(frozen=True)
class RestorePlan:
    condition_id: str
    preserve: tuple[str, ...]
    reset: tuple[str, ...]
    required: tuple[str, ...]

    def validate(self, manifest: CheckpointManifest) -> None:
        if not self.condition_id:
            raise ContractError("restore plan needs a condition id")
        for name, values in (
            ("preserve", self.preserve),
            ("reset", self.reset),
            ("required", self.required),
        ):
            if any(not isinstance(value, str) for value in values):
                raise ContractError(f"restore plan {name} entries must be strings")
            if len(values) != len(set(values)):
                raise ContractError(f"restore plan {name} entries must be unique")
        inventory = manifest.by_key()
        preserve = set(self.preserve)
        reset = set(self.reset)
        required = set(self.required)
        unknown = (preserve | reset | required) - set(STATE_KEYS)
        if unknown:
            raise ContractError(f"restore plan contains unknown states: {sorted(unknown)}")
        if preserve & reset:
            raise ContractError("restore plan cannot both preserve and reset a state")
        if preserve | reset != set(STATE_KEYS):
            missing = sorted(set(STATE_KEYS) - (preserve | reset))
            raise ContractError(f"restore plan leaves states unspecified: {missing}")
        for key in required:
            if not inventory[key].included:
                raise ContractError(f"required state is absent from checkpoint: {key}")
        for key in preserve:
            if not inventory[key].included or not inventory[key].restore_compatible:
                raise ContractError(f"preserved state is not restore-compatible: {key}")


def manifest_from_mapping(payload: Mapping[str, Any]) -> CheckpointManifest:
    if not isinstance(payload, Mapping):
        raise ContractError("checkpoint manifest must be a JSON object")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or any(
        not isinstance(entry, Mapping) for entry in raw_entries
    ):
        raise ContractError("checkpoint manifest entries must be a list of objects")
    if type(payload.get("save_completed")) is not bool:
        raise ContractError("checkpoint manifest save_completed must be a boolean")
    try:
        entries_list: list[StateEntry] = []
        for raw_entry in raw_entries:
            entry = dict(raw_entry)
            raw_files = entry.get("files")
            if raw_files is None:
                legacy_path = entry.pop("relative_path", None)
                legacy_size = entry.pop("size_bytes", None)
                legacy_sha256 = entry.pop("sha256", None)
                raw_files = []
                if entry.get("included"):
                    raw_files.append({
                        "relative_path": legacy_path,
                        "size_bytes": legacy_size,
                        "sha256": legacy_sha256,
                    })
            if not isinstance(raw_files, list):
                raise ContractError("checkpoint state files must be a list")
            files = tuple(CheckpointFile(**file) for file in raw_files)
            entries_list.append(StateEntry(
                key=entry["key"],
                included=entry["included"],
                files=files,
                restore_compatible=entry["restore_compatible"],
                reset_reason=entry.get("reset_reason"),
                logical_locator=entry.get("logical_locator"),
            ))
        entries = tuple(entries_list)
        manifest = CheckpointManifest(
            run_id=payload["run_id"],
            global_step=payload["global_step"],
            checkpoint_kind=payload["checkpoint_kind"],
            entries=entries,
            save_completed=payload["save_completed"],
            manifest_sha256=payload.get("manifest_sha256"),
            schema_version=payload.get("schema_version", 1),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid checkpoint manifest fields: {exc}") from exc
    manifest.validate()
    return manifest


def verify_checkpoint_files(
    manifest: CheckpointManifest,
    checkpoint_root: str | Path,
) -> list[dict[str, Any]]:
    """Verify each included state file against the manifest's size and SHA-256."""
    manifest.validate()
    root = Path(checkpoint_root).expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"checkpoint root is not a directory: {root}")

    verified: list[dict[str, Any]] = []
    for entry in manifest.entries:
        if not entry.included:
            continue
        for file in entry.files:
            relative = PurePosixPath(file.relative_path)
            candidate = root.joinpath(*relative.parts)
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise ContractError(
                    f"checkpoint state file is missing or escapes checkpoint root: "
                    f"{entry.key} ({file.relative_path})"
                ) from exc
            if not resolved.is_file():
                raise ContractError(
                    f"checkpoint state path is not a regular file: "
                    f"{entry.key} ({file.relative_path})"
                )
            actual_size = resolved.stat().st_size
            if actual_size != file.size_bytes:
                raise ContractError(
                    f"checkpoint state file size mismatch for {entry.key}/{file.relative_path}: "
                    f"expected {file.size_bytes}, got {actual_size}"
                )
            actual_sha256 = sha256_file(resolved)
            if actual_sha256 != file.sha256:
                raise ContractError(
                    f"checkpoint state file SHA-256 mismatch for "
                    f"{entry.key}/{file.relative_path}"
                )
            verified.append({
                "state": entry.key,
                "relative_path": file.relative_path,
                "size_bytes": actual_size,
                "sha256": actual_sha256,
            })
    return verified


def restore_plan_from_condition(condition: Mapping[str, Any]) -> RestorePlan:
    plan = condition.get("restore_plan")
    if not isinstance(plan, Mapping):
        raise ContractError(f"condition {condition.get('id')!r} has no restore_plan")
    condition_id = condition.get("id")
    if not isinstance(condition_id, str) or not condition_id:
        raise ContractError("restore condition needs a non-empty text id")
    values_by_name = {
        key: plan.get(key) for key in ("preserve", "reset", "required")
    }
    for name, values in values_by_name.items():
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
        ):
            raise ContractError(f"restore plan {name} must be a unique string list")
    restore_plan = RestorePlan(
        condition_id=condition_id,
        preserve=tuple(values_by_name["preserve"]),
        reset=tuple(values_by_name["reset"]),
        required=tuple(values_by_name["required"]),
    )
    if not restore_plan.condition_id:
        raise ContractError("restore condition needs an id")
    return restore_plan
