"""Runtime root resolution shared by local and cloud runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import ConfigurationError


@dataclass(frozen=True)
class RuntimeRoots:
    """Payload roots. The checkout is never used as an implicit payload mount."""

    workspace_root: Path
    data_root: Path
    split_root: Path
    model_root: Path
    checkpoint_root: Path
    run_root: Path
    cache_root: Path

    @classmethod
    def from_env(
        cls,
        workspace_root: str | Path,
        environ: Mapping[str, str] | None = None,
    ) -> "RuntimeRoots":
        env = os.environ if environ is None else environ
        workspace = Path(workspace_root).expanduser().resolve()

        def root(name: str, default: Path) -> Path:
            value = env.get(name)
            if not value:
                return default.resolve()
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = workspace / candidate
            return candidate.resolve()

        data = root("SC_DATA_ROOT", workspace / "data")
        return cls(
            workspace_root=workspace,
            data_root=data,
            split_root=root("SC_SPLIT_ROOT", data / "splits"),
            model_root=root("SC_MODEL_ROOT", workspace / "models"),
            checkpoint_root=root("SC_CHECKPOINT_ROOT", workspace / "checkpoints"),
            run_root=root("SC_RUN_ROOT", workspace / "runs"),
            cache_root=root("SC_CACHE_ROOT", workspace / "cache"),
        )

    def by_env_name(self, env_name: str) -> Path:
        roots = {
            "SC_DATA_ROOT": self.data_root,
            "SC_SPLIT_ROOT": self.split_root,
            "SC_MODEL_ROOT": self.model_root,
            "SC_CHECKPOINT_ROOT": self.checkpoint_root,
            "SC_RUN_ROOT": self.run_root,
            "SC_CACHE_ROOT": self.cache_root,
        }
        try:
            return roots[env_name]
        except KeyError as exc:
            raise ConfigurationError(f"unknown runtime root: {env_name}") from exc

    def resolve_payload(self, env_name: str, relative_path: str) -> Path:
        """Resolve a config path and reject absolute paths or root escapes."""
        candidate_text = str(relative_path)
        candidate = Path(candidate_text)
        if candidate.is_absolute():
            raise ConfigurationError(
                f"payload path must be relative to {env_name}, got {candidate_text!r}"
            )
        root = self.by_env_name(env_name)
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ConfigurationError(
                f"payload path escapes {env_name}: {candidate_text!r}"
            ) from exc
        return resolved

    def resolve_repo(self, relative_path: str) -> Path:
        """Resolve a tracked repository path without accepting an absolute path."""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ConfigurationError(f"repository path must be relative: {relative_path!r}")
        resolved = (self.workspace_root / candidate).resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ConfigurationError(
                f"repository path escapes workspace: {relative_path!r}"
            ) from exc
        return resolved
