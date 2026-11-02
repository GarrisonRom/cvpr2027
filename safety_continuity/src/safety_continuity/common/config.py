"""Safe YAML loading and deterministic configuration hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError
from .roots import RuntimeRoots


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {file_path}")
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {file_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {file_path}")
    return payload


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config_reference(
    reference: str,
    *,
    roots: RuntimeRoots,
    root_env: str | None = None,
) -> Path:
    """Resolve either a tracked config/manifest or a payload path."""
    if root_env:
        return roots.resolve_payload(root_env, reference)
    return roots.resolve_repo(reference)


def require_schema_version(payload: dict[str, Any], expected: int = 1) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        raise ConfigurationError(
            f"expected schema_version={expected}, got {actual!r}"
        )
