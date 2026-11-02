"""Checkpoint manifest compatibility checks for resume conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.config import load_yaml
from ..common.errors import ContractError
from .contracts import (
    manifest_from_mapping,
    restore_plan_from_condition,
    verify_checkpoint_files,
)


def validate_restore_plan_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a checkpoint state inventory against one interruption condition"
    )
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="checkpoint payload directory containing the manifest's relative state paths",
    )
    parser.add_argument(
        "--interruption-config",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "configs" / "interruption" / "mvp_matrix.yaml",
    )
    parser.add_argument("--condition", required=True, help="condition id such as A1 or A2")
    args = parser.parse_args(argv)

    try:
        checkpoint_payload = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read checkpoint manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid checkpoint manifest JSON: {exc}") from exc
    if not isinstance(checkpoint_payload, dict):
        raise ContractError("checkpoint manifest root must be a JSON object")
    manifest = manifest_from_mapping(checkpoint_payload)

    config = load_yaml(args.interruption_config)
    condition = next(
        (item for item in config.get("conditions", []) if item.get("id") == args.condition),
        None,
    )
    if condition is None:
        raise ContractError(f"condition not found in interruption config: {args.condition}")
    plan = restore_plan_from_condition(condition)
    plan.validate(manifest)
    verified_files = verify_checkpoint_files(manifest, args.checkpoint_root)
    print(json.dumps({
        "schema": "restore_preflight.v1",
        "status": "compatible",
        "condition_id": plan.condition_id,
        "checkpoint_run_id": manifest.run_id,
        "checkpoint_kind": manifest.checkpoint_kind,
        "global_step": manifest.global_step,
        "verified_state_file_count": len(verified_files),
        "verified_state_files": verified_files,
        "preserve": list(plan.preserve),
        "reset": list(plan.reset),
        "required": list(plan.required),
    }, ensure_ascii=True, indent=2, sort_keys=True))
    return 0
