"""Command-line entry points for validating and analyzing explicit run sets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..common.config import load_yaml
from ..common.errors import ContractError
from ..common.identity import validate_run_id
from ..common.roots import RuntimeRoots
from ..data.io import read_episode_jsonl
from .episodes import analyze_runs


def _workspace_default() -> Path:
    return Path(__file__).resolve().parents[4]


def validate_episodes_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate canonical episode JSONL files for explicit run ids"
    )
    parser.add_argument("--workspace-root", type=Path, default=_workspace_default())
    parser.add_argument("--run-id", action="append", required=True,
                        help="run id to validate; may be repeated")
    args = parser.parse_args(argv)

    roots = RuntimeRoots.from_env(args.workspace_root)
    report = []
    for run_id in args.run_id:
        validate_run_id(run_id)
        path = roots.run_root / run_id / "episodes" / "episodes.jsonl"
        records = read_episode_jsonl(path, expected_run_id=run_id)
        identities = {(row.model_id, row.condition_id, row.training_seed) for row in records}
        if len(identities) != 1:
            raise ContractError(
                f"run {run_id} mixes model, condition or seed: {sorted(identities)}"
            )
        identity = next(iter(identities))
        group_counts = Counter((row.benchmark_id, row.suite_id) for row in records)
        report.append({
            "run_id": run_id,
            "episode_file": str(path),
            "episode_count": len(records),
            "model_id": identity[0],
            "condition_id": identity[1],
            "training_seed": identity[2],
            "groups": [
                {"benchmark_id": key[0], "suite_id": key[1], "episode_count": value}
                for key, value in sorted(group_counts.items())
            ],
            "status": "valid",
        })
    print(json.dumps({"schema": "episode_validation_report.v1", "runs": report},
                     ensure_ascii=True, indent=2, sort_keys=True))
    return 0


def analyze_mvp_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze paired MVP episodes from explicitly listed run ids"
    )
    parser.add_argument("--workspace-root", type=Path, default=_workspace_default())
    parser.add_argument("--run-id", action="append", required=True,
                        help="completed run id to include; may be repeated")
    parser.add_argument("--analysis-id", default=None,
                        help="immutable output id; defaults to today's primary-analysis id")
    parser.add_argument("--comparison", default=None,
                        help="target:reference condition ids, for example A2:A1")
    parser.add_argument("--analysis-config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="defaults to SC_RUN_ROOT or workspace/runs")
    args = parser.parse_args(argv)

    workspace_root = args.workspace_root.resolve()
    config_path = args.analysis_config or (
        workspace_root / "safety_continuity" / "configs" / "analysis" / "mvp_primary.yaml"
    )
    config = load_yaml(config_path)
    comparison = args.comparison
    if comparison is None:
        configured = config["primary_safety_estimand"]["comparison"]
        parts = configured.split("_minus_")
        if len(parts) != 2:
            raise ContractError(
                "primary_safety_estimand.comparison must use TARGET_minus_REFERENCE"
            )
        comparison = f"{parts[0]}:{parts[1]}"
    utc_date = datetime.now(timezone.utc).date()
    analysis_id = args.analysis_id or f"{utc_date:%Y%m%d}_analysis_mvp_primary_s00"
    output_dir = analyze_runs(
        workspace_root=workspace_root,
        run_ids=args.run_id,
        analysis_id=analysis_id,
        comparison=comparison,
        config_path=config_path,
        output_root=args.output_root,
    )
    print(f"analysis output: {output_dir}")
    return 0
