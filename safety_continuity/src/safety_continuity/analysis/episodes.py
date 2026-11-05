"""Run-level summaries and seed-clustered paired comparisons."""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..common.config import load_yaml, sha256_file, sha256_payload
from ..common.errors import ContractError
from ..common.identity import validate_run_id
from ..common.provenance import provenance_from_mapping
from ..common.roots import RuntimeRoots
from ..data.io import (
    read_batch_consumption_jsonl,
    read_episode_jsonl,
    read_update_sample_jsonl,
)
from ..data.schemas import EpisodeRecord
from ..training.contracts import (
    BatchManifest,
    BatchManifestRef,
    batch_manifest_ref_from_mapping,
)

PAIR_FIELDS = (
    "benchmark_id",
    "suite_id",
    "task_id",
    "scene_id",
    "instruction_id",
    "instruction_type",
    "initial_state_seed",
    "evaluation_seed",
    "interruption_step",
)


def _pair_key(record: EpisodeRecord) -> tuple[object, ...]:
    return tuple(getattr(record, name) for name in PAIR_FIELDS)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap_interval(
    seed_effects: Sequence[float],
    *,
    confidence_level: float,
    resamples: int,
    random_seed: int,
) -> tuple[float | None, float | None]:
    if len(seed_effects) < 2:
        return None, None
    rng = random.Random(random_seed)
    sample_count = len(seed_effects)
    means = []
    for _ in range(resamples):
        sample = [seed_effects[rng.randrange(sample_count)] for _ in range(sample_count)]
        means.append(sum(sample) / sample_count)
    tail = (1 - confidence_level) / 2
    return _percentile(means, tail), _percentile(means, 1 - tail)


def summarize_runs(records: Sequence[EpisodeRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.run_id, record.model_id, record.condition_id, record.benchmark_id,
                 record.training_seed)].append(record)

    summaries: list[dict[str, Any]] = []
    for (
        run_id,
        model_id,
        condition_id,
        benchmark_id,
        training_seed,
    ), rows in sorted(grouped.items()):
        by_suite: dict[str, list[EpisodeRecord]] = defaultdict(list)
        for row in rows:
            by_suite[row.suite_id].append(row)
        for suite_id, suite_rows in sorted(by_suite.items()):
            success_values = [row.success for row in suite_rows if row.success is not None]
            safety_rows = [row for row in suite_rows if row.safe_outcome != "unknown"]
            harmful = [row for row in suite_rows
                       if row.instruction_type == "harmful" and row.safe_outcome != "unknown"]
            benign = [row for row in suite_rows
                      if row.instruction_type == "benign" and row.safe_outcome != "unknown"]
            costs = [row.safety_cost for row in suite_rows if row.safety_cost is not None]
            summary = {
                "run_id": run_id,
                "model_id": model_id,
                "condition_id": condition_id,
                "benchmark_id": benchmark_id,
                "suite_id": suite_id,
                "training_seed": training_seed,
                "episode_count": len(suite_rows),
                "success_n": sum(value is True for value in success_values),
                "success_den": len(success_values),
                "success_rate": _rate(
                    sum(value is True for value in success_values), len(success_values)
                ),
                "unsafe_n": sum(row.safe_outcome == "unsafe" for row in safety_rows),
                "safety_den": len(safety_rows),
                "unsafe_episode_rate": _rate(
                    sum(row.safe_outcome == "unsafe" for row in safety_rows), len(safety_rows)
                ),
                "refusal_n": sum(row.safe_outcome == "refusal" for row in harmful),
                "harmful_instruction_den": len(harmful),
                "refusal_rate": _rate(
                    sum(row.safe_outcome == "refusal" for row in harmful), len(harmful)
                ),
                "false_refusal_n": sum(row.safe_outcome == "false_refusal" for row in benign),
                "benign_instruction_den": len(benign),
                "false_refusal_rate": _rate(
                    sum(row.safe_outcome == "false_refusal" for row in benign), len(benign)
                ),
                "safety_cost_n": len(costs),
                "mean_safety_cost": sum(costs) / len(costs) if costs else None,
            }
            summaries.append(summary)
    return summaries


def parse_comparison(value: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ContractError("comparison must be text in TARGET:REFERENCE form")
    parts = value.split(":")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ContractError(f"comparison must be TARGET:REFERENCE, got {value!r}")
    target, reference = (part.strip() for part in parts)
    if target == reference:
        raise ContractError("target and reference conditions must be different")
    return target, reference


def condition_id_for_label(value: str, condition_labels: Mapping[str, str]) -> str:
    if not isinstance(condition_labels, Mapping) or any(
        not isinstance(key, str) or not isinstance(label, str) or not label.strip()
        for key, label in condition_labels.items()
    ):
        raise ContractError("condition_labels must map condition ids to non-empty display names")
    candidate = value.strip()
    if candidate in condition_labels:
        return candidate
    reverse = {
        label.strip().casefold(): condition_id
        for condition_id, label in condition_labels.items()
    }
    try:
        return reverse[candidate.casefold()]
    except KeyError as exc:
        raise ContractError(f"unknown condition name: {value!r}") from exc


def parse_condition_relation(
    value: str,
    condition_labels: Mapping[str, str],
    *,
    relation: str,
    legacy_relation: str,
) -> tuple[str, str]:
    parts = value.split(relation)
    if len(parts) != 2:
        parts = value.split(legacy_relation)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ContractError(f"condition relation must contain {relation.strip()!r}")
    left = condition_id_for_label(parts[0], condition_labels)
    right = condition_id_for_label(parts[1], condition_labels)
    if left == right:
        raise ContractError("condition relation must name two different conditions")
    return left, right


def paired_comparison(
    records: Sequence[EpisodeRecord],
    *,
    target_condition: str,
    reference_condition: str,
    equivalence_margin: float,
    confidence_safety: float,
    confidence_capability: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(target_condition, str) or not isinstance(reference_condition, str):
        raise ContractError("paired comparison condition ids must be text")
    target_condition = target_condition.strip()
    reference_condition = reference_condition.strip()
    if (
        not target_condition
        or not reference_condition
        or target_condition == reference_condition
    ):
        raise ContractError("paired comparison requires two distinct non-empty conditions")
    groups: dict[
        tuple[str, str, str, int, str], dict[tuple[object, ...], EpisodeRecord]
    ] = defaultdict(dict)
    run_for_condition_seed: dict[tuple[str, str, str, str, int], str] = {}

    for record in records:
        key = (
            record.model_id,
            record.condition_id,
            record.benchmark_id,
            record.suite_id,
            record.training_seed,
        )
        run_id = run_for_condition_seed.setdefault(key, record.run_id)
        if run_id != record.run_id:
            raise ContractError(
                "analysis includes multiple runs for the same model/condition/benchmark/seed: "
                f"{key}"
            )
        group_key = (record.model_id, record.benchmark_id, record.suite_id,
                     record.training_seed, record.condition_id)
        pair_key = _pair_key(record)
        if pair_key in groups[group_key]:
            raise ContractError(f"duplicate paired episode key in group {group_key}: {pair_key}")
        groups[group_key][pair_key] = record

    strata = sorted({(model, benchmark, suite) for model, benchmark, suite, _, _ in groups})
    safety_rows: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []

    for model_id, benchmark_id, suite_id in strata:
        seed_effects_safety: list[float] = []
        seed_effects_capability: list[float] = []
        per_seed_safety: list[dict[str, Any]] = []
        per_seed_capability: list[dict[str, Any]] = []
        seeds = sorted({
            seed for model, benchmark, suite, seed, condition in groups
            if (model, benchmark, suite) == (model_id, benchmark_id, suite_id)
            and condition in {target_condition, reference_condition}
        })

        for training_seed in seeds:
            group_target = groups.get(
                (model_id, benchmark_id, suite_id, training_seed, target_condition), {}
            )
            group_reference = groups.get(
                (model_id, benchmark_id, suite_id, training_seed, reference_condition), {}
            )
            common = sorted(set(group_target) & set(group_reference), key=repr)
            if not common:
                continue
            unknown_outcomes = sum(
                group_target[key].safe_outcome == "unknown"
                or group_reference[key].safe_outcome == "unknown"
                for key in common
            )
            safety_pairs = [key for key in common
                            if group_target[key].safe_outcome != "unknown"
                            and group_reference[key].safe_outcome != "unknown"]
            unsafe_target = sum(group_target[key].safe_outcome == "unsafe" for key in safety_pairs)
            unsafe_reference = sum(
                group_reference[key].safe_outcome == "unsafe" for key in safety_pairs
            )
            target_safety_rate = _rate(unsafe_target, len(safety_pairs))
            reference_safety_rate = _rate(unsafe_reference, len(safety_pairs))
            safety_difference = (
                target_safety_rate - reference_safety_rate
                if target_safety_rate is not None and reference_safety_rate is not None
                else None
            )
            if safety_difference is not None:
                seed_effects_safety.append(safety_difference)
            per_seed_safety.append({
                "row_type": "training_seed",
                "model_id": model_id,
                "benchmark_id": benchmark_id,
                "suite_id": suite_id,
                "target_condition": target_condition,
                "reference_condition": reference_condition,
                "training_seed": training_seed,
                "matched_pair_count": len(common),
                "safety_eligible_pairs": len(safety_pairs),
                "unknown_outcome_pairs": unknown_outcomes,
                "unmatched_target": len(group_target) - len(common),
                "unmatched_reference": len(group_reference) - len(common),
                "unsafe_target_n": unsafe_target,
                "unsafe_reference_n": unsafe_reference,
                "target_unsafe_rate": target_safety_rate,
                "reference_unsafe_rate": reference_safety_rate,
                "risk_difference": safety_difference,
            })

            capability_pairs = [key for key in common
                                if group_target[key].success is not None
                                and group_reference[key].success is not None]
            target_success = sum(group_target[key].success is True for key in capability_pairs)
            reference_success = sum(
                group_reference[key].success is True for key in capability_pairs
            )
            target_success_rate = _rate(target_success, len(capability_pairs))
            reference_success_rate = _rate(reference_success, len(capability_pairs))
            capability_difference = (
                target_success_rate - reference_success_rate
                if target_success_rate is not None and reference_success_rate is not None
                else None
            )
            if capability_difference is not None:
                seed_effects_capability.append(capability_difference)
            per_seed_capability.append({
                "row_type": "training_seed",
                "model_id": model_id,
                "benchmark_id": benchmark_id,
                "suite_id": suite_id,
                "target_condition": target_condition,
                "reference_condition": reference_condition,
                "training_seed": training_seed,
                "matched_pair_count": len(common),
                "capability_eligible_pairs": len(capability_pairs),
                "missing_success_pairs": len(common) - len(capability_pairs),
                "unmatched_target": len(group_target) - len(common),
                "unmatched_reference": len(group_reference) - len(common),
                "success_target_n": target_success,
                "success_reference_n": reference_success,
                "target_success_rate": target_success_rate,
                "reference_success_rate": reference_success_rate,
                "success_rate_difference": capability_difference,
            })

        if not per_seed_safety and not per_seed_capability:
            continue

        safety_low, safety_high = _cluster_bootstrap_interval(
            seed_effects_safety,
            confidence_level=confidence_safety,
            resamples=bootstrap_resamples,
            random_seed=bootstrap_seed,
        )
        safety_mean = (
            sum(seed_effects_safety) / len(seed_effects_safety)
            if seed_effects_safety else None
        )
        safety_rows.extend(per_seed_safety)
        safety_rows.append({
            "row_type": "seed_cluster_summary",
            "model_id": model_id,
            "benchmark_id": benchmark_id,
            "suite_id": suite_id,
            "target_condition": target_condition,
            "reference_condition": reference_condition,
            "training_seed_count": len(seed_effects_safety),
            "mean_seed_level_risk_difference": safety_mean,
            "bootstrap_confidence_level": confidence_safety,
            "bootstrap_ci_low": safety_low,
            "bootstrap_ci_high": safety_high,
            "interval_method": (
                "percentile bootstrap over training-seed clusters; pilot descriptive"
            ),
        })

        cap_low, cap_high = _cluster_bootstrap_interval(
            seed_effects_capability,
            confidence_level=confidence_capability,
            resamples=bootstrap_resamples,
            random_seed=bootstrap_seed + 1,
        )
        cap_mean = (
            sum(seed_effects_capability) / len(seed_effects_capability)
            if seed_effects_capability else None
        )
        interval_inside_margin = (
            cap_low is not None and cap_high is not None
            and cap_low >= -equivalence_margin and cap_high <= equivalence_margin
        )
        capability_rows.extend(per_seed_capability)
        capability_rows.append({
            "row_type": "seed_cluster_summary",
            "model_id": model_id,
            "benchmark_id": benchmark_id,
            "suite_id": suite_id,
            "target_condition": target_condition,
            "reference_condition": reference_condition,
            "training_seed_count": len(seed_effects_capability),
            "mean_seed_level_success_rate_difference": cap_mean,
            "equivalence_margin_absolute": equivalence_margin,
            "bootstrap_confidence_level": confidence_capability,
            "bootstrap_ci_low": cap_low,
            "bootstrap_ci_high": cap_high,
            "interval_within_margin": interval_inside_margin if cap_low is not None else None,
            "equivalence_screen": (
                "interval_within_margin" if interval_inside_margin else
                "interval_not_within_margin" if cap_low is not None else
                "insufficient_training_seeds"
            ),
            "interval_method": (
                "percentile bootstrap over training-seed clusters; "
                "pilot descriptive, not TOST"
            ),
        })

    return safety_rows, capability_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=True, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _git_commit(workspace_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workspace_root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_dirty(workspace_root: Path) -> bool | None:
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"], cwd=workspace_root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    benchmark_id: str,
    suite_id: str,
    target: str,
    reference: str,
) -> Mapping[str, Any] | None:
    return next((
        row for row in rows
        if row.get("row_type") == "seed_cluster_summary"
        and row.get("model_id") == model_id
        and row.get("benchmark_id") == benchmark_id
        and row.get("suite_id") == suite_id
        and row.get("target_condition") == target
        and row.get("reference_condition") == reference
    ), None)


def _assess_resume_control(
    *,
    model_ids: Sequence[str],
    safety_rows: Sequence[Mapping[str, Any]],
    capability_rows: Sequence[Mapping[str, Any]],
    target: str,
    reference: str,
    safety_benchmark_id: str,
    safety_suites: Sequence[str],
    safety_margin: float,
    capability_benchmark_id: str,
    capability_suite_id: str,
    capability_margin: float,
    minimum_training_seeds: int,
    confidence_level: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_statuses: dict[str, str] = {}

    checks = [
        ("unsafe_episode_rate", safety_benchmark_id, suite, safety_margin, safety_rows)
        for suite in safety_suites
    ]
    checks.append((
        "success_rate",
        capability_benchmark_id,
        capability_suite_id,
        capability_margin,
        capability_rows,
    ))

    for model_id in model_ids:
        statuses: list[str] = []
        for metric, benchmark_id, suite_id, margin, source_rows in checks:
            summary = _summary_row(
                source_rows,
                model_id=model_id,
                benchmark_id=benchmark_id,
                suite_id=suite_id,
                target=target,
                reference=reference,
            )
            seed_count = int(summary.get("training_seed_count", 0)) if summary else 0
            ci_low = summary.get("bootstrap_ci_low") if summary else None
            ci_high = summary.get("bootstrap_ci_high") if summary else None
            if seed_count < minimum_training_seeds or ci_low is None or ci_high is None:
                status = "insufficient_data"
            elif ci_low >= -margin and ci_high <= margin:
                status = "pass"
            else:
                status = "fail"
            statuses.append(status)
            rows.append({
                "model_id": model_id,
                "target_condition": target,
                "reference_condition": reference,
                "metric": metric,
                "benchmark_id": benchmark_id,
                "suite_id": suite_id,
                "training_seed_count": seed_count,
                "minimum_training_seeds": minimum_training_seeds,
                "equivalence_margin_absolute": margin,
                "confidence_level": confidence_level,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "status": status,
            })
        model_statuses[model_id] = (
            "pass" if all(status == "pass" for status in statuses)
            else "fail" if any(status == "fail" for status in statuses)
            else "insufficient_data"
        )

    overall_status = (
        "pass" if model_statuses and all(status == "pass" for status in model_statuses.values())
        else "fail" if any(status == "fail" for status in model_statuses.values())
        else "insufficient_data"
    )
    return rows, {"status": overall_status, "by_model": model_statuses}


def _assess_candidate_signal(
    *,
    model_ids: Sequence[str],
    safety_rows: Sequence[Mapping[str, Any]],
    target: str,
    reference: str,
    benchmark_id: str,
    suites: Sequence[str],
    risk_threshold: float,
    minimum_training_seeds: int,
    same_direction_minimum: int,
    require_all_suites: bool,
    applicable: bool,
) -> dict[str, Any]:
    if not applicable:
        return {
            "status": "not_applicable_comparison_override",
            "target_condition": target,
            "reference_condition": reference,
            "suites": [],
        }

    suite_results: list[dict[str, Any]] = []
    for model_id in model_ids:
        for suite_id in suites:
            summary = _summary_row(
                safety_rows,
                model_id=model_id,
                benchmark_id=benchmark_id,
                suite_id=suite_id,
                target=target,
                reference=reference,
            )
            seed_rows = [
                row for row in safety_rows
                if row.get("row_type") == "training_seed"
                and row.get("model_id") == model_id
                and row.get("benchmark_id") == benchmark_id
                and row.get("suite_id") == suite_id
                and row.get("target_condition") == target
                and row.get("reference_condition") == reference
                and row.get("risk_difference") is not None
            ]
            effects = [float(row["risk_difference"]) for row in seed_rows]
            mean_effect = (
                float(summary["mean_seed_level_risk_difference"])
                if summary and summary.get("mean_seed_level_risk_difference") is not None
                else None
            )
            if mean_effect is None or len(effects) < minimum_training_seeds:
                direction_count = 0
                status = "insufficient_data"
            else:
                direction_count = sum(effect > 0 for effect in effects)
                if (
                    mean_effect >= risk_threshold
                    and direction_count >= same_direction_minimum
                ):
                    status = "candidate_signal"
                else:
                    status = "not_met"
            suite_results.append({
                "model_id": model_id,
                "benchmark_id": benchmark_id,
                "suite_id": suite_id,
                "target_condition": target,
                "reference_condition": reference,
                "training_seed_count": len(effects),
                "minimum_training_seeds": minimum_training_seeds,
                "mean_seed_level_risk_difference": mean_effect,
                "risk_difference_threshold_absolute": risk_threshold,
                "expected_direction": "positive_risk_increase",
                "same_direction_seed_count": direction_count,
                "same_direction_seeds_minimum": same_direction_minimum,
                "status": status,
            })

    statuses = [row["status"] for row in suite_results]
    if not statuses or any(status == "insufficient_data" for status in statuses):
        overall = "insufficient_data"
    elif require_all_suites:
        overall = "candidate_signal" if all(
            status == "candidate_signal" for status in statuses
        ) else "not_met"
    else:
        overall = "candidate_signal" if any(
            status == "candidate_signal" for status in statuses
        ) else "not_met"
    return {
        "status": overall,
        "target_condition": target,
        "reference_condition": reference,
        "require_all_suites": require_all_suites,
        "suites": suite_results,
    }


def _read_run_batch_evidence(
    *,
    run_dir: Path,
    roots: RuntimeRoots,
    run_id: str,
    model_id: str,
    condition_id: str,
    training_seed: int,
) -> dict[str, Any]:
    reference_path = run_dir / "metadata" / "batch_manifest.json"
    consumption_path = run_dir / "events" / "batch_consumption.jsonl"
    if not reference_path.is_file() or not consumption_path.is_file():
        raise ContractError("batch manifest reference or batch consumption trace is missing")
    try:
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid batch manifest reference: {exc}") from exc
    if not isinstance(reference_payload, dict):
        raise ContractError("batch manifest reference must be a JSON object")
    reference = batch_manifest_ref_from_mapping(reference_payload)
    if (
        reference.run_id != run_id
        or reference.condition_id != condition_id
        or reference.training_seed != training_seed
    ):
        raise ContractError("batch manifest reference identity does not match the run")
    manifest_path = roots.resolve_payload(reference.root_env, reference.relative_path)
    if not manifest_path.is_file():
        raise ContractError(f"frozen batch manifest is missing: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != reference.sha256:
        raise ContractError("frozen batch manifest SHA-256 does not match run metadata")
    samples = read_update_sample_jsonl(manifest_path)
    if len(samples) != reference.sample_count:
        raise ContractError("frozen batch manifest sample count does not match run metadata")
    manifest = BatchManifest(
        manifest_id=reference.manifest_id,
        dataset_revision=reference.dataset_revision,
        samples=samples,
    )
    manifest.validate()
    consumption = read_batch_consumption_jsonl(
        consumption_path,
        expected_run_id=run_id,
    )
    expected_samples = samples[
        reference.schedule_start_ordinal:
        reference.schedule_start_ordinal + reference.expected_optimizer_steps
    ]
    if len(consumption) != len(expected_samples):
        raise ContractError("consumed sample count does not match this run's batch schedule slice")
    for index, (record, expected_sample) in enumerate(zip(consumption, expected_samples)):
        if (
            record.sequence_index != index
            or record.sample != expected_sample
            or record.sample.ordinal != reference.schedule_start_ordinal + index
        ):
            raise ContractError(
                f"consumed sample at sequence index {index} differs from the frozen schedule"
            )
        if (
            record.condition_id != condition_id
            or record.training_seed != training_seed
        ):
            raise ContractError("batch consumption identity does not match the run")
    if len(consumption) != reference.expected_optimizer_steps:
        raise ContractError("batch consumption trace length differs from the configured update count")
    return {
        "run_id": run_id,
        "model_id": model_id,
        "condition_id": condition_id,
        "training_seed": training_seed,
        "reference": reference,
        "manifest_sha256": manifest_sha256,
        "samples": samples,
        "consumption": consumption,
        "reference_file_sha256": sha256_file(reference_path),
        "consumption_file_sha256": sha256_file(consumption_path),
    }


def _audit_control_batch_sequences(
    *,
    model_ids: Sequence[str],
    training_seeds: Sequence[int],
    evidence_by_run: Mapping[str, Mapping[str, Any] | None],
    evidence_errors: Mapping[str, str],
    identities: Mapping[str, tuple[str, str, int]],
    target: str,
    reference: str,
    minimum_training_seeds: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_statuses: dict[str, str] = {}
    for model_id in model_ids:
        model_rows: list[dict[str, Any]] = []
        for seed in training_seeds:
            target_runs = [
                run_id for run_id, identity in identities.items()
                if identity == (model_id, target, seed)
            ]
            reference_runs = [
                run_id for run_id, identity in identities.items()
                if identity == (model_id, reference, seed)
            ]
            row: dict[str, Any] = {
                "model_id": model_id,
                "target_condition": target,
                "reference_condition": reference,
                "training_seed": seed,
                "target_run_id": target_runs[0] if len(target_runs) == 1 else None,
                "reference_run_id": reference_runs[0] if len(reference_runs) == 1 else None,
                "minimum_training_seeds": minimum_training_seeds,
            }
            if len(target_runs) != 1 or len(reference_runs) != 1:
                row.update({
                    "status": "insufficient_data",
                    "reason": "expected exactly one run for each control condition and seed",
                })
            else:
                target_id, reference_id = target_runs[0], reference_runs[0]
                errors = {
                    run_id: evidence_errors[run_id]
                    for run_id in (target_id, reference_id)
                    if run_id in evidence_errors
                }
                left = evidence_by_run.get(target_id)
                right = evidence_by_run.get(reference_id)
                if errors:
                    row.update({"status": "fail", "reason": json.dumps(errors, sort_keys=True)})
                elif left is None or right is None:
                    row.update({"status": "insufficient_data", "reason": "batch evidence missing"})
                else:
                    left_ref = left["reference"]
                    right_ref = right["reference"]
                    left_sequence = [
                        (item.optimizer_step, item.microbatch_index, item.sample_position, item.sample)
                        for item in left["consumption"]
                    ]
                    right_sequence = [
                        (item.optimizer_step, item.microbatch_index, item.sample_position, item.sample)
                        for item in right["consumption"]
                    ]
                    same_manifest = (
                        left_ref.manifest_id == right_ref.manifest_id
                        and left_ref.dataset_revision == right_ref.dataset_revision
                        and left_ref.sha256 == right_ref.sha256
                        and left_ref.sample_count == right_ref.sample_count
                        and left["samples"] == right["samples"]
                    )
                    suffix_start = max(
                        left_ref.schedule_start_ordinal,
                        right_ref.schedule_start_ordinal,
                    )
                    left_suffix = [
                        item for item in left_sequence
                        if item[3].ordinal >= suffix_start
                    ]
                    right_suffix = [
                        item for item in right_sequence
                        if item[3].ordinal >= suffix_start
                    ]
                    left_identity = [(item[1], item[2], item[3]) for item in left_suffix]
                    right_identity = [(item[1], item[2], item[3]) for item in right_suffix]
                    expected_suffix_count = left_ref.sample_count - suffix_start
                    same_consumption = (
                        left_ref.schedule_start_ordinal <= suffix_start
                        and right_ref.schedule_start_ordinal <= suffix_start
                        and len(left_identity) == expected_suffix_count
                        and len(right_identity) == expected_suffix_count
                        and left_identity == right_identity
                        and [item[3].ordinal for item in left_suffix]
                        == list(range(suffix_start, left_ref.sample_count))
                        and [item[3].ordinal for item in right_suffix]
                        == list(range(suffix_start, right_ref.sample_count))
                    )
                    row.update({
                        "batch_manifest_id": left_ref.manifest_id,
                        "dataset_revision": left_ref.dataset_revision,
                        "batch_manifest_sha256_target": left_ref.sha256,
                        "batch_manifest_sha256_reference": right_ref.sha256,
                        "sample_count_target": len(left["samples"]),
                        "sample_count_reference": len(right["samples"]),
                        "consumed_record_count_target": len(left_sequence),
                        "consumed_record_count_reference": len(right_sequence),
                        "expected_optimizer_steps": left_ref.expected_optimizer_steps,
                        "target_schedule_start_ordinal": left_ref.schedule_start_ordinal,
                        "reference_schedule_start_ordinal": right_ref.schedule_start_ordinal,
                        "compared_schedule_start_ordinal": suffix_start,
                        "expected_compared_samples": expected_suffix_count,
                        "compared_samples_target": len(left_identity),
                        "compared_samples_reference": len(right_identity),
                        "same_manifest": same_manifest,
                        "same_consumption_sequence": same_consumption,
                        "status": "pass" if same_manifest and same_consumption else "fail",
                        "reason": None if same_manifest and same_consumption else
                                  "manifest or actual consumed sequence differs",
                    })
            model_rows.append(row)
            rows.append(row)

        passing_seed_count = sum(row["status"] == "pass" for row in model_rows)
        if any(row["status"] == "fail" for row in model_rows):
            model_status = "fail"
        elif passing_seed_count >= minimum_training_seeds and all(
            row["status"] == "pass" for row in model_rows
        ):
            model_status = "pass"
        else:
            model_status = "insufficient_data"
        model_statuses[model_id] = model_status
    if model_statuses and all(status == "pass" for status in model_statuses.values()):
        overall = "pass"
    elif any(status == "fail" for status in model_statuses.values()):
        overall = "fail"
    else:
        overall = "insufficient_data"
    return rows, {"status": overall, "by_model": model_statuses}


def _assess_capability_equivalence(
    *,
    model_ids: Sequence[str],
    capability_rows: Sequence[Mapping[str, Any]],
    benchmark_id: str,
    suite_id: str,
    target: str,
    reference: str,
    margin: float,
    minimum_training_seeds: int,
    confidence_level: float,
    applicable: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not applicable:
        return [], {
            "status": "not_applicable_comparison_override",
            "target_condition": target,
            "reference_condition": reference,
        }
    rows: list[dict[str, Any]] = []
    model_statuses: dict[str, str] = {}
    for model_id in model_ids:
        summary = _summary_row(
            capability_rows,
            model_id=model_id,
            benchmark_id=benchmark_id,
            suite_id=suite_id,
            target=target,
            reference=reference,
        )
        seed_count = int(summary.get("training_seed_count", 0)) if summary else 0
        ci_low = summary.get("bootstrap_ci_low") if summary else None
        ci_high = summary.get("bootstrap_ci_high") if summary else None
        if seed_count < minimum_training_seeds or ci_low is None or ci_high is None:
            status = "insufficient_data"
        elif ci_low >= -margin and ci_high <= margin:
            status = "pass"
        else:
            status = "fail"
        model_statuses[model_id] = status
        rows.append({
            "model_id": model_id,
            "benchmark_id": benchmark_id,
            "suite_id": suite_id,
            "target_condition": target,
            "reference_condition": reference,
            "training_seed_count": seed_count,
            "minimum_training_seeds": minimum_training_seeds,
            "equivalence_margin_absolute": margin,
            "confidence_level": confidence_level,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "status": status,
            "test_label": "descriptive_cluster_interval_screen_not_TOST",
        })
    if model_statuses and all(status == "pass" for status in model_statuses.values()):
        overall = "pass"
    elif any(status == "fail" for status in model_statuses.values()):
        overall = "fail"
    else:
        overall = "insufficient_data"
    return rows, {
        "status": overall,
        "target_condition": target,
        "reference_condition": reference,
        "by_model": model_statuses,
        "equivalence_margin_absolute": margin,
        "confidence_level": confidence_level,
    }


def analyze_runs(
    *,
    workspace_root: str | Path,
    run_ids: Sequence[str],
    analysis_id: str,
    comparison: str,
    config_path: str | Path,
    output_root: str | Path | None = None,
) -> Path:
    if not run_ids or len(run_ids) != len(set(run_ids)):
        raise ContractError("provide a non-empty list of unique run ids")
    for run_id in run_ids:
        validate_run_id(run_id)
    validate_run_id(analysis_id)
    roots = RuntimeRoots.from_env(workspace_root)
    analysis_config = load_yaml(config_path)
    target, reference = parse_comparison(comparison)
    condition_labels = analysis_config.get("condition_labels", {})
    configured_primary = str(analysis_config["primary_safety_estimand"]["comparison"])
    primary_target, primary_reference = parse_condition_relation(
        configured_primary,
        condition_labels,
        relation=" minus ",
        legacy_relation="_minus_",
    )
    control_config = analysis_config["resume_control"]
    control_target, control_reference = parse_condition_relation(
        str(control_config["comparison"]),
        condition_labels,
        relation=" equivalent to ",
        legacy_relation="_equivalent_to_",
    )
    candidate_config = analysis_config["candidate_signal"]
    if candidate_config.get("required_control") != "resume_control.comparison":
        raise ContractError("candidate signal must require the configured resume control")
    if candidate_config.get("expected_direction") != "positive_risk_increase":
        raise ContractError("candidate signal direction must be positive_risk_increase")
    if candidate_config.get("direction_rule") != "positive_seed_effects":
        raise ContractError("candidate signal direction_rule must be positive_seed_effects")
    margin = float(analysis_config["primary_capability_estimand"]["equivalence_margin_absolute"])
    control_safety = control_config["safety_estimand"]
    control_capability = control_config["capability_estimand"]
    if control_capability.get("equivalence_margin_source") != (
        "primary_capability_estimand.equivalence_margin_absolute"
    ):
        raise ContractError("resume control capability margin must use the configured primary margin")
    control_safety_margin = float(control_safety["equivalence_margin_absolute"])
    control_minimum_seeds = int(control_config["minimum_training_seeds"])
    raw_control_seeds = control_config.get("training_seeds")
    if (
        not isinstance(raw_control_seeds, list)
        or not raw_control_seeds
        or any(type(seed) is not int or seed < 0 for seed in raw_control_seeds)
        or len(raw_control_seeds) != len(set(raw_control_seeds))
    ):
        raise ContractError("resume control training_seeds must be unique non-negative integers")
    control_training_seeds = list(raw_control_seeds)
    if len(control_training_seeds) < control_minimum_seeds:
        raise ContractError("resume control seed list is below minimum_training_seeds")
    control_confidence = float(control_config["confidence_level"])
    bootstrap = analysis_config.get("bootstrap", {})
    resamples = int(bootstrap.get("resamples", 20000))
    safety_level = float(bootstrap.get("safety_confidence_level", 0.95))
    capability_level = float(bootstrap.get("capability_confidence_level", 0.90))
    bootstrap_seed = int(bootstrap.get("random_seed", 20261102))
    if (
        resamples <= 0
        or not 0 < safety_level < 1
        or not 0 < capability_level < 1
        or not 0 < control_confidence < 1
        or control_minimum_seeds <= 0
        or not math.isfinite(margin)
        or margin < 0
        or not math.isfinite(control_safety_margin)
        or control_safety_margin < 0
        or not math.isfinite(float(candidate_config["safety_risk_difference_absolute"]))
        or float(candidate_config["safety_risk_difference_absolute"]) < 0
        or int(candidate_config["minimum_training_seeds"]) <= 0
        or int(candidate_config["same_direction_seeds_minimum"]) <= 0
        or int(candidate_config["same_direction_seeds_minimum"])
        > int(candidate_config["minimum_training_seeds"])
    ):
        raise ContractError("invalid bootstrap settings in analysis config")

    source_files: list[dict[str, Any]] = []
    records: list[EpisodeRecord] = []
    run_identities: dict[str, tuple[str, str, int]] = {}
    batch_evidence: dict[str, dict[str, Any] | None] = {}
    batch_evidence_errors: dict[str, str] = {}
    for run_id in run_ids:
        run_dir = roots.run_root / run_id
        episode_path = roots.run_root / run_id / "episodes" / "episodes.jsonl"
        provenance_path = run_dir / "metadata" / "provenance.json"
        resolved_config_path = run_dir / "metadata" / "resolved_config.yaml"
        run_records = read_episode_jsonl(episode_path, expected_run_id=run_id)
        if not provenance_path.is_file() or not resolved_config_path.is_file():
            raise ContractError(
                f"run {run_id} is missing metadata/provenance.json or "
                "metadata/resolved_config.yaml"
            )
        try:
            provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid provenance for run {run_id}: {exc}") from exc
        provenance = provenance_from_mapping(provenance_payload)
        resolved_run_config = load_yaml(resolved_config_path)
        if provenance.run_id != run_id:
            raise ContractError(f"provenance run_id does not match requested run: {run_id}")
        if provenance.ended_at_utc is None:
            raise ContractError(f"run {run_id} has no end timestamp; it is not a completed run")
        if sha256_payload(resolved_run_config) != provenance.config_hash:
            raise ContractError(f"resolved config hash mismatch for run {run_id}")
        run_identity = {
            (record.model_id, record.condition_id, record.training_seed)
            for record in run_records
        }
        if len(run_identity) != 1:
            raise ContractError(
                f"run {run_id} mixes model, condition or training seed identities: "
                f"{sorted(run_identity)}"
            )
        model_id, condition_id, training_seed = next(iter(run_identity))
        if provenance.model_id != model_id or provenance.model_seed != training_seed:
            raise ContractError(
                f"episode identity does not match provenance for run {run_id}"
            )
        run_identities[run_id] = (model_id, condition_id, training_seed)
        batch_ref_path = run_dir / "metadata" / "batch_manifest.json"
        batch_consumption_path = run_dir / "events" / "batch_consumption.jsonl"
        if not batch_ref_path.is_file() or not batch_consumption_path.is_file():
            batch_evidence[run_id] = None
        else:
            try:
                batch_evidence[run_id] = _read_run_batch_evidence(
                    run_dir=run_dir,
                    roots=roots,
                    run_id=run_id,
                    model_id=model_id,
                    condition_id=condition_id,
                    training_seed=training_seed,
                )
            except (ContractError, OSError, ValueError) as exc:
                batch_evidence[run_id] = None
                batch_evidence_errors[run_id] = str(exc)
        records.extend(run_records)
        source_row = {
            "run_id": run_id,
            "path": str(episode_path),
            "sha256": sha256_file(episode_path),
            "provenance_sha256": sha256_file(provenance_path),
            "resolved_config_file_sha256": sha256_file(resolved_config_path),
            "episode_count": len(run_records),
        }
        evidence = batch_evidence.get(run_id)
        if evidence is not None:
            source_row["batch_manifest_ref_sha256"] = evidence["reference_file_sha256"]
            source_row["batch_manifest_sha256"] = evidence["manifest_sha256"]
            source_row["batch_consumption_sha256"] = evidence["consumption_file_sha256"]
        source_files.append(source_row)

    run_summary = summarize_runs(records)
    safety_rows, capability_rows = paired_comparison(
        records,
        target_condition=target,
        reference_condition=reference,
        equivalence_margin=margin,
        confidence_safety=safety_level,
        confidence_capability=capability_level,
        bootstrap_resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )
    if not any(row.get("row_type") == "training_seed" for row in safety_rows):
        raise ContractError(f"no paired records found for comparison {target}:{reference}")
    libero10_rows = [
        row for row in capability_rows
        if row.get("benchmark_id") == "libero10"
    ]
    if not libero10_rows:
        raise ContractError("no paired capability records found for benchmark_id='libero10'")

    control_safety_rows, control_capability_rows = paired_comparison(
        records,
        target_condition=control_target,
        reference_condition=control_reference,
        equivalence_margin=margin,
        confidence_safety=control_confidence,
        confidence_capability=control_confidence,
        bootstrap_resamples=resamples,
        bootstrap_seed=bootstrap_seed + 2,
    )
    model_ids = sorted({record.model_id for record in records})
    raw_control_suites = control_safety.get("suites")
    if (
        not isinstance(raw_control_suites, list)
        or not raw_control_suites
        or any(not isinstance(suite, str) or not suite.strip() for suite in raw_control_suites)
        or len(raw_control_suites) != len(set(raw_control_suites))
    ):
        raise ContractError("resume control safety suites must be a non-empty unique string list")
    control_suites = list(raw_control_suites)
    control_rows, resume_control_result = _assess_resume_control(
        model_ids=model_ids,
        safety_rows=control_safety_rows,
        capability_rows=control_capability_rows,
        target=control_target,
        reference=control_reference,
        safety_benchmark_id=str(control_safety["benchmark_id"]),
        safety_suites=control_suites,
        safety_margin=control_safety_margin,
        capability_benchmark_id=str(control_capability["benchmark_id"]),
        capability_suite_id=str(control_capability["suite_id"]),
        capability_margin=margin,
        minimum_training_seeds=control_minimum_seeds,
        confidence_level=control_confidence,
    )
    candidate_applicable = (target, reference) == (primary_target, primary_reference)
    control_batch_rows, control_batch_result = _audit_control_batch_sequences(
        model_ids=model_ids,
        training_seeds=control_training_seeds,
        evidence_by_run=batch_evidence,
        evidence_errors=batch_evidence_errors,
        identities=run_identities,
        target=control_target,
        reference=control_reference,
        minimum_training_seeds=control_minimum_seeds,
    )
    if resume_control_result["status"] == "fail" or control_batch_result["status"] == "fail":
        resume_control_status = "fail"
    elif (
        resume_control_result["status"] == "pass"
        and control_batch_result["status"] == "pass"
    ):
        resume_control_status = "pass"
    else:
        resume_control_status = "insufficient_data"
    resume_control_result = {
        **resume_control_result,
        "statistical_status": resume_control_result["status"],
        "batch_sequence_status": control_batch_result["status"],
        "batch_sequence_audit": control_batch_result,
        "status": resume_control_status,
    }
    primary_capability_config = analysis_config["primary_capability_estimand"]
    primary_capability_rows, primary_capability_result = _assess_capability_equivalence(
        model_ids=model_ids,
        capability_rows=capability_rows,
        benchmark_id=str(primary_capability_config["benchmark_id"]),
        suite_id=str(primary_capability_config["suite_id"]),
        target=target,
        reference=reference,
        margin=margin,
        minimum_training_seeds=int(candidate_config["minimum_training_seeds"]),
        confidence_level=capability_level,
        applicable=candidate_applicable,
    )
    if candidate_applicable:
        primary_batch_rows, primary_batch_result = _audit_control_batch_sequences(
            model_ids=model_ids,
            training_seeds=control_training_seeds,
            evidence_by_run=batch_evidence,
            evidence_errors=batch_evidence_errors,
            identities=run_identities,
            target=target,
            reference=reference,
            minimum_training_seeds=control_minimum_seeds,
        )
    else:
        primary_batch_rows = []
        primary_batch_result = {
            "status": "not_applicable_comparison_override",
            "target_condition": target,
            "reference_condition": reference,
        }
    candidate_result = _assess_candidate_signal(
        model_ids=model_ids,
        safety_rows=safety_rows,
        target=target,
        reference=reference,
        benchmark_id=str(control_safety["benchmark_id"]),
        suites=control_suites,
        risk_threshold=float(candidate_config["safety_risk_difference_absolute"]),
        minimum_training_seeds=int(candidate_config["minimum_training_seeds"]),
        same_direction_minimum=int(candidate_config["same_direction_seeds_minimum"]),
        require_all_suites=bool(candidate_config["require_all_safety_suites"]),
        applicable=candidate_applicable,
    )
    candidate_claim_supported = (
        candidate_result["status"] == "candidate_signal"
        and resume_control_result["status"] == "pass"
        and primary_capability_result["status"] == "pass"
        and primary_batch_result["status"] == "pass"
    )

    condition_labels = analysis_config.get("condition_labels", {})
    if not isinstance(condition_labels, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in condition_labels.items()
    ):
        raise ContractError("condition_labels must map condition ids to non-empty display names")

    def add_condition_labels(row: dict[str, Any]) -> None:
        for field, label_field in (
            ("condition_id", "condition_label"),
            ("target_condition", "target_condition_label"),
            ("reference_condition", "reference_condition_label"),
        ):
            value = row.get(field)
            if value is not None:
                row[label_field] = condition_labels.get(value, value)

    for collection in (
        run_summary,
        safety_rows,
        capability_rows,
        control_rows,
        control_batch_rows,
        primary_capability_rows,
        primary_batch_rows,
        control_safety_rows,
        control_capability_rows,
    ):
        for row in collection:
            add_condition_labels(row)
    for result in (
        resume_control_result,
        primary_capability_result,
        primary_batch_result,
        control_batch_result,
        candidate_result,
    ):
        add_condition_labels(result)

    root = Path(output_root).expanduser().resolve() if output_root else roots.run_root
    output_dir = root / analysis_id
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_dir = output_dir / "metadata"
    metrics_dir = output_dir / "metrics"
    metadata_dir.mkdir()
    metrics_dir.mkdir()

    resolved_config = {
        **analysis_config,
        "resolved": {
            "analysis_id": analysis_id,
            "source_run_ids": list(run_ids),
            "comparison": {"target": target, "reference": reference},
            "display_comparison": {
                "target": condition_labels.get(target, target),
                "reference": condition_labels.get(reference, reference),
            },
            "condition_labels": dict(condition_labels),
            "resume_control_comparison": {
                "target": control_target,
                "reference": control_reference,
            },
            "bootstrap_resamples": resamples,
            "bootstrap_random_seed": bootstrap_seed,
        },
    }
    resolved_config_hash = sha256_payload(resolved_config)
    analysis_config_path = Path(config_path).expanduser().resolve()
    source_analysis_config_hash = sha256_file(analysis_config_path)
    created_at_utc = datetime.now(timezone.utc).isoformat()
    code_commit = _git_commit(roots.workspace_root)
    code_dirty = _git_dirty(roots.workspace_root)
    manifest = {
        "schema": "analysis_manifest.v1",
        "analysis_id": analysis_id,
        "created_at_utc": created_at_utc,
        "workspace_commit": code_commit,
        "workspace_dirty": code_dirty,
        "analysis_config": str(analysis_config_path),
        "source_analysis_config_sha256": source_analysis_config_hash,
        "resolved_config_sha256": resolved_config_hash,
        "analysis_source_sha256": {
            relative: sha256_file(Path(__file__).resolve().parents[1] / relative)
            for relative in (
                "analysis/episodes.py",
                "analysis/cli.py",
                "common/config.py",
                "common/identity.py",
                "common/provenance.py",
                "common/roots.py",
                "data/io.py",
                "data/schemas.py",
                "training/contracts.py",
            )
        },
        "comparison": {"target": target, "reference": reference},
        "display_comparison": {
            "target": condition_labels.get(target, target),
            "reference": condition_labels.get(reference, reference),
        },
        "condition_labels": dict(condition_labels),
        "resume_control_comparison": {
            "target": control_target,
            "reference": control_reference,
        },
        "source_runs": source_files,
        "episode_count": len(records),
        "statistical_unit": "training_seed",
        "interval_method": "percentile bootstrap over training-seed clusters; descriptive pilot",
    }
    (metadata_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "analysis_summary.v1",
        "analysis_id": analysis_id,
        "created_at_utc": created_at_utc,
        "code_commit": code_commit,
        "workspace_dirty": code_dirty,
        "source_analysis_config_sha256": source_analysis_config_hash,
        "resolved_config_sha256": resolved_config_hash,
        "source_run_ids": list(run_ids),
        "comparison": {"target": target, "reference": reference},
        "resume_control": resume_control_result,
        "resume_control_checks": control_rows,
        "resume_control_batch_sequence_audit": control_batch_result,
        "resume_control_batch_sequence_rows": control_batch_rows,
        "primary_capability_equivalence": primary_capability_result,
        "primary_capability_equivalence_rows": primary_capability_rows,
        "primary_batch_sequence_audit": primary_batch_result,
        "primary_batch_sequence_rows": primary_batch_rows,
        "resume_control_safety_results": control_safety_rows,
        "resume_control_capability_results": [
            row for row in control_capability_rows
            if row.get("benchmark_id") == control_capability["benchmark_id"]
            and row.get("suite_id") == control_capability["suite_id"]
        ],
        "candidate_signal": candidate_result,
        "candidate_claim_supported": candidate_claim_supported,
        "total_episode_records": len(records),
        "statistical_unit": "training_seed",
        "run_summaries": run_summary,
        "paired_safety_results": safety_rows,
        "libero10_capability_results": libero10_rows,
        "exclusions": {
            "unmatched_target_pairs": sum(
                int(row.get("unmatched_target", 0))
                for row in safety_rows if row.get("row_type") == "training_seed"
            ),
            "unmatched_reference_pairs": sum(
                int(row.get("unmatched_reference", 0))
                for row in safety_rows if row.get("row_type") == "training_seed"
            ),
            "pairs_with_unknown_safety_outcome": sum(
                int(row.get("unknown_outcome_pairs", 0))
                for row in safety_rows if row.get("row_type") == "training_seed"
            ),
            "pairs_missing_capability_label": sum(
                int(row.get("missing_success_pairs", 0))
                for row in libero10_rows if row.get("row_type") == "training_seed"
            ),
        },
        "confirmatory_claims_allowed": bool(
            analysis_config.get("pilot", {}).get("confirmatory_claims_allowed", False)
        ),
        "outputs": [
            "summary.json",
            "run_summary.csv",
            "paired_safety_risk_by_condition.csv",
            "libero10_capability_equivalence.csv",
            "resume_control_equivalence.csv",
            "resume_control_safety_risk_by_condition.csv",
            "resume_control_capability_equivalence.csv",
            "resume_control_batch_sequence_audit.csv",
            "primary_capability_equivalence_gate.csv",
            "primary_batch_sequence_audit.csv",
            "candidate_signal.json",
        ],
    }
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(metrics_dir / "run_summary.csv", run_summary)
    _write_csv(metrics_dir / "paired_safety_risk_by_condition.csv", safety_rows)
    _write_csv(metrics_dir / "libero10_capability_equivalence.csv", libero10_rows)
    _write_csv(metrics_dir / "resume_control_equivalence.csv", control_rows)
    _write_csv(metrics_dir / "resume_control_batch_sequence_audit.csv", control_batch_rows)
    _write_csv(metrics_dir / "primary_capability_equivalence_gate.csv", primary_capability_rows)
    _write_csv(metrics_dir / "primary_batch_sequence_audit.csv", primary_batch_rows)
    control_safety_output = [
        {"comparison_role": "resume_control", **row}
        for row in control_safety_rows
    ]
    control_capability_output = [
        {"comparison_role": "resume_control", **row}
        for row in control_capability_rows
        if row.get("benchmark_id") == control_capability["benchmark_id"]
        and row.get("suite_id") == control_capability["suite_id"]
    ]
    _write_csv(metrics_dir / "resume_control_safety_risk_by_condition.csv", control_safety_output)
    _write_csv(metrics_dir / "resume_control_capability_equivalence.csv", control_capability_output)
    (metrics_dir / "candidate_signal.json").write_text(
        json.dumps({
            "schema": "candidate_signal_assessment.v1",
            "candidate_signal": candidate_result,
            "resume_control": resume_control_result,
            "resume_control_batch_sequence_audit": control_batch_result,
            "primary_capability_equivalence": primary_capability_result,
            "primary_batch_sequence_audit": primary_batch_result,
            "claim_supported": candidate_claim_supported,
            "claim_limit": "pilot assessment only; not confirmatory evidence",
        }, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir
