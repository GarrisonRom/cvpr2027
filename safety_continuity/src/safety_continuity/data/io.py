"""Strict JSONL readers for canonical experiment records."""

from __future__ import annotations

import json
from pathlib import Path

from ..common.errors import ContractError
from .schemas import (
    BatchConsumptionRecord,
    EpisodeRecord,
    UpdateSampleRef,
    batch_consumption_from_mapping,
    episode_from_mapping,
    update_sample_from_mapping,
    validate_episode_batch,
)


def read_episode_jsonl(
    path: str | Path,
    *,
    expected_run_id: str | None = None,
) -> list[EpisodeRecord]:
    source = Path(path)
    if not source.is_file():
        raise ContractError(f"episode file does not exist: {source}")

    records: list[EpisodeRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ContractError("record must be a JSON object")
                if payload.get("schema") != "episode.v1":
                    raise ContractError(
                        f"expected schema='episode.v1', got {payload.get('schema')!r}"
                    )
                record = episode_from_mapping(payload)
                if expected_run_id is not None and record.run_id != expected_run_id:
                    raise ContractError(
                        f"record run_id {record.run_id!r} does not match requested "
                        f"run {expected_run_id!r}"
                    )
                records.append(record)
            except (json.JSONDecodeError, TypeError, KeyError, ContractError) as exc:
                raise ContractError(f"{source}:{line_number}: {exc}") from exc

    if not records:
        raise ContractError(f"episode file contains no records: {source}")
    validate_episode_batch(records)
    return records


def read_update_sample_jsonl(path: str | Path) -> tuple[UpdateSampleRef, ...]:
    source = Path(path)
    if not source.is_file():
        raise ContractError(f"update sample manifest does not exist: {source}")
    samples: list[UpdateSampleRef] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ContractError("update sample row must be a JSON object")
                samples.append(update_sample_from_mapping(payload))
            except (json.JSONDecodeError, TypeError, KeyError, ContractError) as exc:
                raise ContractError(f"{source}:{line_number}: {exc}") from exc
    if not samples:
        raise ContractError(f"update sample manifest is empty: {source}")
    ordinals = [sample.ordinal for sample in samples]
    if ordinals != list(range(len(samples))):
        raise ContractError("update sample ordinals must match file order from zero")
    return tuple(samples)


def read_batch_consumption_jsonl(
    path: str | Path,
    *,
    expected_run_id: str,
) -> tuple[BatchConsumptionRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise ContractError(f"batch consumption trace does not exist: {source}")
    records: list[BatchConsumptionRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ContractError("batch consumption row must be a JSON object")
                record = batch_consumption_from_mapping(payload)
                if record.run_id != expected_run_id:
                    raise ContractError(
                        f"record run_id {record.run_id!r} does not match "
                        f"requested run {expected_run_id!r}"
                    )
                records.append(record)
            except (json.JSONDecodeError, TypeError, KeyError, ContractError) as exc:
                raise ContractError(f"{source}:{line_number}: {exc}") from exc
    if not records:
        raise ContractError(f"batch consumption trace is empty: {source}")
    if [record.sequence_index for record in records] != list(range(len(records))):
        raise ContractError("batch consumption sequence_index must be contiguous from zero")
    positions = [
        (record.optimizer_step, record.microbatch_index, record.sample_position)
        for record in records
    ]
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise ContractError("batch consumption positions must be strictly ordered and unique")
    steps = sorted({record.optimizer_step for record in records})
    if steps != list(range(1, steps[-1] + 1)):
        raise ContractError("batch consumption optimizer steps must be contiguous from one")
    grouped_positions: dict[tuple[int, int], list[int]] = {}
    for record in records:
        grouped_positions.setdefault(
            (record.optimizer_step, record.microbatch_index), []
        ).append(record.sample_position)
    for group, sample_positions in grouped_positions.items():
        if sample_positions != list(range(len(sample_positions))):
            raise ContractError(f"batch consumption positions are not contiguous in {group}")
    for step in steps:
        microbatches = sorted({
            record.microbatch_index for record in records
            if record.optimizer_step == step
        })
        if microbatches != list(range(len(microbatches))):
            raise ContractError(f"batch consumption microbatches are not contiguous in step {step}")
    return tuple(records)
