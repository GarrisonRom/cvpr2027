"""Immutable run identity helpers."""

from __future__ import annotations

import re
from datetime import date

from .errors import ContractError

_RUN_ID = re.compile(r"^[0-9]{8}_[a-z0-9]+_[a-z0-9]+_[a-z0-9]+_s[0-9]{2}$")


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ContractError(
            "run_id must match YYYYMMDD_branch_model_condition_sNN using lowercase ASCII"
        )
    return run_id


def make_run_id(
    branch: str,
    model: str,
    condition: str,
    seed: int,
    *,
    run_date: date | None = None,
) -> str:
    if seed < 0 or seed > 99:
        raise ContractError("seed must fit the two-digit run ID suffix")
    run_id = f"{(run_date or date.today()):%Y%m%d}_{branch}_{model}_{condition}_s{seed:02d}"
    return validate_run_id(run_id)
