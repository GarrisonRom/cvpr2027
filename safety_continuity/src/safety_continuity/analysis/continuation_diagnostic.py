"""Descriptive paired diagnostics, without episode-level training replication."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.config import sha256_file
from ..common.errors import ContractError
from ..evaluation.continuation_diagnostic import compare_actions, write_json


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_events(original: list[dict[str, Any]], adapted: list[dict[str, Any]], field: str) -> dict[str, Any]:
    def index(rows):
        result = {row["episode_key"]: row for row in rows}
        if len(result) != len(rows):
            raise ContractError("duplicate paired episode key")
        return result
    left, right = index(original), index(adapted)
    if not left or left.keys() != right.keys():
        raise ContractError("paired episode coverage differs")
    retained = []
    excluded = 0
    for key in left:
        a, b = left[key], right[key]
        if (a["initial_state_sha256"] != b["initial_state_sha256"]
                or a["evaluation_seed"] != b["evaluation_seed"]
                or a["state_manifest_sha256"] != b["state_manifest_sha256"]):
            raise ContractError("paired initial state or RNG differs")
        if field == "new_contact_without_initial_contact" and (a["initial_human_contact"] or b["initial_human_contact"]):
            excluded += 1
            continue
        if type(a[field]) is not bool or type(b[field]) is not bool:
            raise ContractError("paired diagnostic event must be boolean")
        retained.append((a, b))
    if not retained:
        return {"n": 0, "excluded_initial_contact": excluded, "delta": None, "status": "no_eligible_pairs"}
    task_ids = sorted({a["task_id"] for a, _ in retained})
    task_deltas = []
    for task in task_ids:
        pairs = [(a,b) for a,b in retained if a["task_id"] == task]
        task_deltas.append(sum(int(b[field])-int(a[field]) for a,b in pairs)/len(pairs))
    return {"n": len(retained), "excluded_initial_contact": excluded,
            "original_count": sum(a[field] for a,b in retained), "adapted_count": sum(b[field] for a,b in retained),
            "new_events": sum(not a[field] and b[field] for a,b in retained),
            "removed_events": sum(a[field] and not b[field] for a,b in retained),
            "delta": sum(task_deltas)/len(task_deltas), "task_deltas": dict(zip(task_ids, task_deltas)),
            "status": "descriptive_only"}


def repetition_audit(root: Path, candidate: str, repetition: str, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    result = {"source": candidate, "pass": repetition, "episodes": 0,
              "event_mismatches": 0, "trajectory_mismatches": 0, "state_hash_mismatches": 0,
              "max_action_delta": 0.0, "input_identity_mismatches": 0, "first_divergences": []}
    for suite in ("capability", "human_safety", "reasoning_safety"):
        base_dir = root/"evaluations/original/primary"/suite
        test_dir = root/"evaluations"/candidate/repetition/suite
        left = {r["episode_key"]:r for r in read_rows(base_dir/"episodes.jsonl")}
        rows = read_rows(test_dir/"episodes.jsonl")
        if not rows:
            raise ContractError("empty repetition batch")
        keys = {r["episode_key"] for r in rows}
        if len(keys) != len(rows) or not keys <= left.keys():
            raise ContractError("repetition episode coverage differs")
        steps = {}
        for label, folder in (("original", base_dir), ("candidate", test_dir)):
            for row in read_rows(folder/"steps.jsonl"):
                if row["episode_key"] in keys:
                    steps.setdefault((label,row["episode_key"]), []).append(row)
        for row in rows:
            a = left[row["episode_key"]]
            if (a["initial_state_sha256"] != row["initial_state_sha256"]
                    or a["evaluation_seed"] != row["evaluation_seed"]):
                raise ContractError("repetition input identity differs")
            fields = ("success", "post_takeover_human_contact", "initial_human_contact", "harmful_completion", "steps")
            result["event_mismatches"] += any(a[f] != row[f] for f in fields)
            original_steps, candidate_steps = (steps[(label,row["episode_key"])] for label in ("original","candidate"))
            if len(original_steps) != len(candidate_steps):
                result["trajectory_mismatches"] += 1
            else:
                for a_step,b_step in zip(original_steps,candidate_steps):
                    identity_diff = a_step.get("policy_query_input") != b_step.get("policy_query_input")
                    result["input_identity_mismatches"] += identity_diff
                    action_diff = not np.allclose(a_step["action"],b_step["action"],atol=config["action_atol"],rtol=config["action_rtol"])
                    if identity_diff or action_diff:
                        result["first_divergences"].append({"episode_key":row["episode_key"],"step":a_step["step"],
                            "input_differs":identity_diff,"original_input":a_step.get("policy_query_input"),
                            "candidate_input":b_step.get("policy_query_input"),"action_exceeds_tolerance":action_diff})
                        break
                actions = compare_actions(np.asarray([r["action"] for r in original_steps]),
                                          np.asarray([r["action"] for r in candidate_steps]),
                                          atol=config["action_atol"], rtol=config["action_rtol"])
                result["max_action_delta"] = max(result["max_action_delta"], actions["max_abs_delta"])
                result["trajectory_mismatches"] += not actions["within_tolerance"]
                result["state_hash_mismatches"] += sum(a["sim_state_sha256"] != b["sim_state_sha256"]
                                                       for a,b in zip(original_steps,candidate_steps))
            result["episodes"] += 1
    result["event_and_action_repeatability_verified"] = not (result["event_mismatches"] or result["trajectory_mismatches"])
    return result


def analyze(root: Path) -> dict[str, Any]:
    import numpy as np
    plan = json.loads((root/"metadata/plan.json").read_text(encoding="utf-8"))
    status = json.loads((root/"metadata/diagnostic_status.json").read_text(encoding="utf-8"))
    if status["status"] != "measurements_ready_for_analysis":
        raise ContractError("diagnostic measurements are not ready")
    provenance = json.loads((root/"metadata/provenance.json").read_text(encoding="utf-8"))
    if sha256_file(root/"metadata/resolved_config.yaml") != provenance["config_sha256"]:
        raise ContractError("resolved diagnostic config changed")
    for item in status["completed"]:
        if "suite" in item:
            path = Path(item["summary"])
            summary = json.loads(path.read_text())
            for name, digest in (("episodes.jsonl", "episodes_sha256"), ("steps.jsonl", "steps_sha256")):
                if sha256_file(path.parent/name) != summary[digest]:
                    raise ContractError("rollout record hash mismatch")
    audit = [repetition_audit(root,"original","repeat",plan["config"]),
             repetition_audit(root,"reloaded","primary",plan["config"])]
    events = []
    fields = {"capability": ["success"], "human_safety": ["success", "post_takeover_human_contact", "new_contact_without_initial_contact"],
              "reasoning_safety": ["success", "harmful_completion"]}
    for source in plan["sources"]:
        if source["id"] == "original" or not source["rollout"]:
            continue
        for suite in fields:
            original = read_rows(root/"evaluations/original/primary"/suite/"episodes.jsonl")
            adapted = read_rows(root/"evaluations"/source["id"]/"primary"/suite/"episodes.jsonl")
            for field in fields[suite]:
                events.append({"source_id": source["id"], "label": source["label"], "training_seed": source["training_seed"],
                               "suite": suite, "field": field, **paired_events(original,adapted,field)})
    contrasts = []
    ids = {s["id"] for s in plan["sources"]}
    for seed in (42,43,44):
        for left,right in ((f"continued2000s{seed}",f"fullresumes{seed}"),
                           (f"fullresumes{seed}",f"momentresets{seed}"),
                           (f"fullresumes{seed}",f"weightsrestarts{seed}")):
            if left not in ids or right not in ids:
                continue
            files = sorted((root/"probes"/left).glob("*.npz"))
            if not files:
                raise ContractError("empty action contrast")
            differences = []
            for path in files:
                with np.load(path,allow_pickle=False) as a, np.load(root/"probes"/right/path.name,allow_pickle=False) as b:
                    differences.append(compare_actions(a["actions"],b["actions"],atol=plan["config"]["action_atol"],rtol=plan["config"]["action_rtol"]))
            contrasts.append({"reference": left,"candidate":right,"probe_count":len(differences),
                              "changed_probe_count":sum(not d["within_tolerance"] for d in differences),
                              "max_abs_action_delta":max(d["max_abs_delta"] for d in differences)})
    summary = {"schema":"continuation_diagnostic_analysis.v1","run_id":root.name,
               "scope":plan["config"]["scope"], "repetition_audits":audit, "paired_events":events,
               "resume_action_contrasts":contrasts,"scientific_claim_supported":False,
               "capability_equivalence_established":False,"official_safety_claim_eligible":False,
               "limits":["Exploratory small state subset; no confirmatory training-population inference.",
                         "Human contact is not injury; task-specific permission and dynamic severity not established.",
                         "No new benign-data audit or broader adaptation recipe was run.",
                         "Reload control restarts the inference process; it does not save or convert the checkpoint."]}
    write_json(root/"metrics/analysis.json",summary)
    lines = ["# 继续训练稳定性诊断", "", f"Run: `{root.name}`", "",
             "这是历史输出层训练的探索诊断，不能直接证明良性适配安全遗忘或真实伤害增加。", "",
             "## 重复性与重新加载", ""]
    for row in audit:
        lines.append(f'- {row["source"]}/{row["pass"]}: {row["episodes"]} 条配对轨迹，事件不一致 {row["event_mismatches"]}，动作轨迹超差 {row["trajectory_mismatches"]}，最大动作差 {row["max_action_delta"]:.6g}。')
    lines += ["", "## 适配前后配对计数", "", "| 条件 | 套件 | 事件 | 原始/适配计数 | 配对数 | 任务等权差值 |", "|---|---|---|---|---|---|"]
    for row in events:
        counts = f'{row.get("original_count",0)}/{row.get("adapted_count",0)}'
        delta = "不可估计" if row["delta"] is None else f'{row["delta"]*100:+.1f}pp'
        lines.append(f'| {row["label"]} | {row["suite"]} | {row["field"]} | {counts} | {row["n"]} | {delta} |')
    lines += ["", "完整动作比较、排除的初始接触数及限制见 `metrics/analysis.json`。",
              "动作变化不等于安全问题；没有危险完成也不等于已拒绝。", ""]
    (root/"metrics/report.md").write_text("\n".join(lines),encoding="utf-8")
    return summary
