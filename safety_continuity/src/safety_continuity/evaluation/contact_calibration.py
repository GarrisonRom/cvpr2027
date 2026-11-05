"""Live MuJoCo extraction checks, kept separate from policy evaluation."""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

from ..common.config import sha256_file, sha256_payload
from ..common.errors import ContractError
from ..safety.contacts import CONTACT_VERSION, HumanContactMonitor, build_contact_policy


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContractError(f"refusing to overwrite calibration artifact: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _case(env: Any, policy: Any, name: str, expected: str | None, required_pairs: list[list[str]], state_dir: Path) -> dict[str, Any]:
    import numpy as np
    monitor = HumanContactMonitor(policy)
    result = monitor.observe(simulator_state=env.sim, action=[0.0] * 7)
    actual_pairs = {frozenset((c["geom1"], c["geom2"])) for c in monitor.contacts}
    present = all(frozenset(pair) in actual_pairs for pair in required_pairs)
    state_path = state_dir / f"{name}.npy"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("xb") as handle:
        np.save(handle, np.asarray(env.sim.get_state().flatten()))
    return {
        "case": name, "expected_violation": expected,
        "observed_violation": result.violation_type, "required_pairs": required_pairs,
        "required_contacts_observed": present, "contacts": monitor.contacts,
        "matched": present and result.violation_type == expected,
        "state_path": str(state_path), "state_sha256": sha256_file(state_path),
        "method": "independent_geometry_ids_with_kinematic_placement_and_mj_forward",
    }


def _place_until_contact(env: Any, *, obj: Any, source_geom: str, targets: set[str], forbidden: set[str], forbidden_pairs: set[frozenset[int]] | None = None) -> list[str] | None:
    """Search fixed offsets; oracle uses explicit simulator IDs, never monitor labels."""
    import numpy as np
    sim = env.sim
    model, data = sim.model, sim.data
    if len(obj.joints) != 1:
        raise ContractError("calibration placement requires one free object joint")
    joint = obj.joints[0]
    original = data.get_joint_qpos(joint).copy()
    if len(original) != 7:
        raise ContractError("calibration placement requires a free joint")
    source_id = int(model.geom_name2id(source_geom))
    target_ids = {int(model.geom_name2id(name)) for name in targets}
    forbidden_ids = {int(model.geom_name2id(name)) for name in forbidden}
    source_origin = data.geom_xpos[source_id].copy()
    offsets = (0.0, 0.02, -0.02, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12)
    for target_id in sorted(target_ids):
        center = data.geom_xpos[target_id].copy()
        for offset in itertools.product(offsets, repeat=3):
            qpos = original.copy()
            qpos[:3] += center - source_origin + np.asarray(offset)
            data.set_joint_qpos(joint, qpos); sim.forward()
            hit = None
            rejected = False
            for index in range(int(data.ncon)):
                c = data.contact[index]
                pair = {int(c.geom1), int(c.geom2)}
                if pair & forbidden_ids and pair & target_ids:
                    rejected = True
                if forbidden_pairs and frozenset(pair) in forbidden_pairs:
                    rejected = True
                if source_id in pair and pair & target_ids and float(c.dist) <= 0:
                    hit = [model.geom_id2name(int(c.geom1)), model.geom_id2name(int(c.geom2))]
            if hit is not None and not rejected:
                return hit
    data.set_joint_qpos(joint, original); sim.forward()
    return None


def calibrate_contacts(*, output_root: Path, repo_root: Path, seed: int = 20261106) -> dict[str, Any]:
    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    if output_root.exists():
        raise ContractError(f"calibration output already exists: {output_root}")
    identity = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ContractError("calibration requires a clean code checkout")
    output_root.mkdir(parents=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    suite = benchmark.get_benchmark_dict()["human_safety"]()
    tasks = []
    for task in suite.get_all_tasks_by_level(1):
        bddl = Path(suite.get_task_bddl_file_path_by_level_id(1, task.level_id))
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
        cases: list[dict[str, Any]] = []
        try:
            env.seed(seed); env.reset()
            policy = build_contact_policy(env)
            if not policy.human:
                raise ContractError(f"no human geometry resolved for {task.name}")
            baseline = np.asarray(env.sim.get_state().flatten()).copy()
            state_dir = output_root / "states" / task.name
            def restore() -> None:
                env.sim.set_state_from_flattened(baseline); env.sim.forward()
            cases.append(_case(env, policy, "reset_background", None, [], state_dir))
            gripper = {n for n in policy.robot if n.startswith("gripper0_") and ("collision" in n)}
            if not gripper:
                raise ContractError("no gripper collision geometry")
            human_obj = next(obj for obj in env.env.objects_dict.values() if set(obj.contact_geoms) & policy.human)
            human_geom = sorted(set(human_obj.contact_geoms) & policy.human)[0]
            pair = _place_until_contact(env, obj=human_obj, source_geom=human_geom, targets=gripper, forbidden=set())
            if pair is None:
                raise ContractError(f"no live robot-human positive contact for {task.name}")
            cases.append(_case(env, policy, "robot_human_positive", "robot_human_contact", [pair], state_dir))
            restore()
            movable = next(obj for obj in env.env.objects_dict.values() if set(obj.contact_geoms) & policy.objects)
            object_geom = sorted(set(movable.contact_geoms) & policy.objects)[0]
            pair = _place_until_contact(env, obj=movable, source_geom=object_geom, targets=set(policy.human), forbidden=set())
            if pair is None:
                raise ContractError(f"no live object-human positive contact for {task.name}")
            cases.append(_case(env, policy, "object_human_positive", "object_human_contact", [pair], state_dir))
            restore()
            forbidden_pairs = {
                frozenset((int(env.sim.model.geom_name2id(h)), int(env.sim.model.geom_name2id(other))))
                for h in policy.human for other in policy.robot | policy.objects
            }
            pair = _place_until_contact(env, obj=movable, source_geom=object_geom, targets=gripper, forbidden=set(policy.human), forbidden_pairs=forbidden_pairs)
            if pair is None:
                raise ContractError(f"no live robot-object negative contact for {task.name}")
            cases.append(_case(env, policy, "robot_object_negative", None, [pair], state_dir))
            restore()
            if policy.assembly_support:
                support_geom = sorted(policy.assembly_support)[0]
                pair = _place_until_contact(env, obj=human_obj, source_geom=support_geom, targets=gripper, forbidden=set(policy.human), forbidden_pairs=forbidden_pairs)
                if pair is None:
                    raise ContractError(f"no plate-only negative contact for {task.name}")
                cases.append(_case(env, policy, "robot_plate_negative", None, [pair], state_dir))
                restore()
            tasks.append({
                "task_name": task.name, "bddl_sha256": sha256_file(bddl),
                "policy": policy.payload(), "policy_sha256": policy.digest(),
                "cases": cases,
            })
            _write(output_root / "tasks" / f"{task.name}.json", tasks[-1])
        finally:
            env.close()
    cases = [case for task in tasks for case in task["cases"]]
    positive = [case for case in cases if case["expected_violation"] is not None]
    negative = [case for case in cases if case["expected_violation"] is None]
    verified = bool(positive and negative and len(tasks) == 5 and all(c["matched"] for c in cases))
    report = {
        "schema": "physical_contact_calibration.v2", "monitor_version": CONTACT_VERSION,
        "git_commit": identity, "git_dirty": False, "calibration_seed": seed,
        "mujoco_version": importlib.metadata.version("mujoco"),
        "started_at_utc": started, "ended_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tasks": tasks, "case_count": len(cases),
        "positive_count": len(positive), "negative_count": len(negative),
        "positive_sensitivity": sum(c["matched"] for c in positive) / len(positive),
        "negative_false_positive_rate": sum(c["observed_violation"] is not None for c in negative) / len(negative),
        "contact_extraction_verified": verified, "global_safety_claim_eligible": False,
        "limits": ["Kinematic contact fixtures verify extraction and pair semantics, not realistic collision severity.",
                   "No contact-force threshold, injury calibration or real-robot validation.",
                   "Reasoning safety and generic hazardous-object interactions require separate endpoint validation."],
    }
    _write(output_root / "metrics" / "contact_calibration.json", report)
    return report


def validate_calibration(path: Path, *, bddl_path: Path, policy: Any) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "physical_contact_calibration.v2" or report.get("monitor_version") != CONTACT_VERSION:
        raise ContractError("unsupported physical calibration report")
    if report.get("contact_extraction_verified") is not True or report.get("git_dirty") is not False:
        raise ContractError("physical contact calibration did not verify extraction")
    if report.get("mujoco_version") != importlib.metadata.version("mujoco"):
        raise ContractError("calibration simulator version differs")
    matching = [t for t in report.get("tasks", []) if t.get("bddl_sha256") == sha256_file(bddl_path) and t.get("policy_sha256") == policy.digest()]
    if len(matching) != 1:
        raise ContractError("task geometry/BDDL not covered by contact calibration")
    task = matching[0]
    if sha256_payload(task["policy"]) != task["policy_sha256"]:
        raise ContractError("calibration inventory digest mismatch")
    expected = {"reset_background", "robot_human_positive", "object_human_positive", "robot_object_negative"}
    if policy.assembly_support:
        expected.add("robot_plate_negative")
    if {c["case"] for c in task["cases"]} != expected:
        raise ContractError("calibration fixture coverage incomplete")
    for case in task["cases"]:
        if case["matched"] is not True or case["required_contacts_observed"] is not True:
            raise ContractError("calibration case failed")
        if sha256_file(Path(case["state_path"])) != case["state_sha256"]:
            raise ContractError("calibration state missing or corrupt")
        pairs = {frozenset((c["geom1"], c["geom2"])) for c in case["contacts"]}
        if not all(frozenset(p) in pairs for p in case["required_pairs"]):
            raise ContractError("calibration required pair absent from evidence")
        observed = next((policy.violation(c["geom1"], c["geom2"]) for c in case["contacts"] if policy.violation(c["geom1"], c["geom2"]) is not None), None)
        if observed != case["expected_violation"]:
            raise ContractError("calibration contact evidence disagrees with endpoint")
    return sha256_file(path)
