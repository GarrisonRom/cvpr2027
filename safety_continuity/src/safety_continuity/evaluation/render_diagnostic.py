"""Open-loop replay to separate cached camera, rendering and policy differences."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any

from ..common.config import sha256_file
from ..common.errors import ContractError
from .continuation_diagnostic import _observation, _task_states, now, write_json


def array_difference(left: Any, right: Any) -> dict[str, Any]:
    import numpy as np
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype or not a.size:
        raise ContractError("compared arrays require identical nonempty shape and dtype")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ContractError("compared arrays must be finite")
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    changed = a != b
    result = {"exact_equal": bool(not changed.any()), "changed_values": int(changed.sum()),
              "max_abs_delta": float(difference.max()), "mean_abs_delta": float(difference.mean())}
    if a.ndim == 3 and a.shape[-1] == 3:
        result["changed_pixels"] = int(changed.any(axis=-1).sum())
    return result


def _snapshot(env: Any, obs: dict[str, Any], task: dict[str, Any], seed: int, step: int,
              output: Path, render_repeats: int) -> dict[str, Any]:
    import numpy as np
    sim = env.sim
    arrays = {"sim_state": sim.get_state().flatten(),
              "cached_agent": obs["agentview_image"].copy(),
              "cached_wrist": obs["robot0_eye_in_hand_image"].copy()}
    element = _observation(obs, task["task_name"], seed, step // 5)
    arrays.update(processed_agent=element["observation/image"],
                  processed_wrist=element["observation/wrist_image"], proprio=element["observation/state"])
    for owner, names in (("data", ("qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat", "xpos", "xmat",
                                   "cam_xpos", "cam_xmat", "light_xpos", "light_xdir")),
                         ("model", ("geom_rgba", "site_rgba", "cam_pos", "cam_quat", "light_pos", "light_dir"))):
        obj = getattr(sim, owner)
        for name in names:
            if hasattr(obj, name):
                value = np.asarray(getattr(obj, name))
                if value.size:
                    arrays[f"{owner}_{name}"] = value.copy()
    sensors = {name: {key: getattr(value, key) for key in
                     ("_sampled", "_time_since_last_sample", "_current_delay", "_sampling_timestep")}
               for name, value in env.env._observables.items() if name.endswith("_image")}
    held = {}
    for label, camera in (("agent", "agentview"), ("wrist", "robot0_eye_in_hand")):
        images = [sim.render(camera_name=camera, width=256, height=256).copy()
                  for _ in range(render_repeats)]
        arrays[f"direct_{label}"] = images[0]
        held[label] = [array_difference(images[0], image) for image in images[1:]]
        # Retain every image so a summary can be independently audited.
        for index, image in enumerate(images[1:], 1):
            arrays[f"held_{label}_{index}"] = image
    after = sim.get_state().flatten()
    if not np.array_equal(arrays["sim_state"], after):
        raise ContractError("rendering changed saved simulator state")
    np.savez_compressed(output, **arrays)
    return {"step": step, "file": str(output), "sha256": sha256_file(output), "sensors": sensors,
            "held_render_differences": held,
            "sim_state_sha256": hashlib.sha256(np.ascontiguousarray(after).tobytes()).hexdigest()}


def configure_replay_renderer(env: Any, profile: str) -> dict[str, Any]:
    """Apply an explicit visual intervention after each hard reset, before state load."""
    if profile not in ("native", "egl_no_msaa_v1"):
        raise ContractError("unsupported replay renderer profile")
    context = env.sim._render_context_offscreen
    requested_before = int(env.sim.model.vis.quality.offsamples)
    actual_before = int(context.con.offSamples)
    if profile == "egl_no_msaa_v1":
        # The inspected vendored binding uses this exact method to recreate an
        # offscreen buffer after resize. Rebuild without advancing physics.
        env.sim.model.vis.quality.offsamples = 0
        context.con.free()
        context._set_mujoco_context_and_buffers()
        if int(context.con.offSamples) != 0:
            raise ContractError("renderer did not disable multisampling")
    return {"profile": profile, "requested_samples_before": requested_before,
            "actual_samples_before": actual_before, "actual_samples_after": int(context.con.offSamples)}


def _render_ablations(env: Any, output: Path) -> dict[str, Any]:
    """Same held scene, one rendering setting changed at a time; diagnostic only."""
    import mujoco
    import numpy as np

    sim = env.sim
    context = sim._render_context_offscreen
    original_flags = context.scn.flags.copy()
    original_samples = sim.model.vis.quality.offsamples
    original_state = sim.get_state().flatten().copy()
    result = {"original_offsamples": int(original_samples), "profiles": {}}
    arrays = {}
    profiles = ("native_before", "no_shadows", "no_reflections", "no_shadows_or_reflections",
                "native_recreated_context", "no_multisampling", "native_after")
    try:
        for profile in profiles:
            context.scn.flags[:] = original_flags
            if profile in ("no_shadows", "no_shadows_or_reflections"):
                context.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            if profile in ("no_reflections", "no_shadows_or_reflections"):
                context.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            if profile in ("native_recreated_context", "no_multisampling", "native_after"):
                sim.model.vis.quality.offsamples = 0 if profile == "no_multisampling" else original_samples
                context.con.free()
                context._set_mujoco_context_and_buffers()
            images = [sim.render(camera_name="robot0_eye_in_hand", width=256, height=256).copy()
                      for _ in range(64)]
            differences = [array_difference(images[0], image) for image in images[1:]]
            result["profiles"][profile] = {"varying_pairs": sum(not d["exact_equal"] for d in differences),
                                           "pairs": len(differences), "differences": differences}
            arrays[profile] = np.stack(images)
    finally:
        context.scn.flags[:] = original_flags
        sim.model.vis.quality.offsamples = original_samples
        context.con.free()
        context._set_mujoco_context_and_buffers()
    if not np.array_equal(original_state, sim.get_state().flatten()):
        raise ContractError("render ablation changed simulator state")
    file = output/"render_ablations.npz"
    np.savez_compressed(file, **arrays)
    result.update(file=str(file), sha256=sha256_file(file), physics_state_unchanged=True)
    write_json(output/"render_ablations.json", result)
    return result


def run(source_run: Path, output: Path, *, steps: int, repeats: int, render_repeats: int,
        render_ablations: bool = False, render_profile: str = "native") -> None:
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv

    if not 5 <= steps <= 400 or not 2 <= repeats <= 10 or not 2 <= render_repeats <= 64:
        raise ContractError("requires steps 5..400 and repeats 2..10, render repeats 2..64")
    if render_profile not in ("native", "egl_no_msaa_v1") or (render_ablations and render_profile != "native"):
        raise ContractError("render ablations require native baseline; unsupported profile")
    request_file = source_run / "metadata/requests/original_primary_reasoning_safety.json"
    request = json.loads(request_file.read_text(encoding="utf-8"))
    tasks, _ = _task_states(request)
    task = tasks[0]
    trace_file = source_run / "evaluations/original/primary/reasoning_safety/steps.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = {}
    with trace_file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(row["episode_key"], []).append(row)
    for episode_index in range(request["episodes_per_task"]):
        key = f'reasoning_safety:{task["task_id"]}:{episode_index}'
        if len(grouped.get(key, [])) < steps:
            raise ContractError("insufficient original open-loop actions")
        for index, row in enumerate(grouped[key][:steps]):
            if row["step"] != index or np.asarray(row["action"]).shape != (7,) or not np.isfinite(row["action"]).all():
                raise ContractError("invalid saved action sequence")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": "camera_replay_diagnostic.v1", "created_at_utc": now(),
                "source_run": str(source_run), "source_request_sha256": sha256_file(request_file),
                "source_trace_sha256": sha256_file(trace_file), "steps": steps, "repeats": repeats,
                "render_repeats": render_repeats, "policy_inference": False, "new_training": False,
                "render_ablations": render_ablations,
                "render_profile": render_profile,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "status": "running"}
    write_json(output/"metadata.json", metadata)
    records = []
    try:
        for repeat in range(repeats):
            env = OffScreenRenderEnv(bddl_file_name=task["bddl_path"], camera_heights=256, camera_widths=256)
            try:
                for episode_index in range(request["episodes_per_task"]):
                    seed = request["config"]["evaluation_seed"] + episode_index
                    np.random.seed(seed)
                    random.seed(seed)
                    env.seed(seed)
                    env.reset()
                    renderer = configure_replay_renderer(env, render_profile)
                    obs = env.set_init_state(task["states"][episode_index])
                    for _ in range(request["config"]["settling_steps"]):
                        obs, _, _, _ = env.step([0.0]*6+[-1.0])
                    key = f'reasoning_safety:{task["task_id"]}:{episode_index}'
                    rows = grouped[key]
                    frames = []
                    for index in range(steps + 1):
                        # Query 1 at step 5 is where the failed run first diverged.
                        if index <= 6 or index % 5 == 0:
                            frame = _snapshot(env, obs, task, seed, index,
                                              output/f"r{repeat}_e{episode_index}_t{index}.npz", render_repeats)
                            frame["reference_sim_state_match"] = (None if index == 0 else
                                frame["sim_state_sha256"] == rows[index-1]["sim_state_sha256"])
                            frames.append(frame)
                        if render_ablations and repeat == 0 and episode_index == 0 and index == 5:
                            _render_ablations(env, output)
                        if index < steps:
                            obs, _, _, _ = env.step(rows[index]["action"])
                    records.append({"repeat": repeat, "episode": episode_index, "renderer": renderer, "frames": frames})
                    write_json(output/"records.json", records)
                    print(json.dumps({"repeat": repeat, "episode": episode_index, "complete": True}), flush=True)
            finally:
                env.close()
        comparisons = []
        for record in records:
            if record["repeat"] == 0:
                continue
            reference = next(r for r in records if r["repeat"] == 0 and r["episode"] == record["episode"])
            for left, right in zip(reference["frames"], record["frames"], strict=True):
                with np.load(left["file"], allow_pickle=False) as a, np.load(right["file"], allow_pickle=False) as b:
                    differences = {}
                    for key in a.files:
                        difference = array_difference(a[key], b[key])
                        if not difference["exact_equal"]:
                            differences[key] = difference
                comparisons.append({"repeat": record["repeat"], "episode": record["episode"], "step": right["step"],
                                    "differences": differences, "sensors_equal": left["sensors"] == right["sensors"]})
        write_json(output/"comparisons.json", comparisons)
        metadata.update(status="complete", finished_at_utc=now(),
                        differing_replay_frames=sum(bool(row["differences"]) for row in comparisons),
                        varying_held_render_pairs=sum(not d["exact_equal"] for r in records for f in r["frames"]
                                                      for ds in f["held_render_differences"].values() for d in ds))
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output/"metadata.json", metadata)
