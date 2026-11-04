# Safety Continuity Experiments

This folder owns experiment software and documentation for measuring safety changes caused by training interruption/resumption and deployment-time weight updates.

## Start here

- ../AGENTS.md is the project-wide contract for file ownership, naming, run records, analysis outputs, plots/tables, Git rules and local/cloud synchronization.
- ../实验设置方案.md is the current research design draft. It is a proposal, not a frozen protocol or a source of experimental results.
- docs/mvp_experiment.md is the proposed π0.5/LIBERO pilot for checking whether the measurement and attribution workflow is feasible.
- docs/safety_measurement_protocol.md is the single source of truth for official versus self-built safety labels, state generation, monitor rules, calibration and interpretation limits.
- docs/experiment_protocol.md defines protocol states, promotion gates and the frozen protocol manifest.
- docs/metrics.md defines aggregation, pairing, denominators and analysis gates; safety event definitions remain in safety_measurement_protocol.md.
- docs/data_dictionary.md defines episode and per-step record schemas.
- docs/cloud_setup.md records the current AutoDL payload mapping and checksum locations.
- docs/interface_contract.md freezes the boundaries between configs, adapters,
  checkpoints, evaluation records and analysis.
- manifests/models/open_policy_inventory.yaml catalogs downloaded, gated and unavailable model candidates.
- manifests/artifacts/additional_models.tsv pins every extra model/encoder file revision, path and expected size.

## Planned project tree

- configs/: version-controlled data, model, training, interruption, hot-update, evaluation, analysis and plotting settings.
- src/safety_continuity/: reusable Python implementation, with separate modules for data, model adapters, checkpointing, training, deployment, safety metrics, evaluation, probes, mitigation, analysis, plotting and reporting.
- scripts/: thin command-line entry points for preflight, train, resume, hot update, evaluation, analysis, plotting and reports.
- tests/: unit, integration and smoke tests.
- manifests/: small dataset/model revisions, licenses, split definitions, experiment matrices and checksums.
- environments/: pinned dependency specifications, separated when model stacks conflict.
- docs/: experiment protocol, metrics, data schemas, cloud setup and design decisions.
- results/paper/: reviewed figures/tables mapped to source runs by result_index.yaml.

Large payloads live in sibling workspace roots data/, models/, checkpoints/, runs/ and cache/. Their contents are excluded from Git.

## Experiment workflow

1. Validate code, environment, dataset manifest, model revision, storage and visible GPU with preflight.
2. Resolve one immutable config and create a unique run ID.
3. Record code commit, model/data revisions, config hash, seeds, GPU/driver and software environment.
4. Run training or evaluation and append checkpoint, interruption, resume and hot-update events.
5. Validate episode records and hashes before analysis.
6. Aggregate by run, seed, task and condition; run configured paired statistics and capability-equivalence analysis.
7. Generate plots and tables from validated machine-readable summaries.
8. Copy only reviewed paper artifacts into results/paper/ and update result_index.yaml with their source runs and analysis commit.

The canonical file map, output schema, naming rules and artifact requirements are in ../AGENTS.md.

## Local preflight

Install the contract layer from the repository root:

    python -m pip install -e safety_continuity

Run the configuration check from the repository root:

    python safety_continuity/scripts/preflight/check_contracts.py

Use `--strict` after adapter camera/action fields, optimizer/batch
settings and the update split manifest have all been resolved:

    python -m safety_continuity.cli --workspace-root . --strict

The check validates YAML schema versions, tracked references and root-safe
runtime paths. Warnings for unresolved model fields or a not-yet-created split
manifest are expected while the GPU-specific adapter inspection is pending. An
experiment launch must not proceed while the preflight reports an error.

Validate a saved checkpoint inventory against one configured resume condition:

    python safety_continuity/scripts/resume/validate_restore_plan.py --checkpoint-manifest runs/20261104_mvp_pi05_a2_s42/metadata/checkpoint_manifest.json --checkpoint-root checkpoints/training/20261104_mvp_pi05_a2_s42/1000 --condition A2

State file paths are resolved relative to `--checkpoint-root`, which points to
the actual checkpoint payload directory (not the run metadata directory).

## Episode validation and MVP analysis

Validate completed runs by listing each run id explicitly:

    python safety_continuity/scripts/analyze/validate_episodes.py --run-id 20261104_mvp_pi05_a1_s42

Generate the paired fixed-A2-versus-A1 analysis by listing all three-seed runs
for A0, A1 and fixed A2 (each run should contain the configured LIBERO-10 and safety
benchmark episodes). A0 and A1 are required for the resume-control gate:

    python safety_continuity/scripts/analyze/analyze_mvp.py --run-id 20261104_mvp_pi05_a0_s42 --run-id 20261104_mvp_pi05_a0_s43 --run-id 20261104_mvp_pi05_a0_s44 --run-id 20261104_mvp_pi05_a1_s42 --run-id 20261104_mvp_pi05_a1_s43 --run-id 20261104_mvp_pi05_a1_s44 --run-id 20261104_mvp_pi05_a2fix_s42 --run-id 20261104_mvp_pi05_a2fix_s43 --run-id 20261104_mvp_pi05_a2fix_s44 --analysis-id 20261104_analysis_pi05_a2fix_s00

The analysis writes a new, immutable directory under `runs/<analysis_id>/` with
source hashes, run summaries, seed-level paired effects, and pilot bootstrap
intervals. It also verifies the frozen and actually consumed batch sequence for
A0/A1 and fixed-A2/A1, evaluates the configured A0/A1 safety/capability control, and
checks the primary LIBERO-10 capability interval against its equivalence margin.
The primary signal is supported only when the safety threshold, batch audits,
capability gate and A0/A1 control all pass. It checks completed-run provenance
and the resolved-config hash, and refuses to reuse an existing analysis id. The
pilot screens are descriptive and are not formal equivalence tests.

## OpenPI MVP runner

The executable GPU runner is `scripts/train/openpi_mvp.py`. It selects the first
available demonstration episode for each official LIBERO-10 task, freezes a
2,000-sample schedule, forces `shuffle=False`, records every consumed sample,
and supports `PRE`, `A0`, `A1`, `A2` and `A3` stages. The released safety
checkpoint is trained through its native `action_out_proj` subset because the
official LoRA configuration adds modules absent from the checkpoint.

The runner expects OpenPI in an external checkout and uses the runtime roots
from the shell. A typical GPU invocation is:

    PYTHONPATH=/root/cvpr2027/safety_continuity/src /root/openpi-venv-compatible/bin/python safety_continuity/scripts/train/openpi_mvp.py --openpi-root /root/openpi --payload-root /root/autodl-tmp/cvpr2027-payloads --run-id 20261104_mvp_pi05_pre_s42 --condition PRE --seed 42 --total-steps 1000 --schedule-count 1000 --schedule-path /root/autodl-tmp/cvpr2027-payloads/data/splits/mvp_update_samples_s42.jsonl --selected-episodes 0 1 2 6 7 8 10 12 14 27 --checkpoint-name pre_s42

The interruption branches reuse the resulting checkpoint directory with
`--resume` or `--source-checkpoint-dir`; their metadata and batch-consumption
records are required inputs to analysis.

## Self-built safety MVP runner

`scripts/evaluate/run_self_built_mvp.py` is the single entry point for the
adapted LIBERO-Safety protocol. It keeps generated simulator states and
rule-based safety labels separate from official benchmark labels.

Run the CPU-only contract smoke:

    python safety_continuity/scripts/evaluate/run_self_built_mvp.py --mode smoke --output-root runs/self_built_smoke

Run all A0/A1/A2/A3 conditions for one seed and suite on a GPU instance:

    python safety_continuity/scripts/evaluate/run_self_built_mvp.py --mode full --suite human_safety --condition all --seed 42 --payload-root /root/autodl-tmp/cvpr2027-payloads --checkpoint-root /root/autodl-tmp/cvpr2027-payloads/checkpoints --openpi-root /root/openpi --output-root /root/autodl-tmp/cvpr2027-payloads/runs/self_built_human_s42

Full mode requires LIBERO-Safety, MuJoCo/EGL and the OpenPI websocket
dependencies in the selected environment. Generated labels use
`self_built_rules.v1`; they are valid for paired adapted-protocol analysis and
must not be reported as official LIBERO-Safety unsafe rates.

## One-command unattended MVP

Use `scripts/evaluate/run_mvp_unattended.py` when the complete seed matrix
should run without interactive supervision. It runs training, capability
evaluation and both self-built safety suites serially. Each stage has its own
log; an existing run/output is never overwritten. The final gate is
`needs_review` by default because self-built monitor calibration must be
explicitly supplied before a safety claim is allowed.

```bash
nohup python safety_continuity/scripts/evaluate/run_mvp_unattended.py \
  --mode all \
  --payload-root /root/autodl-tmp/cvpr2027-payloads \
  --openpi-root /root/openpi \
  --libero-root /root/LIBERO-classic \
  --libero-config /root/libero-classic-config \
  > /root/autodl-tmp/cvpr2027-payloads/runs/unattended_mvp.log 2>&1 &
```

The script reports possible A2-versus-A1 risk differences but never edits or
"repairs" a checkpoint automatically. A result requiring intervention is
written to `unattended_mvp_gate_<date>.json`.

## AutoDL runtime layout

The current GPU instance keeps one clean root at
`/root/autodl-tmp/cvpr2027_RoBo`:

- `cvpr2027/`: clean Git checkout of this repository; run `git rev-parse HEAD` before a job.
- `openpi/`: pinned external OpenPI checkout and its `.venv`.
- `lerobot-<revision>/`: pinned LeRobot source used by OpenPI's data loader.
- `LIBERO-Safety/` and `LIBERO-classic/`: simulator/evaluation sources; their archives are removed after extraction.
- `cvpr2027-payloads/`: runtime-only data, models, checkpoints, caches, logs and run artifacts.
- `libero-config/` and `libero-classic-config/`: machine-local simulator path maps.

The root must not contain Git bundles or source archives. Setup logs are kept in
`cvpr2027-payloads/logs/setup/`; one-off evaluator scripts are kept under the
final analysis run's `metadata/legacy_scripts/` when needed for audit. The
runtime payload root is intentionally outside the Git checkout and is never
committed or mounted over source code.

## Model adapters

Each supported model needs a manifest and adapter that explicitly define observation fields, camera mapping, embodiment, action shape, chunk length, normalization, supported training modes, checkpoint-state coverage and evaluation support. Do not claim full resume support until save/restore state and post-resume data sequence have been audited.

## Local and cloud use

The local Git checkout is the source of truth. Cloud jobs run from a clean checkout of the recorded commit and mount storage at the same runtime roots. Cloud-side code/config changes must be brought back, reviewed and committed locally before the next run.
