# Safety Continuity Experiments

This folder owns experiment software and documentation for measuring safety changes caused by training interruption/resumption and deployment-time weight updates.

## Start here

- ../AGENTS.md is the project-wide contract for file ownership, naming, run records, analysis outputs, plots/tables, Git rules and local/cloud synchronization.
- ../实验设置方案.md is the current research design draft. It is a proposal, not a frozen protocol or a source of experimental results.
- docs/mvp_experiment.md is the proposed π0.5/LIBERO pilot for checking whether the measurement and attribution workflow is feasible.
- docs/experiment_protocol.md will hold the frozen protocol for each study phase, including hypotheses, factors, inclusion rules, metric definitions and decision gates.
- docs/metrics.md will define operational metrics, units, labels and aggregation.
- docs/data_dictionary.md will define episode and per-step record schemas.
- docs/cloud_setup.md records the current AutoDL payload mapping and checksum locations.
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

## Model adapters

Each supported model needs a manifest and adapter that explicitly define observation fields, camera mapping, embodiment, action shape, chunk length, normalization, supported training modes, checkpoint-state coverage and evaluation support. Do not claim full resume support until save/restore state and post-resume data sequence have been audited.

## Local and cloud use

The local Git checkout is the source of truth. Cloud jobs run from a clean checkout of the recorded commit and mount storage at the same runtime roots. Cloud-side code/config changes must be brought back, reviewed and committed locally before the next run.
