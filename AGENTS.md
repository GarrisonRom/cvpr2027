# Workspace Instructions

This file is the project-wide source of truth for file ownership, experiment records, analysis outputs, plots and local/cloud collaboration. Follow it when adding or changing project files.

## Workspace boundaries

- Keep current paper sources and CVPR template files at the workspace root unless the user asks to reorganize them.
- Keep experiment code, configs, manifests, tests, environment locks and experiment documentation under safety_continuity/.
- Keep large datasets, model weights, checkpoints, caches and run outputs under the root runtime directories listed below.
- Do not move, rename, overwrite or regenerate existing paper files as a side effect of experiment setup.
- The root README.md currently contains inherited CVPR template notes. Preserve those notes until an intentional README migration is requested. The experiment project entry point is safety_continuity/README.md.

## Canonical directory plan

The tracked project tree is planned as follows. Add files when their interfaces or content are needed; do not create empty placeholder code files.

- safety_continuity/README.md: experiment project quick start and navigation.
- safety_continuity/pyproject.toml: package metadata, developer tools and command-line entry points when executable code is added.
- safety_continuity/configs/defaults.yaml: path-independent shared defaults.
- safety_continuity/configs/data/: dataset IDs, source revisions, splits, preprocessing and field maps.
- safety_continuity/configs/models/: model revision, processor, embodiment, camera and action-schema configuration.
- safety_continuity/configs/training/: optimizer, schedule, precision, batch, trainable modules and save cadence.
- safety_continuity/configs/interruption/: interruption location, missing-state intervention and recovery environment.
- safety_continuity/configs/hot_update/: publication timing, switch boundary, runtime-state policy and rollback behavior.
- safety_continuity/configs/evaluation/: benchmark tasks, seeds, episode count, instruction split and safety monitors.
- safety_continuity/configs/analysis/: estimands, pairing, exclusions, confidence intervals, equivalence margins and multiplicity rules.
- safety_continuity/configs/plotting/: validated summary selection, grouping, labels, units and output formats; never hand-enter results.
- safety_continuity/src/safety_continuity/common/: config resolution, run IDs, provenance, hashing and RNG initialization.
- safety_continuity/src/safety_continuity/data/: dataset adapters, validation, split resolution and record schemas.
- safety_continuity/src/safety_continuity/models/: model registry and embodiment/input/action adapters.
- safety_continuity/src/safety_continuity/checkpointing/: state inventory, save/restore, integrity checks and resume audit.
- safety_continuity/src/safety_continuity/training/: shared training instrumentation and callbacks.
- safety_continuity/src/safety_continuity/deployment/: policy serving, atomic weight publication and hot-update events.
- safety_continuity/src/safety_continuity/safety/: safety labels, monitors, physical costs, refusal and false-refusal metrics.
- safety_continuity/src/safety_continuity/evaluation/: benchmark runners and episode-level evaluation.
- safety_continuity/src/safety_continuity/probes/: activation capture, probe fitting, held-out evaluation and interventions.
- safety_continuity/src/safety_continuity/mitigation/: release gate, safety replay/anchoring and rollback methods.
- safety_continuity/src/safety_continuity/analysis/: validation, aggregation, paired inference, equivalence and sensitivity analysis.
- safety_continuity/src/safety_continuity/plotting/: reusable figure and table renderers.
- safety_continuity/src/safety_continuity/reporting/: machine-readable run reports and paper result exports.
- safety_continuity/scripts/preflight/: hardware, dependency, data, model and output-path checks.
- safety_continuity/scripts/train/: training launchers.
- safety_continuity/scripts/resume/: resume and checkpoint-ablation launchers.
- safety_continuity/scripts/hot_update/: deployment switch and rollback launchers.
- safety_continuity/scripts/evaluate/: benchmark evaluation launchers.
- safety_continuity/scripts/analyze/: validation, aggregation, statistics and probe analysis.
- safety_continuity/scripts/plot/: figure and table generation from validated summaries.
- safety_continuity/scripts/report/: run report and paper bundle assembly.
- safety_continuity/tests/unit/: metrics, labels, config, manifests, checkpoint inventory and table calculations.
- safety_continuity/tests/integration/: tiny train-save-restore-evaluate and policy-server update workflows.
- safety_continuity/tests/smoke/: short end-to-end checks with synthetic or smallest available data.
- safety_continuity/manifests/datasets/: source, revision, license, split definition and checksums.
- safety_continuity/manifests/models/: source, revision, license/access, processor, action schema and checksums.
- safety_continuity/manifests/experiments/: run matrix, factor definitions and frozen protocol identifiers.
- safety_continuity/environments/: pinned dependency specifications, separated when model stacks conflict.
- safety_continuity/docs/experiment_protocol.md: frozen hypotheses, factors, inclusion rules and decision gates.
- safety_continuity/docs/mvp_experiment.md: the minimal π0.5/LIBERO feasibility pilot, its controls, outputs and claim limits.
- safety_continuity/docs/metrics.md: operational metric definitions, units, labels and aggregation.
- safety_continuity/docs/data_dictionary.md: episode and per-step record schema.
- safety_continuity/docs/cloud_setup.md: instance setup and storage mounts.
- safety_continuity/docs/decisions.md: dated design decisions and rationale.
- safety_continuity/docs/file_map.md: maintained module ownership map when the implementation grows.
- safety_continuity/results/paper/tables/: reviewed CSV and TeX tables selected for the paper.
- safety_continuity/results/paper/figures/: reviewed PDF/SVG/PNG figures selected for the paper.
- safety_continuity/results/paper/result_index.yaml: mapping from each paper artifact to source runs and analysis commit.

Do not create duplicate sources of truth. YAML configs define settings, manifests identify immutable resources, source modules implement behavior, and run records store resolved settings and outcomes.

## Runtime-only payload layout

- data/raw/<dataset>/<revision>/: immutable source downloads or collected data.
- data/processed/<dataset>/<hash>/: deterministic conversions and derived payloads.
- data/splits/<split_id>/: split manifests; small manifests may be tracked, large split payloads may not.
- models/base/<vendor>/<model>/<revision>/: pretrained weights.
- models/finetuned/<model>/<run_id>/: inference-ready exports.
- models/adapters/<model>/<run_id>/: LoRA or other adapter weights.
- checkpoints/training/<run_id>/<step>/: complete training-state checkpoint.
- checkpoints/release/<run_id>/: candidate release bundle.
- checkpoints/hot_update/<run_id>/: old/new versions and switch fixtures.
- runs/<run_id>/metadata/: resolved config, provenance, environment and hashes.
- runs/<run_id>/logs/: framework and application logs.
- runs/<run_id>/events/: checkpoint, interruption, resume and hot-update event JSONL.
- runs/<run_id>/episodes/: per-episode records and optional step traces.
- runs/<run_id>/metrics/: validated machine-readable summaries.
- runs/<run_id>/probes/: activations and probe outputs.
- runs/<run_id>/figures/: exploratory and intermediate figures.
- cache/: Hugging Face, Torch and dataset caches.

Runtime payloads are ignored by Git except .gitkeep markers and deliberately tracked small manifests. Cloud storage may be mounted at these runtime roots, but never over the tracked safety_continuity/ source tree.

## File responsibilities and naming

### Configs

- YAML configs are the human-editable source of settings. Compose reusable configs by ID; save the fully resolved config and SHA-256 in every run.
- Data configs define dataset revision, split manifest, preprocessing and field mapping. They contain no credentials or machine-specific absolute paths.
- Model configs define model/processor revision, input camera mapping, embodiment, action dimensions/chunking, normalization source and supported inference settings.
- Training configs define trainable modules, optimizer, scheduler, precision, effective batch size, seed, total optimizer steps and checkpoint cadence.
- Interruption configs define one explicit intervention, such as optimizer reset or data-cursor reset, and list all other state that must remain fixed.
- Hot-update configs define old/new model versions, source run, switch boundary, action queue handling, runtime-state handling, publication delay and rollback behavior.
- Evaluation configs define benchmark version, task list, initial-state seeds, episode count, instruction split, safety monitor and capability metrics.
- Analysis configs define primary estimands, pairing unit, exclusion rules, confidence intervals, equivalence margin and correction for multiple comparisons.
- Plotting configs select validated summaries and define labels, groups, units, uncertainty display and output formats. Results never belong as literals in plotting config.
- Use descriptive lowercase filenames, for example libero_safety.yaml, pi05.yaml, resume_full.yaml, resume_optimizer_reset.yaml, switch_episode_boundary.yaml and primary_analysis.yaml. Version suffixes are for meaningful protocol revisions, not routine edits.

### Python modules

- common owns shared infrastructure, not model behavior.
- data owns input validation, adapters, immutable split resolution and schemas; raw files are never edited in place.
- models owns explicit per-model adapters. Each adapter declares observation keys, camera mapping, embodiment, action dimensions, chunk semantics, normalization and checkpoint format.
- checkpointing owns state inventories and restore semantics. APIs and logs distinguish full-state resume, weights-only initialization and inference export.
- training owns reusable instrumentation; native training loops remain behind documented adapters.
- deployment owns serving and hot-update transactions, including requested/actual switch times, active version hashes, queued actions, state handling, latency and rollback.
- safety owns operational definitions and code for physical violations, harmful instruction outcomes, safe alternatives and false refusal.
- evaluation owns benchmark protocols and episode records; adapters cannot silently change the benchmark's success or safety definitions.
- probes owns activation extraction, data separation, frozen-probe evaluation and causal interventions.
- mitigation owns release gates and safety-preservation methods. Gate thresholds use calibration data separate from final tests.
- analysis owns validated aggregation and inference, not simulator control.
- plotting owns reproducible figures/tables from validated summaries; figures are never hand-edited to change values.
- reporting owns run reports and paper export bundles with source run and version links.

### Scripts, tests and notebooks

- Scripts are thin command-line entry points. They validate arguments and inputs, call package functions and choose output roots; they do not duplicate experiment logic.
- Every script provides help and fails early on incompatible model/data/checkpoint inputs.
- Unit tests cover metric calculations, labels, config validation, checkpoint inventory and table calculations.
- Integration tests cover a tiny train-save-restore-evaluate path and policy-server hot update with a fake or small model.
- Smoke tests verify environment and interfaces with synthetic data or the smallest licensed subset; smoke results are not paper evidence.
- Notebooks are exploratory only. Any paper claim or figure must be reproducible by tracked code and config.

### README responsibilities

- Root README.md currently preserves inherited CVPR template documentation. Do not silently replace it. On a deliberate migration, make it the repository landing page and preserve template attribution/instructions under safety_continuity/docs/cvpr_template_notes.md.
- safety_continuity/README.md is the experiment quick start: research scope, supported workflows, setup, directory map, preflight/train/evaluate/analyze examples, storage roots and links to protocol/metrics/schema documentation.
- Experiment protocol docs freeze design choices; AGENTS.md governs file ownership and workflow. Neither document contains changing result numbers.
- When file responsibility changes, update AGENTS.md and safety_continuity/README.md in the same change.

## Experiment identity and output schema

- Use immutable run IDs: YYYYMMDD_<branch>_<model>_<condition>_sNN. IDs use lowercase ASCII and no spaces.
- **Current primary MVP training seed set:** `42`, `43`, `44`. New training,
  capability and self-built safety runs must use these three seeds unless a
  protocol revision explicitly says otherwise. Existing `11`, `23`, `37` runs
  are historical immutable artifacts and must not be renamed or overwritten.
- Keep training, evaluation-initial-state and evaluator RNG seeds separate in
  every run record; changing the training seed set does not silently change
  the paired environment seed manifest.
- Every run contains metadata/resolved_config.yaml, metadata/provenance.json and, where applicable, metadata/checkpoint_manifest.json, plus logs and a machine-readable summary.
- provenance.json records Git commit and dirty status, config hash, model/processor revision, dataset manifest/hash, split ID, seed, GPU name/visible memory, driver, Python/framework/CUDA packages, simulator version and start/end times.
- checkpoint_manifest.json lists included state items, shard paths, sizes, hashes, global step, save completion and restore compatibility. Missing state is explicit.
- Training runs store metadata/batch_manifest.json with the immutable split path/hash, dataset revision, seed and optimizer horizon; events/batch_consumption.jsonl records every consumed sample in order.
- Append events to checkpoint_events.jsonl, interruption_events.jsonl, resume_events.jsonl and hot_update_events.jsonl. Each record contains run ID and UTC timestamp.
- Store episode data as episodes/episodes.jsonl or Parquet, one record per episode. Stable fields include run/model/condition, benchmark/suite, task, scene, instruction ID/type, training/evaluation/initial-state seeds, model version, success, safe/unsafe outcome, violation type, safety cost, exposure duration, completion time and termination reason.
- Store step-level traces separately under episodes/step_traces/ as compressed Parquet/JSONL. Include timestep, action, monitor margins, active policy version and simulator-state reference.
- metrics/summary.json is the canonical run aggregate. metrics/episode_metrics.csv is the inspectable flat export. Include sample counts, exclusions, interval estimates, units and metric definitions.
- Never overwrite a completed run. A rerun receives a new ID and records any parent run.

## Analysis, plots and tables

- Required pipeline: validate schemas/hashes -> validate episode labels -> aggregate by run/seed/task/condition -> paired inference and capability-equivalence analysis -> probe analysis -> plots/tables -> reviewed paper export.
- Analysis consumes explicit run IDs or a tracked run manifest; it never scans arbitrary output directories and guesses which runs to include.
- Every aggregate records source run IDs, analysis config/hash, code commit, exclusions and creation date.
- Save intermediate summaries and plots under runs/<analysis_id>/metrics/ and runs/<analysis_id>/figures/.
- Use descriptive names such as resume_safety_risk_by_condition.csv, capability_equivalence_by_model.csv, hot_update_risk_over_time.csv, probe_layerwise_auc.csv, fig_resume_safety_risk.pdf and fig_hot_update_timeline.pdf.
- Plot sources remain tracked under src/safety_continuity/plotting/ and configs/plotting/. Intermediate outputs remain in ignored runs/.
- Only reviewed artifacts are copied to results/paper/tables/ and results/paper/figures/. Keep a machine-readable source plus TeX for tables; keep editable figure source/config plus PDF/SVG and a raster preview when useful.
- result_index.yaml maps each paper artifact to source run IDs, analysis config/hash and Git commit.
- Never type statistical values into paper tables by hand when they can be generated from canonical summaries.
- Nonsignificance is not equivalence. Capability preservation requires the configured equivalence test and margin; sample counts and intervals remain visible.
- Captions and table notes define metric, unit, statistical unit, interval method and tested conditions in tracked source/config.

## Local-to-cloud consistency

- The local Git checkout is the canonical source for experiment code, configs, scripts, manifests and analysis.
- Cloud jobs run from a clean checkout of the exact commit recorded in provenance.json.
- Do not make uncommitted code/config edits directly on cloud. Bring changes to local, review, commit, then sync a new revision.
- Use identical relative layout locally and in the cloud. Configure roots through environment variables or ignored local .env; never hard-code machine-specific absolute paths.
- Recommended roots: SC_DATA_ROOT, SC_SPLIT_ROOT, SC_MODEL_ROOT, SC_CHECKPOINT_ROOT, SC_RUN_ROOT and SC_CACHE_ROOT. Defaults point to the workspace runtime directories; SC_SPLIT_ROOT defaults to data/splits.
- Keep the code checkout on instance storage and mount large storage at data/model/checkpoint/run/cache roots. Never mount a data volume over code.
- Before long runs, execute preflight and verify free disk, visible GPU memory, framework GPU access, model load, data manifest and output write access.

## Reproducibility and resources

- Keep raw datasets immutable. Track revision, license, split definition, source and checksum under manifests/datasets/.
- Track model revision, license/access requirements, processor/action schema, weight file list and checksum under manifests/models/. Weights never go into Git.
- Record checkpoint state when applicable: model/adapter, optimizer structure, optimizer moments, scheduler/global step, Python/NumPy/CPU/GPU RNG, sampler/data cursor, gradient accumulation, AMP scaler, EMA, normalization and distributed layout.
- Write checkpoints to a temporary path, verify shards and checksums, then publish atomically when supported.
- A weights-only checkpoint is named and reported as weights-only, never full training resume.
- Record model, environment, data-order and evaluation initial-state seeds separately.
- Do not download large datasets or model weights unless the user requests it.

## Git, credentials and generated files

- Git tracks source, small configs/manifests, docs, tests and reviewed paper artifacts. It does not track raw/processed data payloads, model weights, adapters, checkpoints, caches, rollouts, activations, logs or exploratory figures.
- Keep .gitkeep markers for runtime directories. Do not weaken ignore rules to add large payloads.
- Never commit access tokens, SSH credentials, private keys, .env files, signed URLs or machine credentials.
- Do not upload data/model artifacts to external services unless explicitly requested.
- LaTeX build products stay out of Git unless the user asks to publish a rendered artifact.

## Environment and interface changes

- Pin Python, framework, CUDA-facing packages, simulator, model code and evaluation dependencies under environments/.
- Keep incompatible model stacks isolated; do not force every model into one dependency environment.
- Before changing shared package versions, data schemas, metric definitions, action adapters or config interfaces, inspect affected code and document compatibility impact.
