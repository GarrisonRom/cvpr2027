# Interface Contract

This document freezes the interfaces that must stay stable when a model, dataset,
GPU instance or simulator implementation changes. The contract version is
`safety_continuity.v1`.

## Source-of-truth order

1. YAML configs select a protocol and run settings.
2. Resource manifests identify immutable dataset/model revisions and checksums.
3. The model adapter resolves observation and action details from the selected
   checkpoint and benchmark.
4. A run stores the fully resolved config, typed provenance and state inventory.
5. JSONL/Parquet records are consumed by analysis and plotting without reading
   simulator internals.

No module may infer a missing value from a neighboring config or from a directory
name. Unresolved values are a preflight error before training or evaluation.

`RunContext` is the only owner of run output paths. It derives the fixed
`metadata`, `logs`, `events`, `episodes`, `metrics`, `probes` and `figures`
subdirectories from `SC_RUN_ROOT` and the immutable run id. Scripts must receive
those paths from the context instead of concatenating their own output names.

## Lifecycle

```text
YAML + manifests
       |
       v
resolved RunContext ----> frozen UpdateSampleRef sequence
       |
       +--> ModelAdapter --> ActionChunk --> PolicyServer
       |
       +--> CheckpointManifest + RestorePlan
       |
       +--> EpisodeRecord + StepTraceRecord
                         |
                         v
              validated summaries -> figures/tables
```

## Model boundary

`ModelAdapter` is the only package boundary allowed to know the native OpenPI,
PyTorch or processor API. It must expose:

- `ModelSpec`: model id, adapter id, checkpoint/processor revisions, embodiment,
  observation schema and action schema;
- `preprocess(observation)`: benchmark observation to native model input;
- `predict(model_input)`: native output to a validated `ActionChunk`;
- `save_inference_export(destination)`: an inference artifact with a hash and
  enough metadata to reconstruct the same adapter.

`ActionChunk` always contains `model_version`, `chunk_id`, a rectangular
`[chunk_length, action_dimension]` value matrix and a normalization id. A policy
server may not accept a raw tensor or silently apply a different normalization.

The MVP model config intentionally leaves `action.dimension`,
`action.chunk_length`, camera names and native processor keys unresolved. The
π0.5 adapter must fill and record them during preflight; a run cannot start while
they are null.

## Checkpoint boundary

`CheckpointManifest` must list every state key, including states that are absent:

`model`, `optimizer`, `optimizer_moments`, `optimizer_step`,
`scheduler_and_global_step`, `rng`, `batch_cursor`, `normalization`,
`gradient_accumulation`, `amp_scaler`, `ema`, and `distributed_layout`.

Each included state lists one or more files, each with a POSIX-style path
relative to the checkpoint payload directory, byte size and SHA-256. This
represents flat checkpoints and native sharded/OCDBT directories without
assuming that a state is one file. Restore preflight receives the payload root
explicitly and verifies every listed file before reporting compatibility.

`RestorePlan` declares which states are preserved, reset or required. The
checkpoint loader rejects a plan that preserves a missing or incompatible state,
or that both preserves and resets the same state. `weights_only` is a distinct
checkpoint kind and is never reported as a full resume.

## Training schedule boundary

Each branch validates its `BatchManifest` before comparison. Schedule identity
includes the manifest id, dataset revision and the complete ordered
`UpdateSampleRef` sequence: sample, dataset, task, source episode, augmentation
seed and ordinal. Repeated sample IDs are allowed across training epochs; the
ordinal identifies each scheduled occurrence.

Every run stores a hash-pinned manifest reference at
`metadata/batch_manifest.json` and an actual consumption trace at
`events/batch_consumption.jsonl`. Analysis verifies the source manifest hash,
sample count and full ordered consumption trace. A0/A1 control cannot pass when
either artifact is absent, corrupt or differs; the primary A2/A1 candidate gate
also requires matching consumption traces.

## Evaluation boundary

`EpisodeRecord` is the statistical observation. Its within-condition uniqueness
key is:

`model_id + benchmark_id + suite_id + task_id + scene_id + instruction_id +
instruction_type + condition_id + training_seed + initial_state_seed +
evaluation_seed`.

For a paired comparison, analysis removes `condition_id` from that key and
matches the same evaluation case across conditions. It reports unmatched and
unknown-label counts instead of silently treating them as valid pairs. A paired
comparison must name two distinct conditions.

The record keeps capability (`success`) separate from safety (`safe_outcome`,
`violation_type`, `safety_cost`). The evaluator must retain failed and terminated
episodes; exclusions are represented in analysis summaries rather than deleting
raw records.

`StepTraceRecord` is optional but required for hot-update transient analysis. Each
row identifies the active policy version and action chunk, so a switch cannot be
reconstructed from wall-clock timestamps alone.

## Deployment boundary

Hot updates use `HotUpdateRequest` and `HotUpdateEvent`. The request fixes the
switch boundary, action queue policy, runtime-state policy and rollback policy.
The event records requested and actual UTC switch times, active model/adapter
hashes, queued actions before/after, failure/rollback reasons and whether the
event is the same-weight no-op control. A successful event requires an actual
switch timestamp; no-op events require identical model and adapter hashes.

## Compatibility gates

Before a run is admitted:

1. model manifest, model config and adapter id agree;
2. model/adapter id, processor revision, embodiment, complete observation fields
   and camera mapping, action schema/normalization, checkpoint format and
   supported modes agree;
3. update sample manifest is immutable and disjoint from evaluation ids;
4. every interruption condition has a valid `RestorePlan` against the checkpoint;
5. episode and step-trace schemas match the evaluation config;
6. resolved config hash, Git commit, resource revisions and environment are
   written to provenance;
7. every output is written under the run id and no completed run is overwritten.

Provenance start/end values are timezone-aware UTC ISO-8601 timestamps. An end
timestamp earlier than the start is rejected. Analysis metadata records the raw
analysis config file hash separately from the resolved analysis config hash.

The current implementation contains the shared dataclasses, path/config checks
and a configuration preflight. Native model loading, simulator control and
statistical aggregation are deliberately downstream implementations of these
contracts.
