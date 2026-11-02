# Data Dictionary

## Update sample manifest (`update_sample.v1`)

One JSONL row identifies one deterministic training sample:

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | string | Immutable source sample id |
| `dataset_id` | string | Manifest resource id and revision |
| `task_id` | string | Benchmark task identifier |
| `episode_id` | string | Source episode identifier |
| `augmentation_seed` | integer | Seed for deterministic preprocessing/augmentation |
| `ordinal` | integer | Position in the frozen update schedule |

The ordinal sequence is the audit source for A0/A1/A2/A3 batch identity. A
random seed alone is not evidence that two branches consumed the same data.
Ordinals are contiguous from zero; repeated sample IDs are allowed when a sample
is deliberately repeated, because ordinal and augmentation seed distinguish the
scheduled occurrence.

## Run batch manifest reference (`batch_manifest_ref.v1`)

Every training run stores `metadata/batch_manifest.json` with run/condition/seed,
manifest id, dataset revision, `SC_SPLIT_ROOT` relative path, exact source-file
SHA-256, sample count and expected optimizer-step count.

## Actual batch consumption (`batch_consumption.v1`)

Every optimizer run stores `events/batch_consumption.jsonl`, one row for each
sample actually consumed. Each row records the run/condition/training seed,
contiguous sequence index, optimizer step, microbatch position and complete
`UpdateSampleRef`. Analysis verifies the file against the immutable manifest and
compares the actual ordered records across paired branches.

## Episode record (`episode.v1`)

One row is one evaluation episode. Required identity fields are `run_id`,
`model_id`, `condition_id`, `benchmark_id`, `suite_id`, `task_id`, `scene_id`,
`instruction_id`, `instruction_type`, `initial_state_seed` and `training_seed`.
`evaluation_seed` is recorded when the policy is stochastic. `success` is the
capability result and may be null when an episode terminates before a capability
label exists. `safe_outcome` is one of
`safe`, `unsafe`, `refusal`, `false_refusal` or `unknown`.

`instruction_type` is one of `benign`, `harmful`, `neutral`, `other` or
`unknown`. A `false_refusal` must be a benign instruction. Refusal and false
refusal remain distinct from task success.

`violation_type`, `safety_cost`, `exposure_duration_s` and `completion_time_s`
remain nullable because benchmark suites do not expose every field for every
episode. Missing values are not converted to zero. `termination_reason` is
always retained.

## Step trace (`step_trace.v1`)

Step traces contain `run_id`, `episode_id`, `timestep`, an action vector,
`active_policy_version`, `action_chunk_id`, monitor margins and a simulator state
reference. They are required for switch-transient plots and optional for static
capability-only evaluation. Actions and monitor margins must contain only finite
numeric values; safety costs follow the same rule.

## Provenance and output rules

Every run also stores resolved configuration, resource revisions, code commit,
GPU/software information and checkpoint inventory under `runs/<run_id>/metadata/`.
Raw episode records are append-only. Analysis may exclude rows only by recording
the rule and counts in its validated summary.

## Hot-update event (`hot_update_event.v1`)

Each event includes old/new model artifact hashes and optional adapter hashes,
requested/actual UTC timestamps, the switch boundary, queue policy and before/
after queue counts. Successful events require an actual switch time. Failed
events include a failure reason; rollback reasons are recorded separately.

## Checkpoint manifest (`checkpoint_manifest.v1`)

Each state entry names whether it is included and whether it is compatible with
restore. Included states list one or more file records; each file record has a
checkpoint-root-relative POSIX path, byte size and SHA-256. Excluded states carry
an explicit reset reason. This supports both a single-file state and native
sharded checkpoint directories.
