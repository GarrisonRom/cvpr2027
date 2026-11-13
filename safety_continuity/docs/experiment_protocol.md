# Experiment Protocol

The implemented interruption protocol is preserved in
[`mvp_experiment.md`](mvp_experiment.md). Safety event definitions and
self-built label rules are owned exclusively by
[`safety_measurement_protocol.md`](safety_measurement_protocol.md). This file
defines how a protocol is promoted from a plan to a runnable study; it does not
contain changing results.

As of 2026-11-07, that research line is shelved. The active design draft is
[`benign_adaptation_experiment.md`](benign_adaptation_experiment.md); it has not
yet been promoted to an executable protocol. The states and gates below
describe the historical interruption workflow and must not be assumed to
implement the new study.

## Protocol states

- **draft:** configs and resource manifests may still contain unresolved adapter
  fields; no effect claim is allowed.
- **smoke:** one seed and a short horizon validate loading, checkpointing,
  restore, evaluation records and event logs; smoke output is not evidence.
- **pilot:** the three-seed MVP matrix is run with the frozen sample schedule,
  pre-specified pairing rules and a calibrated safety monitor.
- **confirmation:** a powered study after the pilot fixes task/scene holdouts,
  model variants, interruption positions and sample size.

## Promotion gates

1. Run the non-strict preflight while resources are being staged.
2. Resolve model observation/action fields and create the immutable update sample
   manifest.
3. Run strict preflight and record its output in the run metadata.
4. Complete the smoke workflow and verify the continuous-training/full-state sample identity.
5. Freeze the protocol manifest and commit before the pilot starts.
6. Analyze only explicit run ids listed by the protocol manifest.
7. Keep official LIBERO-Safety outputs and self-built adapted-protocol outputs
   in separate benchmark IDs and summaries.

The protocol manifest is
`manifests/experiments/mvp_protocol.yaml`. Conditions and pilot scale are in
`mvp_experiment.md`; safety definitions and metric denominators are in
`safety_measurement_protocol.md` and `metrics.md`.
