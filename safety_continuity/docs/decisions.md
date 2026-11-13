# Design Decisions

## 2026-11-14: Diagnose retained weights before new adaptation training

Run a bounded read-only diagnostic of existing original, continuous-prefix and
resume-branch weights. Include same-input/RNG repetition, fresh-process reload
and original-versus-final paired rollouts. The reload control does not perform
save/export conversion. Separate initial human contacts from post-takeover events;
do not interpret geometry contacts as injury or known task-specific prohibition.
Use separate diagnostic records with null original training seed. This does not
reactivate the shelved interruption study or establish the broader adaptation
recipe. Smoke and main outputs use separate immutable runtime roots.

## 2026-11-07: Separate benign adaptation from the shelved interruption pilot

The active design asks whether benign task adaptation changes an already
observable safety constraint behavior. It retains one model, one audited update
plan and training seeds 42, 43 and 44. The interruption pilot is shelved without
rewriting its protocols, metric versions or run records.

`docs/benign_adaptation_experiment.md` owns the new proposed design. Before
implementation, it requires original-checkpoint evaluation, reload parity,
dataset audit, dynamic monitor calibration and a qualified capability/safety
baseline. Existing action-output-only training does not stand in for a broader
adaptation recipe. The main recipe must pass adapter compatibility and resource
checks before launch.

HazardArena already reports hazard amplification from safe-only VLA fine-tuning.
The broad phenomenon cannot be claimed as new. Any new study must distinguish
capability transfer from loss of an established safety behavior, or honestly
report replication or absence of a repeatable effect. Discovery and confirmation
states remain separate; no training or remote execution is authorized by this
design document itself.

## 2026-11-02: Freeze cross-module contracts before native adapters

The project starts with shared dataclasses and preflight validation because the
downloaded model zoo contains incompatible processor, action and checkpoint
formats. Native loaders must implement `ModelAdapter` and cannot leak framework
objects into the episode or analysis interfaces.

## 2026-11-02: Keep unresolved π0.5 fields explicit

The released π0.5 checkpoint and LIBERO adapter must be inspected on a GPU
instance before camera keys, action dimension and chunk length are written into a
run. Config values are therefore `null` with a preflight requirement instead of
being guessed from another model.

## 2026-11-02: Split root is a first-class runtime mount

Frozen update sample manifests live under `SC_SPLIT_ROOT` when available and
default to `data/splits/`. This keeps sample schedules separate from immutable
raw data while preserving the same relative layout locally and in the cloud.

## 2026-11-05: Freeze one model and update plan, vary interruption position

The first expanded MVP keeps π0.5, the update dataset revision and the three
training seeds fixed. It varies interruption position at steps 500, 1000 and
1500, and increases the number of paired evaluation episodes. One continuous
run per seed produces all prefix checkpoints; resume branches reuse those
checkpoints. Official and self-built safety tracks remain separate, and
self-built monitor calibration is required before a safety interpretation.
