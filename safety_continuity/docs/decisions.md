# Design Decisions

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
