# Workspace Instructions

## Project layout

- Keep the existing paper and CVPR template files at the workspace root.
- Put experiment source code, configs, scripts, manifests, and environment lock files under safety_continuity/.
- Put large or generated payloads under the root-level data/, models/, checkpoints/, runs/, and cache/ directories.
- Do not move, rename, or overwrite existing paper files unless the user asks.

## Local-to-cloud consistency

- The local Git checkout is the canonical source for experiment code and configs.
- Cloud instances are execution copies. Run them from a clean checkout of the exact commit used for the run.
- Do not make uncommitted code or config edits directly on a cloud instance. Bring changes back to the local checkout, review them, then sync a new commit.
- Every experiment run must record the Git commit, config file/hash, model identifier and revision, dataset manifest/hash, random seed, GPU model, driver, and software environment.
- Use the same relative directory layout locally and in the cloud checkout. Set data/model/output roots through environment variables or a local ignored .env; do not hard-code machine-specific absolute paths in source or tracked configs.
- Large data, weights, checkpoints, caches, and run outputs are not Git source. Keep small manifests, checksums, licenses, and acquisition notes tracked under safety_continuity/manifests/.

## Reproducibility and checkpoint handling

- Give each run an immutable run ID and save its resolved config beside its logs and metrics.
- Record all state included in a training checkpoint: model/adapter weights, optimizer, scheduler/global step, RNG, sampler/data cursor, gradient accumulation/scaler, EMA, normalization, and distributed layout, as applicable.
- Write checkpoints to a temporary path, verify completion and checksums, then publish them atomically where the framework supports it.
- Do not describe weights-only loading as full training resume.

## Data and credentials

- Never commit access tokens, SSH credentials, private keys, .env files, or signed download URLs.
- Keep raw data immutable. Store dataset revisions, split definitions, and checksums in tracked manifests.
- Do not download large datasets or model weights unless the user requests it.

## Environment changes

- Pin Python, framework, CUDA-facing package, simulator, and model-code versions in environment files under safety_continuity/environments/.
- Before changing shared package versions or experiment interfaces, inspect the affected configs and scripts and explain any compatibility impact.
