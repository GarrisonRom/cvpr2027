# Safety Continuity Experiment Workspace

This directory contains version-controlled experiment code and metadata. Large datasets, model weights, checkpoints, caches, and generated runs live in the sibling workspace directories and are excluded from Git.

## Layout

| Path | Contents |
|---|---|
| configs/data/ | Dataset and split configuration |
| configs/models/ | Model, processor, and action-interface configuration |
| configs/training/ | Optimizer, schedule, precision, and resume configuration |
| configs/evaluation/ | Task suites, seeds, episode counts, and metric settings |
| configs/interruption/ | Interruption points and checkpoint-state ablations |
| configs/hot_update/ | Weight-switch points and runtime-state policies |
| src/safety_continuity/data/ | Dataset adapters and manifest readers |
| src/safety_continuity/checkpointing/ | Checkpoint inventory, save/restore, and integrity checks |
| src/safety_continuity/training/ | Training entry points and resume instrumentation |
| src/safety_continuity/deployment/ | Policy serving and hot-update instrumentation |
| src/safety_continuity/evaluation/ | Safety/capability evaluation adapters and metrics |
| src/safety_continuity/probes/ | Hidden-state probe extraction and analysis |
| src/safety_continuity/mitigation/ | Release gates and safety-preservation methods |
| scripts/ | Reusable setup, preflight, train, evaluate, and analysis commands |
| manifests/datasets/ | Dataset versions, licenses, splits, and checksums |
| manifests/models/ | Model revisions, licenses, weight files, and checksums |
| environments/ | Pinned environment specifications |

## Runtime storage roots

The local and cloud checkout should use the same relative paths by default:

| Workspace path | Contents |
|---|---|
| data/raw/ | Original downloads or collected data |
| data/processed/ | Converted or cached dataset payloads |
| data/splits/ | Dataset split manifests; small manifests may be tracked |
| models/base/ | Downloaded pretrained weights |
| models/finetuned/ | Exported inference-ready model versions |
| models/adapters/ | LoRA and other adapter weights |
| checkpoints/training/ | Full training-state checkpoints |
| checkpoints/release/ | Candidate release snapshots |
| checkpoints/hot_update/ | Old/new deployment weights and switch bundles |
| runs/ | Logs, metrics, rollouts, activations, and figures |
| cache/ | Hugging Face, Torch, and dataset caches |

On the cloud, these directories may point to mounted storage. Keep the code checkout itself on the instance disk and mount large payload storage at the corresponding runtime roots. Do not put model/data payloads in the Git checkout's tracked source directories.

## Run identity

Each run should record the exact Git commit plus resolved config, model and dataset revisions, checksums, random seed, GPU/driver, and environment lock. The cloud run must use the same commit as the local code being reviewed.
