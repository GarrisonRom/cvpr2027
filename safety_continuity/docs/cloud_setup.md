# AutoDL Resource Map

## Current payload location

The MVP resources were downloaded on the current AutoDL instance under:

    /root/autodl-tmp/cvpr2027-payloads

This path is on the current instance data disk. Before stopping, replacing, or recreating the instance, verify whether that disk is preserved or copy the payload directory to storage shared with the next instance.

## Environment roots

Set these variables in the AutoDL shell before running project commands:

    export SC_PAYLOAD_ROOT=/root/autodl-tmp/cvpr2027-payloads
    export SC_DATA_ROOT="$SC_PAYLOAD_ROOT/data"
    export SC_MODEL_ROOT="$SC_PAYLOAD_ROOT/models"
    export SC_CHECKPOINT_ROOT="$SC_PAYLOAD_ROOT/checkpoints"
    export SC_RUN_ROOT="$SC_PAYLOAD_ROOT/runs"
    export SC_CACHE_ROOT="$SC_PAYLOAD_ROOT/cache"
    export HF_ENDPOINT=https://hf-mirror.com

The source checkout remains separate from these payload roots. Do not put model weights, datasets, checkpoints, or run outputs in the Git checkout.

## Downloaded resources

| Resource | Relative path | Revision | Verified size |
|---|---|---|---:|
| LIBERO-Safety π0.5 checkpoint | models/finetuned/pi05_libero_safety/e66d84bfec17770384550593e39975f7e21e3463 | e66d84bfec17770384550593e39975f7e21e3463 | 12,440,507,736 bytes |
| LIBERO-10 no-noops TFRecords | data/raw/modified_libero_rlds/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_10_no_noops/1.0.0 | 6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551 | 3,656,815,424 bytes |
| LIBERO-Safety simulator assets | data/raw/libero_safety_assets/94ccca773d1bab760d1aecec4f62c8d4b8797d40/assets.zip | 94ccca773d1bab760d1aecec4f62c8d4b8797d40 | 10,670,353,443 bytes |

The Safety dataset's full LeRobot/video training payload was not downloaded; the MVP starts from the released safety-tuned checkpoint and only needs the simulator assets for evaluation.

## Additional model zoo

The public action-policy checkpoints listed in safety_continuity/manifests/models/open_policy_inventory.yaml are staged under the current payload root. The per-file revision/path/size manifest is safety_continuity/manifests/artifacts/additional_models.tsv, with checksums in safety_continuity/manifests/artifacts/checksums/additional_models.sha256.

The staged set includes SmolVLA, OpenVLA, OpenVLA-OFT, GR00T N1.7 base and LIBERO inference weights, Xiaomi-Robotics-0 pretrain and LIBERO weights, Octo-Small/Base, RDT-1B, and the PyTorch T5-XXL plus SigLIP encoders used by RDT. The duplicate TensorFlow T5 serialization and GR00T's 16-rank optimizer-state training snapshot were not included.

GR00T N1.7 requires nvidia/Cosmos-Reason2-2B at runtime. That dependency is gated and the anonymous AutoDL mirror request returned HTTP 403, so GR00T weights are present but cannot be assumed runnable until access is approved. ACT has no general pretrained checkpoint; the listed closed humanoid systems and non-action world model are cataloged as unavailable/non-comparable rather than represented by downloaded weights.

## Storage headroom

The payload root currently uses about 133GB of the 150GB AutoDL instance disk, leaving roughly 17GB free. The LIBERO-Safety assets remain as a verified ZIP and have not been extracted. Expand the data disk before extracting simulator assets, creating training checkpoints, or starting runs. Do not delete model/data payloads to make room without first confirming a backup or shared mount.

## Checksums

Tracked per-file SHA-256 lists are under safety_continuity/manifests/artifacts/checksums/. The downloaded payloads were verified against the tracked lists. The assets ZIP also passed an archive integrity check. Do not edit or regenerate checksum lists without re-downloading or independently verifying the corresponding files.

## Current execution status

The current container has no usable NVIDIA driver/device and no Python interpreter, so it was used only for downloads and integrity checks. Training and simulator evaluation require a GPU-enabled instance with the pinned environment installed. Verify that the next instance can access the payload root before beginning a run.
