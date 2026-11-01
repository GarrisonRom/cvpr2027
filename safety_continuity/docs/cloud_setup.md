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

## Checksums

Tracked per-file SHA-256 lists are under safety_continuity/manifests/artifacts/checksums/. The downloaded payloads were verified against the tracked lists. The assets ZIP also passed an archive integrity check. Do not edit or regenerate checksum lists without re-downloading or independently verifying the corresponding files.

## Current execution status

The current container has no usable NVIDIA driver/device and no Python interpreter, so it was used only for downloads and integrity checks. Training and simulator evaluation require a GPU-enabled instance with the pinned environment installed. Verify that the next instance can access the payload root before beginning a run.
