# File Map

The local checkout is the source of truth. Runtime payloads are mounted outside
the checkout and are addressed only through environment roots.

| Area | Current entry point | Responsibility |
|---|---|---|
| Config/path | `src/safety_continuity/common/` | YAML loading, hashes, run context, provenance, run ids and root-safe path resolution |
| Data | `src/safety_continuity/data/` | Frozen sample references and episode/step record schemas |
| Models | `src/safety_continuity/models/` | Registry, config-to-spec translation, native adapters and observation/action compatibility |
| Checkpoints | `src/safety_continuity/checkpointing/` | State inventories, restore plans and compatibility preflight |
| Deployment | `src/safety_continuity/deployment/` | Chunk-level publication and hot-update events |
| Safety | `src/safety_continuity/safety/` | Monitor labels, safety costs and exact live geometry-pair endpoints in `contacts.py` |
| Contact calibration | `src/safety_continuity/evaluation/contact_calibration.py`, `scripts/evaluate/calibrate_contacts.py` | Independent live MuJoCo contact positives/negatives and immutable evidence |
| Expanded safety evaluation | `scripts/evaluate/run_matrix_evaluation.py`, `scripts/evaluate/serve_paired_policy.py` | Explicit training-matrix endpoints, shared frozen states, paired inference RNG and official-resource audit |
| Preflight | `src/safety_continuity/preflight.py` | Configuration and reference checks before a job starts |
| CLI | `src/safety_continuity/cli.py` | Thin command-line entry point |
| Training | `src/safety_continuity/training/` | Frozen schedules, multi-position MVP plan, native-loop instrumentation/callbacks |
| Training launchers | `scripts/train/` | OpenPI branch runner, matrix plan/train launcher and CPU plan smoke |
| Evaluation | `src/safety_continuity/evaluation/` | Benchmark runners that only emit stable records |
| Analysis | `src/safety_continuity/analysis/` | Episode summaries, explicit-run validation, seed-clustered comparisons, resume-control and candidate-signal gates |
| Analysis CLI | `scripts/analyze/` | Explicit run selection, episode validation and MVP paired analysis |
| Plotting | `src/safety_continuity/plotting/` | Future figures/tables from validated summaries |
| Reporting | `src/safety_continuity/reporting/` | Future run reports and paper exports |
| Configs | `configs/` | Human-edited protocol settings; no result values |
| Manifests | `manifests/` | Immutable resource revisions, checksums and protocol identity |
| Runtime | workspace `data/`, `models/`, `checkpoints/`, `runs/`, `cache/` | Ignored payloads and run output only |

When a native implementation is added, it must be an adapter behind the existing
contract. It must not add simulator-specific fields to the shared episode schema.
