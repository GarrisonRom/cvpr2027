"""Frozen LIBERO-10 capability states must be content-addressed and paired."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from safety_continuity.common.config import sha256_file, sha256_payload
from safety_continuity.common.errors import ContractError
from safety_continuity.evaluation.capability_matrix import validate_state_manifest


class CapabilityStateManifestTests(unittest.TestCase):
    def _manifest(self, root: Path) -> tuple[Path, Path]:
        states_path = root / "states.npz"
        arrays = {}
        tasks = []
        for index in range(10):
            values = np.arange(20, dtype=np.float64).reshape(10, 2) + index
            key = f"task_{index:02d}"
            arrays[key] = values
            bddl = root / f"task_{index:02d}.bddl"
            bddl.write_text(f"task {index}", encoding="utf-8")
            source_states = root / f"task_{index:02d}.pruned_init"
            source_states.write_text(f"source states {index}", encoding="utf-8")
            tasks.append({
                "task_id": f"libero10_task_{index:02d}",
                "task_index": index,
                "task_name": f"task {index}",
                "bddl_path": str(bddl),
                "bddl_sha256": sha256_file(bddl),
                "source_state_path": str(source_states),
                "source_state_sha256": sha256_file(source_states),
                "states_key": key,
                "state_count": 10,
                "state_shape": list(values.shape),
                "state_sha256": [],
            })
        np.savez_compressed(states_path, **arrays)
        # The production manifest hashes raw array bytes, matching this test's arrays.
        from safety_continuity.evaluation.capability_matrix import _sha256_bytes
        for task, values in zip(tasks, arrays.values()):
            task["state_sha256"] = [_sha256_bytes(np.ascontiguousarray(value).tobytes()) for value in values]
        source_code = root / "benchmark.py"
        source_code.write_text("pinned benchmark source", encoding="utf-8")
        source_hash = sha256_file(source_code)
        source_files = {"benchmark.py": source_hash}
        payload = {
            "schema": "libero10_capability_state_manifest.v1",
            "benchmark_id": "libero10",
            "suite_id": "libero10_official",
            "libero_classic_root": str(root),
            "libero_classic_revision": None,
            "libero_classic_source_sha256": sha256_payload(source_files),
            "libero_classic_source_files": source_files,
            "evaluation_seed": 2048,
            "episodes_per_task": 10,
            "state_file": str(states_path),
            "state_file_sha256": sha256_file(states_path),
            "tasks": tasks,
        }
        payload["manifest_sha256"] = sha256_payload(payload)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path, states_path

    def test_manifest_validates_exact_frozen_states_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._manifest(Path(directory))
            self.assertEqual(validate_state_manifest(manifest)["episodes_per_task"], 10)

    def test_state_payload_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, states = self._manifest(Path(directory))
            states.write_bytes(states.read_bytes() + b"corruption")
            with self.assertRaises(ContractError):
                validate_state_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
