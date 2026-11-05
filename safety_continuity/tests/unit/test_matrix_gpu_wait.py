"""Resource waiting must not launch an evaluation when memory is occupied."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate/run_matrix_evaluation.py"
spec = importlib.util.spec_from_file_location("matrix_eval_wait_tests", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GpuWaitTests(unittest.TestCase):
    def test_wait_until_memory_is_free(self):
        status = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            with patch.object(runner.subprocess, "run", side_effect=[
                SimpleNamespace(stdout="32607, 5096\n"),
                SimpleNamespace(stdout="32607, 30000\n"),
            ]), patch.object(runner.time, "sleep") as sleep:
                runner.wait_for_gpu(24576, status=status, status_path=path)
                sleep.assert_called_once_with(30)
                self.assertEqual(status["status"], "running")
                self.assertEqual(status["gpu_free_mib"], 30000)

    def test_insufficient_total_memory_fails(self):
        with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="16384, 15000\n")):
            with self.assertRaises(runner.ContractError):
                runner.wait_for_gpu(24576, status={}, status_path=Path("unused.json"))


if __name__ == "__main__":
    unittest.main()
