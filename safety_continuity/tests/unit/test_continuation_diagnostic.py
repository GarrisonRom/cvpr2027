"""Scientific bookkeeping and false-success regression cases for diagnosis."""
import copy
from pathlib import Path
import unittest

import numpy as np

from safety_continuity.common.config import load_yaml
from safety_continuity.common.errors import ContractError
from safety_continuity.evaluation.continuation_diagnostic import compare_actions, select_sources, validate_config
from safety_continuity.analysis.continuation_diagnostic import paired_events


def matrix():
    jobs = []
    for seed in (42, 43, 44):
        jobs.append({"run_id": f"20261105_mvp_pi05_continuous_s{seed}", "condition_label": "Continuous training",
                     "training_seed": seed, "interruption_step": None, "checkpoint_steps": [499,999,1499,1999],
                     "checkpoint_root": "/checkpoints/continuous", "return_code": 0})
        for label, slug in (("Full-state resume", "full"), ("Optimizer-moment reset", "reset"), ("Weights-only restart", "weights")):
            for position in (500,1000,1500):
                jobs.append({"run_id": f"20261105_mvp_pi05_{slug}{position}_s{seed}", "condition_label": label,
                             "training_seed": seed, "interruption_step": position,
                             "checkpoint_steps": [2000-position-1 if slug == "weights" else 1999],
                             "checkpoint_root": "/checkpoints/branch", "return_code": 0})
    return {"schema": "mvp_training_matrix.v1", "status": "training_complete", "completed_jobs": jobs}


class ContinuationDiagnosticTests(unittest.TestCase):
    def test_shared_original_has_no_training_seed_and_steps_are_updates(self):
        rows = select_sources(matrix(), Path("original"), smoke=False)
        self.assertEqual(len(rows), 22)
        self.assertIsNone(rows[0]["training_seed"])
        self.assertEqual(rows[0]["optimizer_updates"], 0)
        self.assertEqual(sum(row["rollout"] for row in rows), 4)
        self.assertEqual([row["optimizer_updates"] for row in rows if row["id"].startswith("continued")][:4], [500,1000,1500,2000])
        self.assertEqual(len(select_sources(matrix(), Path("original"), smoke=True)), 2)

    def test_duplicate_or_missing_matrix_coverage_is_rejected(self):
        value = matrix()
        value["completed_jobs"][-1] = copy.deepcopy(value["completed_jobs"][-2])
        with self.assertRaises(ContractError):
            select_sources(value, Path("original"), smoke=False)
        value = matrix()
        value["completed_jobs"][0]["checkpoint_steps"][0] = 500
        with self.assertRaises(ContractError):
            select_sources(value, Path("original"), smoke=False)

    def test_action_change_detected_without_calling_it_safety(self):
        left = np.zeros((5,7))
        self.assertTrue(compare_actions(left, left, atol=0, rtol=0)["exact_equal"])
        right = left.copy(); right[0,0] = .1
        result = compare_actions(left, right, atol=1e-5, rtol=1e-5)
        self.assertFalse(result["within_tolerance"])
        self.assertAlmostEqual(result["max_abs_delta"], .1)
        self.assertNotIn("unsafe", result)

    def test_nonfinite_and_shape_mismatches_never_pass(self):
        for right in (np.full((5,7), np.nan), np.full((5,7), np.inf), np.zeros((5,6)), np.zeros((4,7))):
            with self.assertRaises(ContractError):
                compare_actions(np.zeros((5,7)), right, atol=1e-5, rtol=1e-5)

    def test_config_rejects_invalid_rng_and_fake_repeats(self):
        config = load_yaml(Path(__file__).resolve().parents[2]/"configs/evaluation/continuation_diagnostic.yaml")
        validate_config(config)
        for key, value in (("probe_repeats", 1), ("evaluation_seed", -1), ("episodes_per_task", True),
                           ("action_atol", float("nan")), ("probe_query_indices", [0,0])):
            invalid = {**config, key: value}
            with self.assertRaises(ContractError):
                validate_config(invalid)

    def test_paired_discordance_and_joint_initial_contact_exclusion(self):
        left = [{"episode_key": str(i), "task_id": "task", "initial_state_sha256": str(i),
                 "state_manifest_sha256": "same", "evaluation_seed": 2048+i,
                 "initial_human_contact": False, "new_contact_without_initial_contact": False,
                 "success": i == 0} for i in range(3)]
        right = copy.deepcopy(left)
        right[1]["new_contact_without_initial_contact"] = True
        right[2]["initial_human_contact"] = True
        result = paired_events(left,right,"new_contact_without_initial_contact")
        self.assertEqual(result["n"],2)
        self.assertEqual(result["excluded_initial_contact"],1)
        self.assertEqual(result["new_events"],1)
        self.assertEqual(result["delta"],.5)
        right[0]["initial_state_sha256"] = "different"
        with self.assertRaises(ContractError):
            paired_events(left,right,"success")

    def test_all_initial_contact_is_unestimable_not_zero_risk(self):
        row = {"episode_key":"x","task_id":"task","initial_state_sha256":"same",
               "state_manifest_sha256":"same","evaluation_seed":2048,"initial_human_contact":True,
               "new_contact_without_initial_contact":False}
        result = paired_events([row],[row],"new_contact_without_initial_contact")
        self.assertEqual(result["n"],0)
        self.assertIsNone(result["delta"])


if __name__ == "__main__":
    unittest.main()
