"""Targeted regression cases for live geometry extraction and pair semantics."""
import math
import unittest
from types import SimpleNamespace

from safety_continuity.common.errors import ContractError
from safety_continuity.safety.contacts import ContactPolicy, HumanContactMonitor, geom_name


class ContactPairTests(unittest.TestCase):
    def setUp(self):
        self.policy = ContactPolicy(
            frozenset({"gripper0_hand_collision"}), frozenset({"plate_with_hand_1_g1"}),
            frozenset({"knife_1_g1"}), frozenset({"plate_with_hand_1_g3"}), (),
        )

    def test_compound_plate_is_not_hand(self):
        self.assertIsNone(self.policy.violation("gripper0_hand_collision", "plate_with_hand_1_g3"))
        self.assertEqual(self.policy.violation("plate_with_hand_1_g1", "gripper0_hand_collision"), "robot_human_contact")
        self.assertIsNone(self.policy.violation("gripper0_hand_collision", "knife_1_g1"))
        self.assertIsNone(self.policy.violation("table", "plate_with_hand_1_g1"))
        self.assertEqual(self.policy.violation("knife_1_g1", "plate_with_hand_1_g1"), "object_human_contact")

    def test_actual_wrapper_api(self):
        names = ["gripper0_hand_collision", "plate_with_hand_1_g1"]
        model = SimpleNamespace(ngeom=2, geom_id2name=lambda i: names[i])
        sim = SimpleNamespace(model=model, data=SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=0, geom2=1, dist=-0.001)]))
        monitor = HumanContactMonitor(self.policy)
        self.assertEqual(monitor.observe(simulator_state=sim, action=[0.0]*7).violation_type, "robot_human_contact")
        self.assertEqual(monitor.contacts[0]["geom2"], names[1])
        sim.data.ncon = 0
        self.assertEqual(monitor.observe(simulator_state=sim, action=[0.0]*7).outcome, "unsafe")

    def test_invalid_state_rejected(self):
        with self.assertRaises(ContractError):
            geom_name(SimpleNamespace(ngeom=1), 0)
        with self.assertRaises(ContractError):
            HumanContactMonitor(self.policy).observe(simulator_state=None, action=[math.nan])
        with self.assertRaises(ContractError):
            ContactPolicy(frozenset({"x"}), frozenset({"x"}), frozenset(), frozenset(), ())


if __name__ == "__main__":
    unittest.main()
