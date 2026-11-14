import unittest
import numpy as np

from safety_continuity.common.errors import ContractError
from safety_continuity.evaluation.render_diagnostic import array_difference


class RenderDiagnosticTests(unittest.TestCase):
    def test_one_channel_one_pixel_changes_without_integer_overflow(self):
        a = np.zeros((2, 2, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0, 0] = 255
        self.assertEqual(array_difference(a, a)["changed_pixels"], 0)
        result = array_difference(a, b)
        self.assertEqual(result["changed_pixels"], 1)
        self.assertEqual(result["changed_values"], 1)
        self.assertEqual(result["max_abs_delta"], 255)
        self.assertEqual(array_difference(b, a), result)

    def test_empty_or_nonfinite_or_different_shape_is_rejected(self):
        for a, b in (([], []), ([float("nan")], [0.]), ([1., 2.], [1.]),
                     (np.array([1], dtype=np.int8), np.array([1], dtype=np.int16))):
            with self.assertRaises(ContractError):
                array_difference(a, b)
