import unittest
import numpy as np

from htlogicalgates.stabilizercode import *


class TestStabilizerCode(unittest.TestCase):
    def test_constructor_string(self):
        c = StabilizerCode("4_2_2")
        self.assertEqual(c.n, 4)
        self.assertEqual(c.k, 2)
        np.testing.assert_equal(c.get_e_matrix().T,
                                      np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 0, 1, 0],
                                                [0, 0, 0, 0, 1, 1, 0, 0],
                                                [1, 1, 1, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 1, 1, 1]]))

    def test_constructor_string_int(self):
        c = StabilizerCode("trivial", 5)
        self.assertEqual(c.n, 5)
        self.assertEqual(c.k, 5)

    def test_constructor_paulis(self):
        x_log = ["X0 X1", "X0 X2"]
        z_log = ["Z0 Z2", "Z0 Z1"]
        stab = ["X0 X1 X2 X3", "Z0 Z1 Z2 Z3"]
        c = StabilizerCode(x_log, z_log, stab)
        self.assertEqual(c.n, 4)
        self.assertEqual(c.k, 2)
        np.testing.assert_equal(c.get_e_matrix().T,
                                      np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 0, 1, 0],
                                                [0, 0, 0, 0, 1, 1, 0, 0],
                                                [1, 1, 1, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 1, 1, 1]]))

    def test_constructor_checks(self):
        x_log = ["X0 X1 X2", "X0 X2"]
        z_log = ["Z0 Z2", "Z0 Z1"]
        stab = ["X0 X1 X2 X3", "Z0 Z1 Z2 Z3"]
        c = StabilizerCode(x_log, z_log, stab, verify=False)
        self.assertEqual(c.n, 4)
        self.assertEqual(c.k, 2)
        np.testing.assert_equal(c.get_e_matrix().T,
                                      np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 0, 1, 0],
                                                [0, 0, 0, 0, 1, 1, 0, 0],
                                                [1, 1, 1, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 1, 1, 1]]))
        self.assertRaises(ValueError, lambda: StabilizerCode((x_log, z_log, stab)))

    def test_calculate_distance(self):
        c = StabilizerCode("4_2_2")
        self.assertEqual(c.d, 2)

    def test_stabilizer_generators(self):
        stabs = ["X0 X1", "X0 X1 X2", "X2", "Z0 Z1"]
        self.assertEqual(are_independent_generators(stabs), False)
        new_stabs = reduce_to_stabilizer_generators(stabs)
        self.assertEqual(new_stabs, ["X0 X1 X2", "X2", "Z0 Z1"])
        self.assertEqual(are_independent_generators(new_stabs), True)
