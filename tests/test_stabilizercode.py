import unittest

from htlogicalgates.stabilizercode import *


class TestStabilizerCode(unittest.TestCase):
    def test_constructor(self):
        c = StabilizerCode("4_2_2", skip_tests=True)
        self.assertEqual(c.n, 4)
        self.assertEqual(c.k, 2)

    def test_constructor_checks(self):
        c = StabilizerCode("4_2_2")
        self.assertEqual(c.n, 4)
        self.assertEqual(c.k, 2)

    def test_calculate_distance(self):
        c = StabilizerCode("4_2_2")
        self.assertEqual(c.d, 2)

    def test_stabilizer_generators(self):
        stabs = ["X0 X1", "X0 X1 X2", "X2", "Z0 Z1"]
        self.assertEqual(are_independent_generators(stabs), False)
        new_stabs = reduce_to_stabilizer_generators(stabs)
        self.assertEqual(new_stabs, ["X0 X1 X2", "X2", "Z0 Z1"])
        self.assertEqual(are_independent_generators(new_stabs), True)
