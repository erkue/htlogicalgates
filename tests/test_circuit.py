import unittest

from htlogicalgates.circuit import *


class TestCircuit(unittest.TestCase):
    def test_constructor(self):
        c = Circuit(2)
        self.assertEqual(c.num_qubits, 2)
        c = Circuit("H 2")
        self.assertEqual(c.num_qubits, 3)

    def test_shallow_optimize(self):
        c = Circuit("H 2\nS 2\nCZ 0 2")
        self.assertEqual(c.gate_count(), 3)
        cliff = c.to_clifford()
        c.shallow_optimize()
        self.assertLessEqual(c.gate_count(), 2)
        self.assertTrue(cliff == c.to_clifford())