import unittest

from htlogicalgates.circuit import *


class TestCircuit(unittest.TestCase):
    def test_constructor(self):
        c = Circuit(2)
        self.assertEqual(c.num_qubits, 2)
        c = Circuit("H 2")
        self.assertEqual(c.num_qubits, 3)

    def test_empty_lines_string(self):
        c = Circuit("H 0\n\nH 1")
        self.assertEqual(c.gate_count(), 2)

    def test_comment_string(self):
        c = Circuit("H 0\nCZ 0 1 #Comment\nH 1")
        self.assertEqual(c.gate_count(), 3)

    def test_no_whitespace_string(self):
        c = Circuit("H 0\nH1\nCZ 0 1")
        self.assertEqual(c.gate_count(), 3)

    def test_shallow_optimize(self):
        c = Circuit("H 2\nS 2\nCZ 0 2")
        self.assertEqual(c.gate_count(), 3)
        cliff = c.to_clifford()
        c.shallow_optimize()
        self.assertLessEqual(c.gate_count(), 2)
        self.assertTrue(cliff == c.to_clifford())