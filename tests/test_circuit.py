import unittest

from htlogicalgates.circuit import *


class TestCircuit(unittest.TestCase):
    def test_constructor(self):
        c = Circuit(2)
        self.assertEqual(c.num_qubits, 2)
        