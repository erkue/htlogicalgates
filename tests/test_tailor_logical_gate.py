import unittest

from htlogicalgates import *


class TestTailorLogicalGate(unittest.TestCase):
    def test_tailor_logical_gate(self):
        conn = Connectivity("circular", n=4)
        qecc = StabilizerCode("4_2_2")
        log_gate = Circuit.from_string("H 0", 1)
        circ, status = tailor_logical_gate(qecc, conn, log_gate, 2)
        self.assertEqual(status, "Optimal")
