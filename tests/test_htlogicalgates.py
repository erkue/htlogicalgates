import unittest

from htlogicalgates import *


class TestTailorLogicalGate(unittest.TestCase):
    def test_tailor_logical_gate(self):
        conn = Connectivity("circular", num_qubits=4)
        qecc = StabilizerCode("4_2_2")
        log_gate = Circuit("H 0", 2)
        circ, status = tailor_logical_gate(qecc, conn, log_gate, 2)
        self.assertEqual(status, "Optimal")
        self.assertEqual(circ.two_qubit_gate_count(), 4)
