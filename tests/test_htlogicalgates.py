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

    def test_tailor_logical_gate_checks(self):
        conn = Connectivity("circular", num_qubits=5)
        qecc = StabilizerCode("4_2_2")
        log_gate = Circuit("H 0", 2)
        self.assertRaises(ValueError, lambda: tailor_logical_gate(
            qecc, conn, log_gate, 2))
        self.assertRaises(ValueError, lambda: tailor_multiple_logical_gates(
            qecc, conn, [0, 1, 121], 2))
        
        conn = Connectivity("circular", num_qubits=4)
        qecc = StabilizerCode("4_2_2")
        log_gate = Circuit("H 0", 3)
        self.assertRaises(ValueError, lambda: tailor_logical_gate(
            qecc, conn, log_gate, 2))

    def test_tailor_multiple_logical_gates(self):
        conn = Connectivity("circular", num_qubits=4)
        qecc = StabilizerCode("4_2_2")
        res = tailor_multiple_logical_gates(qecc, conn, [0, 1, 121], 2)
        self.assertEqual(len(res["Gates"]), 3)
        self.assertEqual(
            Circuit(res["Gates"][0]["Circuit"]).two_qubit_gate_count(), 0)
        self.assertEqual(
            Circuit(res["Gates"][1]["Circuit"]).two_qubit_gate_count(), 4)
        self.assertEqual(
            Circuit(res["Gates"][121]["Circuit"]).two_qubit_gate_count(), 3)
