import unittest
import os
from tempfile import TemporaryDirectory

from htlogicalgates import *


class TestTailorLogicalGate(unittest.TestCase):
    def test_tailor_logical_gate(self):
        conn = Connectivity("circular", num_qubits=4)
        qecc = StabilizerCode("4_2_2")
        log_gate = Circuit("H 0", 2)
        circ, status = tailor_logical_gate(qecc, conn, log_gate, 2)
        self.assertEqual(status, "Optimal")
        self.assertEqual(circ.two_qubit_gate_count(), 4)

    def test_tailor_logical_gate_phase(self):
        conn = Connectivity("all", num_qubits=3)
        qecc = StabilizerCode("trivial", num_qubits=3)
        log_gate = Circuit("H 2\nX 0\nCZ 0 1\nCZ 0 2\nY 1\nH 0\nS 1\nCX 1 0\nY 1\nH 1\nX 2", 3).to_clifford()
        circ, status = tailor_logical_gate(qecc, conn, log_gate, 2)
        self.assertEqual(status, "Optimal")
        self.assertTrue(circ.to_clifford() == log_gate)

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
        self.assertEqual(res["Gates"][0]["Circuit"].two_qubit_gate_count(), 0)
        self.assertEqual(res["Gates"][1]["Circuit"].two_qubit_gate_count(), 4)
        self.assertEqual(res["Gates"][121]
                         ["Circuit"].two_qubit_gate_count(), 3)
        
    def test_tailor_optimize_h_count(self):
        conn = Connectivity("circular", num_qubits=4)
        qecc = StabilizerCode("4_2_2")
        log_gate = Circuit("H 0", 2)
        circ, status = tailor_logical_gate(qecc, conn, log_gate, 2,
                                           cost_function=CostFunction.h_count)
        self.assertEqual(status, "Optimal")


class TestSaveLoadResults(unittest.TestCase):
    def test_save_load_results_dictionary(self):
        conn = Connectivity("circular", num_qubits=4)
        qecc = StabilizerCode("4_2_2")
        res = tailor_multiple_logical_gates(qecc, conn, [0, 1, 121], 2)
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "results.json")
            save_results_dictionary(res, filepath)
            res_loaded = load_results_dictionary(filepath)
        self.assertEqual(res_loaded["Gates"][0]["Circuit"].two_qubit_gate_count(),
                         res["Gates"][0]["Circuit"].two_qubit_gate_count())
        self.assertEqual(res_loaded["Gates"][1]["Circuit"].two_qubit_gate_count(),
                         res["Gates"][1]["Circuit"].two_qubit_gate_count())
        self.assertEqual(res_loaded["Gates"][121]["Circuit"].two_qubit_gate_count(),
                         res["Gates"][121]["Circuit"].two_qubit_gate_count())
        self.assertEqual(res_loaded["Meta"]["Number CZ layers"], 2)
        self.assertIsInstance(res_loaded["Meta"]["Connectivity"], Connectivity)
        self.assertIsInstance(res_loaded["Meta"]["Code"], StabilizerCode)
        self.assertEqual(res_loaded["Meta"]["Connectivity"].num_qubits, 4)
        self.assertEqual(res_loaded["Meta"]["Code"].k, 2)

    def test_save_load_none(self):
        conn = Connectivity("circular", num_qubits=4)
        qecc = StabilizerCode("4_2_2")
        res = tailor_multiple_logical_gates(qecc, conn, [0, 1, 121], 2, time_limit=0.001)
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "results.json")
            save_results_dictionary(res, filepath)
            res_loaded = load_results_dictionary(filepath)
        self.assertEqual(res_loaded["Gates"][0]["Circuit"],
                         res["Gates"][0]["Circuit"])
        self.assertEqual(res_loaded["Gates"][1]["Circuit"],
                         res["Gates"][1]["Circuit"])
        self.assertEqual(res_loaded["Gates"][121]["Circuit"],
                         res["Gates"][121]["Circuit"])
        self.assertEqual(res_loaded["Meta"]["Number CZ layers"], 2)
        self.assertIsInstance(res_loaded["Meta"]["Connectivity"], Connectivity)
        self.assertIsInstance(res_loaded["Meta"]["Code"], StabilizerCode)
        self.assertEqual(res_loaded["Meta"]["Connectivity"].num_qubits, 4)
        self.assertAlmostEqual(res_loaded["Meta"]["Time limit"], 0.001)
        self.assertEqual(res_loaded["Meta"]["Code"].k, 2)
