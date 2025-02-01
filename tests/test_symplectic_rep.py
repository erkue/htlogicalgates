import unittest

import numpy as np

from htlogicalgates.symplectic_rep.clifford_gate import *
from htlogicalgates.symplectic_rep.helper import *
from htlogicalgates.symplectic_rep.random_symplectic import symplectic_matrix, symplectic_matrix_inverse, is_symplectic


class TestHelper(unittest.TestCase):
    def test_int_to_array(self):
        i = 34725
        a = int_to_bitarray(i, 21)
        self.assertEqual(len(a), 21)
        b = bitarray_to_int(a)
        self.assertEqual(i, b)
        self.assertRaises(ValueError, lambda: int_to_bitarray(i, 10))

    def test_matrix_rank(self):
        mat = np.array([[1, 1, 1, 1, 0], [1, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1], [0, 1, 0, 0, 1]])
        self.assertEqual(matrix_rank(mat), 3)


class TestCliffordGate(unittest.TestCase):
    def test_constructor(self):
        m = np.eye(4, 4, dtype=np.int32)
        p = np.zeros(4, dtype=np.int32)
        me = np.eye(5, 5, dtype=np.int32)
        pe = np.zeros(5, dtype=np.int32)
        pe[-1] = 1
        c0 = Clifford(0, 2)
        c1 = Clifford(0, 0, 2)
        c2 = Clifford(m)
        c3 = Clifford(m, p)
        c4 = Clifford(me, pe, extended=True)
        for c in [c1, c2, c3, c4]:
            self.assertTrue(c0 == c)
        self.assertEqual(c0.clifford_int, 0)
        self.assertEqual(c0.pauli_int, 0)
        self.assertTrue(Clifford(32,2), Clifford(32, 0, 2))
        c5 = Clifford(3432, 45, 3)
        self.assertEqual(c5.num_qubits, 3)
        self.assertEqual(c5.clifford_int, 3432)
        self.assertEqual(c5.pauli_int, 45)


class TestRandomSymplectic(unittest.TestCase):
    def test_symplectic_matrix(self):
        for n, i in zip(range(1, 5), [2, 355, 74329, 4234324]):
            m = symplectic_matrix(i, n)
            self.assertTrue(is_symplectic(m))
            self.assertEqual(symplectic_matrix_inverse(m, n), i)
