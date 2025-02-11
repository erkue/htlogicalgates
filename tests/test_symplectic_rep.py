import unittest

import numpy as np

from htlogicalgates.symplectic_rep.clifford_gate import *
from htlogicalgates.symplectic_rep.helper import *
from htlogicalgates.symplectic_rep.integer_symplectic import symplectic_matrix, symplectic_matrix_inverse, is_symplectic


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

    def test_pauli_string_to_list(self):
        arr = pauli_string_to_list("x0 X2 z1 X2 Y3", 4)
        np.testing.assert_array_equal([1, 0, 0, 1, 0, 1, 0, 1], arr)

    def test_AMG_decomposition(self):
        A = np.array([[1, 0, 1, 1],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 1]])
        M, G = get_AMG_decomposition(A)
        M_comp = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [1, 0, 1, 0],
                           [1, 0, 1, 1],])
        G_comp = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],])
        np.testing.assert_equal(M, M_comp)
        np.testing.assert_equal(G, G_comp)


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
        self.assertEqual(c0.id, 0)
        self.assertEqual(c0.pauli_id, 0)
        self.assertTrue(Clifford(32, 2), Clifford(32, 0, 2))
        c5 = Clifford(3432, 45, 3)
        self.assertEqual(c5.num_qubits, 3)
        self.assertEqual(c5.id, 3432)
        self.assertEqual(c5.pauli_id, 45)


    def test_clifford_group_size_mod_pauli(self):
        self.assertEqual(Clifford.group_size_mod_pauli(1), 6)
        self.assertEqual(Clifford.group_size_mod_pauli(2), 720)
        self.assertEqual(Clifford.group_size_mod_pauli(3), 1451520)

class TestRandomSymplectic(unittest.TestCase):
    def test_symplectic_matrix(self):
        for n, i in zip(range(1, 5), [2, 355, 74329, 4234324]):
            m = symplectic_matrix(i, n)
            self.assertTrue(is_symplectic(m))
            self.assertEqual(symplectic_matrix_inverse(m, n), i)
