import unittest

import numpy as np

from htlogicalgates.connectivity import *


class TestConnenctivity(unittest.TestCase):
    def test_constructor(self):
        arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
        c = Connectivity("linear", 3)
        np.testing.assert_equal(c.matrix, arr)
        self.assertEqual(c.num_qubits, 3)
        np.testing.assert_equal(Connectivity(arr).matrix, arr)
