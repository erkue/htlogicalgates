import numpy as np
from typing import List, Optional
from numpy.typing import NDArray

from ._global_vars import ITYPE
from .symplectic_rep.helper import pauli_string_to_list
from .resources.resources import load_qecc

class QECC:
    pass

    def __init__(self, e_mat : NDArray):
        self._e_mat = e_mat
        self._distance = -1

    def get_e_matrix(self) -> NDArray:
        return self._e_mat

    @property
    def n(self) -> int:
        return np.shape(self._e_mat)[0]//2

    @property
    def k(self) -> int:
        return np.shape(self._e_mat)[1] - self.n

    @property
    def d(self) -> int:
        if self._distance == -1:
            self._compute_distance()
        return self._distance

    def _compute_distance(self):
        pass

def get_qecc(*inp) -> QECC:
    if len(inp) == 0:
        raise ValueError(f"Input '{str(inp)}' is invalid.")
    if len(inp) == 1 and isinstance(inp[0], str):
        return get_qecc_from_string(*inp)
    if len(inp) == 1 and isinstance(inp[0], list) and len(inp[0]) == 3:
        return get_qecc_from_paulis(*(inp[0]))
    if len(inp) == 3:
        return get_qecc_from_paulis(*inp)
    raise ValueError(f"Input '{str(inp)}' is invalid!")


def get_qecc_from_paulis(x_logicals, z_logicals, stabilizers) -> QECC:
    k = len(x_logicals)
    n = len(stabilizers) + k
    els = [pauli_string_to_list(i, n) for i in x_logicals + z_logicals + stabilizers]
    return QECC(np.array(els, dtype=ITYPE).T)

def get_qecc_from_string(s : str) -> QECC:
    return QECC(load_qecc(s))