import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from ._global_vars import ITYPE
from .symplectic_rep.helper import pauli_string_to_list
from .resources.resources import load_qecc


class StabilizerCode:
    def __init__(self, *inp):
        if len(inp) == 0: raise ValueError(f"Input '{str(inp)}' is invalid.")
        elif len(inp) == 1 and isinstance(inp[0], str): self._e_mat = load_qecc(*inp)
        elif len(inp) == 1 and isinstance(inp[0], list) and len(inp[0]) == 3: self._e_mat = get_qecc_e_from_paulis(*(inp[0]))
        elif len(inp) == 3: self._e_mat = get_qecc_e_from_paulis(*inp)
        else: raise ValueError(f"Input '{str(inp)}' is invalid!")
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

    @property
    def nkd(self) -> Tuple[int, int, int]:
        return (self.n, self.k, self.d)

    def _compute_distance(self):
        pass

def get_qecc_e_from_paulis(x_logicals, z_logicals, stabilizers) -> StabilizerCode:
    k = len(x_logicals)
    n = len(stabilizers) + k
    els = [pauli_string_to_list(i, n) for i in x_logicals + z_logicals + stabilizers]
    return StabilizerCode(np.array(els, dtype=ITYPE).T)

