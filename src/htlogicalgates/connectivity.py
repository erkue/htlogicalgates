from typing import List, overload
import numpy as np
from numpy.typing import NDArray

from .resources.resources import load_connectivity
from ._utility import _argument_assignment


class Connectivity:
    @overload
    def __init__(self, name: str): ...
    @overload
    def __init__(self, name: str, num_qubits: int): ...
    @overload
    def __init__(self, matrix: NDArray): ...

    def __init__(self, *args, **kwargs):
        options = [{"name": str},
                   {"name": str, "num_qubits": int},
                   {"matrix": np.ndarray}]
        i, a = _argument_assignment(
            options, "Connectivity()", *args, **kwargs)
        if i == 0:
            self._mat = load_connectivity(a["name"], None)
        elif i == 1:
            self._mat = load_connectivity(a["name"], a["num_qubits"])
        elif i == 2:
            self._mat = a["matrix"]
            if len(sh := np.shape(self._mat)) != 2:
                raise ValueError("Connectivity() matrix input needs to be 2 dimensional")
            if sh[0] != sh[1]:
                raise ValueError("Connectivity() matrix input needs to be square")
        self._n = np.shape(self._mat)[0]

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def matrix(self) -> NDArray:
        return self._mat


def get_con_mat_from_list(m: List[List[int]]) -> NDArray:
    return np.array(m, dtype=np.int32)
