from typing import Tuple, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

from .resources.resources import load_connectivity


class Connectivity:
    def __init__(self, inp, num_qubits : Optional[int] = None):
        if isinstance(inp, str): self._mat = get_con_mat_from_name(inp, num_qubits)
        elif isinstance(inp, list): self._mat = get_con_mat_from_list(inp)
        elif isinstance(inp, np.ndarray): self._mat = inp
        else: raise TypeError(f"Input of type '{str(type(inp))}' invalid!")
        if np.shape(self._mat)[0] != np.shape(self._mat)[1]:
            raise ValueError("Connectivity matrix must be a 2d square matrix!")
        self._n = len(self._mat)

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def matrix(self) -> NDArray:
        return self._mat


Conn = Connectivity

def get_con_mat_from_name(s: str, n: Optional[int] = None) -> NDArray:
    return load_connectivity(s, n)

def get_con_mat_from_list(m: List[List[int]]) -> NDArray:
    return np.array(m, dtype=np.int32)
