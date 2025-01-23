from typing import Tuple, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

from .resources.resources import load_connectivity, available_connectivities
from ._global_vars import ITYPE


class Connectivity:
    def __init__(self, mat: NDArray):
        if len(np.shape(mat)) != 2:
            raise ValueError("Connectivity matrix must be a 2d square matrix!")
        if np.shape(mat)[0] != np.shape(mat)[1]:
            raise ValueError("Connectivity matrix must be a 2d square matrix!")
        self._mat = mat
        self._n = len(self._mat)

    @property
    def n(self) -> int:
        return self._n

    @property
    def matrix(self) -> NDArray:
        return self._mat


Conn = Connectivity


def get_conn(inp, n: Optional[int] = None) -> Connectivity:
    if isinstance(inp, str):
        return get_con_from_name(inp, n)
    if isinstance(inp, list):
        return get_con_from_list(inp)
    if isinstance(inp, np.ndarray):
        return get_con_from_matrix(inp)
    raise TypeError(f"Input of type '{str(type(inp))}' invalid!")


def get_con_from_name(s: str, n: Optional[int] = None) -> Connectivity:
    return get_con_from_matrix(load_connectivity(s, n))


def get_con_from_matrix(m: NDArray) -> Connectivity:
    return Connectivity(m)


def get_con_from_list(m: List[List[int]]) -> Connectivity:
    return get_con_from_matrix(np.array(m, dtype=ITYPE))
