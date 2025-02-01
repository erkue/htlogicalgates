from typing import List, overload
import numpy as np
from numpy.typing import NDArray

from .resources.resources import load_connectivity
from ._utility import _argument_assignment


class Connectivity:
    @overload
    def __init__(self, name: str):
        """
        Constructs a connectivity object from a known connectivity name. 
        To query available name, call `available_connectivities()`.

        Parameters
        ----------
        name: str 
            Name of the connectivity
        """
        pass

    @overload
    def __init__(self, name: str, num_qubits: int):
        """
        Constructs a connectivity object by its name and number of qubits.
        To query available name, call `available_connectivities()`.

        Parameters
        ----------
        name: str 
            Name of the connectivity.
        num_qubits: int
            Number of qubits of the connectivity.
        """
        pass

    @overload
    def __init__(self, matrix: NDArray):
        """
        Constructs a connectivity object from a matrix. Here, `matrix[i,j] = 1` if qubits `i`
        and `j` are connected and `matrix[i,j] = 0` otherwise.

        Parameters
        ----------
        matrix: NDArray 
            Matrix of the connectivity.
        """
        pass

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
                raise ValueError(
                    "Connectivity() matrix input needs to be 2 dimensional")
            if sh[0] != sh[1]:
                raise ValueError(
                    "Connectivity() matrix input needs to be square")
        self._n = np.shape(self._mat)[0]

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits the connectivity is defined on.

        Returns
        ----------
        int
            Number of qubits.
        """
        return self._n

    @property
    def matrix(self) -> NDArray:
        """
        Returns the connectivity matrix. Here, `matrix[i,j] = 1` if qubits `i`
        and `j` are connected and `matrix[i,j] = 0` otherwise.

        Returns
        ----------
        NDArray
            Matrix of the connectivity.
        """
        return self._mat
