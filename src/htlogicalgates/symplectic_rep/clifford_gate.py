from __future__ import annotations
from typing import overload, Optional

import numpy as np
from numpy.typing import NDArray

from .._utility import _argument_assignment
from .helper import expand_mat_once, get_u_matrix, get_u_bar_matrix, expand_vec_once, int_to_bitarray, bitarray_to_int
from .integer_symplectic import symplectic_matrix, symplectic_matrix_inverse


class Clifford:
    @overload
    def __init__(self, id: int, num_qubits: int):
        """
        Constructs a Clifford gate without Paulis by its integer representation.

        Parameters
        ----------
        id: int 
            Unique clifford id representing the Clifford part modulo Pauli gates.
        num_qubits: int
            Number of qubits the Clifford is defined on.
        """
        pass

    @overload
    def __init__(self, id: int, pauli_id: int, num_qubits: int):
        """
        Constructs a Clifford gate by its integer representation.

        Parameters
        ----------
        id: int
            Unique clifford id representing the Clifford part modulo Pauli gates.
        pauli_id: int
            Integer representing the Pauli part of the Clifford gate.
        num_qubits: int
            Number of qubits the Clifford is defined on.
        """
        pass

    @overload
    def __init__(self, matrix: NDArray):
        """
        Constructs a Clifford gate from a stabilizer tableau.

        Parameters
        ----------
        matrix: NDArray
            Stabilizer tableau of the Clifford operator.
        """
        pass

    @overload
    def __init__(self, matrix: NDArray, phase: NDArray):
        """
        Constructs a Clifford gate from a stabilizer tableau and a phase vector.

        Parameters
        ----------
        matrix: NDArray
            Stabilizer tableau of the Clifford operator.
        phase: NDArray
            Vector holding the phases under transformation of Pauli operators.
        """
        pass

    def __init__(self, *args, **kwargs):
        options = [{"id": int, "pauli_id": int, "num_qubits": int},
                   {"id": int, "num_qubits": int},
                   {"matrix": np.ndarray},
                   {"matrix": np.ndarray, "phase": np.ndarray},
                   {"matrix": np.ndarray, "phase": np.ndarray, "extended": bool}]
        i, a = _argument_assignment(
            options, "Clifford()", *args, **kwargs)
        if i == 0:
            mat = symplectic_matrix(a["id"], a["num_qubits"])
            phase = int_to_bitarray(a["pauli_id"], 2*a["num_qubits"])
        if i == 1:
            mat = symplectic_matrix(a["id"], a["num_qubits"])
            phase = np.zeros(2*a["num_qubits"], dtype=np.int32)
        if i == 2 or i == 3:
            mat = a["matrix"]
            if i == 2:
                phase = np.zeros(len(mat[0]), dtype=np.int32)
            else:
                phase = a["phase"]
        if i == 4:
            mat = a["matrix"]
            phase = a["phase"].astype(np.int32) % 2
        if not a.get("extended", False):
            self._mat = expand_mat_once(mat)
            self._mat[-1, :-1] = _construct_imag_phase(mat).astype(np.int32)
            self._phase = expand_vec_once(phase).astype(np.int32)
        else:
            self._mat = mat.astype(np.int32)
            self._phase = phase.astype(np.int32)
        self._mat %= 2
        self._phase %= 2
        self._N = (len(self._mat) - 1) // 2
        self._M = len(self._phase) - 1

    @property
    def id(self) -> int:
        """
        Returns an integer identifying the pure Clifford part of the Clifford gate.

        Returns
        -------
        int
            Integer representing the Clifford part.
        """
        if 2 * self.num_qubits != self.m:
            raise AttributeError("Must be a valid Clifford element")
        return symplectic_matrix_inverse(self._mat[:-1, :-1], self.num_qubits)

    @property
    def pauli_id(self) -> int:
        """
        Returns an integer identifying the Pauli part of the Clifford gate.

        Returns
        -------
        int
            Integer representing the Pauli part.
        """
        return bitarray_to_int(self._phase[:-1])

    @property
    def symplectic_matrix(self) -> NDArray:
        """
        Returns the symplectic stabilizer tableau of the Clifford gate.

        Returns
        -------
        NDArray
            The stabilizer tableau.
        """
        return self._mat[:-1, :-1]

    @property
    def phase(self) -> NDArray:
        """
        Returns a vector representing the phases of transformed Pauli operators.

        Returns
        -------
        NDArray
            Vector x containing the phases (-1)^x.
        """
        return self._phase[:-1]

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits the Clifford operation is defined on.

        Returns
        -------
        int
            Number of qubits.
        """
        return self._N

    @property
    def m(self) -> int:
        return self._M

    def is_proper(self) -> bool:
        return self.num_qubits == self.m / 2

    def __matmul__(self, other: Clifford) -> Clifford:
        assert self.m == 2 * \
            other.num_qubits, "Cannot multiply Cliffords due to non-matching dimensions"
        phase = other._phase + other._mat.T @ self._phase
        phase += np.diag(other._mat.T @ np.tril(self._mat.T @
                         get_u_bar_matrix(self.num_qubits) @ self._mat, -1) @ other._mat)
        return Clifford((self._mat @ other._mat) % 2, phase % 2, extended=True)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Clifford):
            return False
        return bool(np.all(self.symplectic_matrix == other.symplectic_matrix) and np.all(self.phase == other.phase))

    @staticmethod
    def group_size_mod_pauli(num_qubits: int):
        u = 1
        for i in range(1, num_qubits + 1):
            u *= 4**i - 1
        return 2 ** (num_qubits * num_qubits) * u


def _construct_imag_phase(m: NDArray) -> NDArray:
    return np.diag(m.T @ get_u_matrix(len(m) // 2) @ m) % 2
