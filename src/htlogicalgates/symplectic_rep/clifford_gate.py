from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .helper import _expand_mat, _get_u, _get_u_bar, _expand_vec


class Clifford:
    def __init__(self, mat: NDArray, phase: NDArray):
        self._mat = mat.astype(np.int32)
        self._mat %= 2
        self._phase = phase.astype(np.int32)
        self._phase %= 2
        self._N = (len(mat) - 1) // 2
        self._M = len(phase) - 1

    @staticmethod
    def from_matrix(mat: NDArray) -> Clifford:  # TODO: Add assert
        e = _expand_mat(mat)
        e[-1, :-1] = Clifford._construct_imag_phase(mat)
        return Clifford(e, np.zeros((len(e[0]),), dtype=np.int32))

    @staticmethod
    # TODO: Add assert
    def from_matrix_with_phase(mat: NDArray, phase: NDArray) -> Clifford:
        e = _expand_mat(mat)
        e[-1, :-1] = Clifford._construct_imag_phase(mat)
        return Clifford(e, _expand_vec(phase))

    @property
    def symplectic_matrix(self) -> NDArray:
        return self._mat[:-1, :-1]  # TODO: Maybe copy

    @property
    def phase(self) -> NDArray:
        return self._phase[:-1]  # TODO: Maybe copy

    @property
    def num_qubits(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return self._M

    def is_proper(self) -> bool:
        return self.num_qubits == self.m / 2

    def __matmul__(self, other: Clifford) -> Clifford:
        assert self.m == 2 * other.num_qubits, "Cannot multiply Cliffords due to non-matching dimensions"
        phase = other._phase + other._mat.T @ self._phase
        phase += np.diag(other._mat.T @ np.tril(self._mat.T @
                         _get_u_bar(self.num_qubits) @ self._mat, -1) @ other._mat)
        return Clifford((self._mat @ other._mat) % 2, phase % 2)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Clifford):
            return False
        return bool(np.all(self.symplectic_matrix == other.symplectic_matrix) and np.all(self.phase == other.phase))

    @staticmethod
    def _construct_imag_phase(m: NDArray) -> NDArray:
        return np.diag(m.T @ _get_u(len(m) // 2) @ m) % 2
