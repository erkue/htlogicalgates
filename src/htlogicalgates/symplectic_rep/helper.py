from functools import cache
import numpy as np
from numpy.typing import NDArray
import copy
from typing import Tuple, List


def matrix_rank(A: NDArray) -> int:
    """ 
    Computes the rank of a matrix A with coefficients in GF(2).
    Taken with permission from package `htstabilizer`
    """
    if (rows := np.shape(A)[0]) != (columns := np.shape(A)[1]):
        nA = np.zeros([max(rows, columns)]*2, dtype=A.dtype)
        nA[0:rows, 0:columns] = A
    else:
        nA = A

    def rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Computes the reduced-row echelon form of a matrix A 
        with entries in the binary field GF(2). 

        Parameters
        ----------
        A : np.ndarray
            Input matrix, should only contain elements 0 or 1 as integers. 

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            RREF of input matrix A as well as column indices of the pivot elements. 
        """
        m, n = A.shape  # m: rows, n: cols
        A = A % 2
        pivot_cols = []
        h = 0
        k = 0
        while h < m and k < n:
            found = False
            i = h
            while not found and i < m:
                if A[i, k] == 1:
                    found = True
                    break
                i += 1
            if not found:
                k += 1
            else:
                pivot_cols.append(k)
                temp = copy.deepcopy(A[h, :])
                A[h, :] = A[i, :]
                A[i, :] = temp
                for i in list(range(h)) + list(range(h+1, m)):
                    A[i, :] = (A[i, :] + A[i, k]*A[h, :]) % 2
                h += 1
                k += 1
        return (A, pivot_cols)

    return len(rref(nA)[1])


def pauli_string_to_list(s: str, n: int) -> List[int]:
    out = [0] * (2*n)
    for p in s.split():
        try:
            tar = int(p[1:])
        except ValueError:
            raise ValueError(
                f"Pauli string '{str(p[1:])}' could not be converted to Pauli! Unknown symbol '{p}'")
        if tar >= n:
            raise ValueError(
                f"Pauli string '{str(p[1:])}' could not be converted to Pauli! Index '{str(tar)}' to large for n={str(n)}.")
        if p[0].upper() == "X":
            out[tar] += 1
        elif p[0].upper() == "Y":
            out[tar] += 1
            out[tar+n] += 1
        elif p[0].upper() == "Z":
            out[tar+n] += 1
        else:
            raise ValueError(
                f"Pauli string '{str(p[1:])}' could not be converted to Pauli! Unknown symbol '{p}'")
    return [i % 2 for i in out]


def max_index_of_pauli(s: str) -> int:
    m = -1
    for p in s.split():
        try:
            tar = int(p[1:])
        except ValueError:
            raise ValueError(
                f"Pauli string '{str(p[1:])}' could not be converted to Pauli! Unknown symbol '{p}'")
        m = max(m, tar)
    return m + 1


def int_to_bitarray(i: int, l: int) -> NDArray:
    bitstring = (r"{0:"+str(l)+r"b}").format(i)
    if l < len(bitstring):
        raise ValueError(
            f"Integer '{str(i)}' too large for bitstring of length '{str(l)}'")
    return np.array([1 if j == "1" else 0 for j in bitstring], dtype=np.int32)


def bitarray_to_int(bitarray: NDArray) -> int:
    string = ""
    for i in bitarray:
        string += "1" if i == 1 else "0"
    return int(string, 2)


def expand_mat_once(m: NDArray) -> NDArray:
    rows, columns = np.shape(m)
    m_bar = np.zeros((rows+1, columns+1), dtype=np.int32)
    m_bar[:rows, :columns] = m
    m_bar[-1, -1] = 1
    return m_bar


def expand_vec_once(v: NDArray) -> NDArray:
    d = len(v)
    v_bar = np.zeros((d+1,), dtype=np.int32)
    v_bar[:d] = v
    return v_bar


@cache
def get_u_matrix(n: int) -> NDArray:
    u = np.zeros((2*n, 2*n))
    u[:n, n:2*n] = np.identity(n, dtype=np.int32)
    return u


@cache
def get_omega_matrix(n: int) -> NDArray:
    u = get_u_matrix(n)
    return u + u.T


@cache
def get_u_bar_matrix(n: int) -> NDArray:
    return expand_mat_once(get_u_matrix(n))


def get_AMG_decomposition(A) -> Tuple[NDArray, NDArray]:
    # from <https://arxiv.org/pdf/quant-ph/0406196> Lemma 7
    n = len(A)
    M = np.eye(n, n, dtype=np.int32)
    G = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i):
            M[i, j] = (A[i, j] + np.dot(M[i, :j], M[j, :j]))%2
    return M, (M@M.T + A)%2


class LinSolver:
    @staticmethod
    def _get_row_adder(control: int, target: int, N: int):
        q = np.identity(N, dtype=np.int32)
        q[target, control] = 1
        return q

    @staticmethod
    def _get_row_swapper(i: int, j: int, N: int):
        q = np.identity(N, dtype=np.int32)
        q[i, i] = q[j, j] = 0
        q[i, j] = q[j, i] = 1
        return q

    def __init__(self, A: NDArray):
        self.N, self.M = np.shape(A)
        self.traf = np.identity(self.N, dtype=np.int32)
        self.ids = []
        A = A.copy()
        row = 0
        for c in range(self.M):
            i = row
            while A[i, c] != 1:
                i += 1
                if i >= self.N:
                    i -= 1
                    break
            A = (LinSolver._get_row_swapper(i, row, self.N)@A) % 2
            self.traf = (LinSolver._get_row_swapper(
                i, row, self.N)@self.traf) % 2
            if A[row, c] == 1:
                for j in range(row+1, self.N):
                    if A[j, c] == 1:
                        A = (LinSolver._get_row_adder(row, j, self.N)@A) % 2
                        self.traf = (LinSolver._get_row_adder(
                            row, j, self.N)@self.traf) % 2
                self.ids.append(c)
                row += 1
            if row >= self.N:
                break
        for i in reversed(range(self.N)):
            c = self.ids[i]
            for j in range(i):
                if A[j, c] == 1:
                    A = (LinSolver._get_row_adder(i, j, self.N)@A) % 2
                    self.traf = (LinSolver._get_row_adder(
                        i, j, self.N)@self.traf) % 2

    def get_solution(self, b: NDArray) -> NDArray:
        q = (self.traf@b) % 2
        s = np.zeros((self.M,), dtype=np.int32)
        for i, e in enumerate(self.ids):
            s[e] = q[i]
        return s
