from functools import cache
import numpy as np
from numpy.typing import NDArray

from .._global_vars import ITYPE

def pauli_string_to_list(s : str, n : int):
    out = [0] * (2*n)
    ps = s.split()
    for p in ps:
        try:
            tar = int(p[1:])
        except ValueError:
            raise ValueError(f"Pauli string '{str(p[1:])}' could not be converted to Pauli! Unknown symbol '{p}'")
        if tar >= n:
            raise ValueError(f"Pauli string '{str(p[1:])}' could not be converted to Pauli! Index '{str(tar)}' to large for n={str(n)}.")
        if p[0].upper() == "X":
            out[tar] += 1
        elif p[0].upper() == "Y":
            out[tar] += 1
            out[tar+n] += 1
        elif p[0].upper() == "Z":
            out[tar+n] += 1
        else:
            raise ValueError(f"Pauli string '{str(p[1:])}' could not be converted to Pauli! Unknown symbol '{p}'")
    return [i%2 for i in out]

def _expand_mat(m : NDArray) -> NDArray:
    rows, columns = np.shape(m)
    m_bar = np.zeros((rows+1, columns+1), dtype=ITYPE)
    m_bar[:rows,:columns] = m
    m_bar[-1,-1] = 1
    return m_bar

def _expand_vec(v : NDArray) -> NDArray:
    d = len(v)
    v_bar = np.zeros((d+1,), dtype=ITYPE)
    v_bar[:d] = v
    return v_bar

@cache
def _get_u(n : int) -> NDArray:
        u = np.zeros((2*n,2*n))
        u[:n,n:2*n] = np.identity(n, dtype=ITYPE)
        return u

@cache
def _get_omega(n : int) -> NDArray:
    u = _get_u(n)
    return u + u.T

@cache
def _get_u_bar(n : int) -> NDArray:
    return _expand_mat(_get_u(n))

class LinSolver:
    @staticmethod
    def _get_row_adder(control : int, target : int, N : int):
        q = np.identity(N, dtype=ITYPE)
        q[target, control] = 1
        return q
    
    @staticmethod
    def _get_row_swapper(i : int, j : int, N : int):
        q = np.identity(N, dtype=ITYPE)
        q[i,i] = q[j,j] = 0
        q[i,j] = q[j,i] = 1
        return q

    def __init__(self, A : NDArray):
        self.N, self.M = np.shape(A)
        self.traf = np.identity(self.N, dtype=ITYPE)
        self.ids = []
        A = A.copy()
        row = 0
        for c in range(self.M):
            i = row
            while A[i,c] != 1:
                i += 1
                if i >= self.N:
                    i -= 1
                    break
            A = (LinSolver._get_row_swapper(i, row, self.N)@A)%2
            self.traf = (LinSolver._get_row_swapper(i, row, self.N)@self.traf)%2
            if A[row,c] == 1:
                for j in range(row+1, self.N):
                    if A[j,c] == 1:
                        A = (LinSolver._get_row_adder(row, j, self.N)@A)%2
                        self.traf = (LinSolver._get_row_adder(row, j, self.N)@self.traf)%2
                self.ids.append(c)
                row += 1
            if row >= self.N:
                break
        for i in reversed(range(self.N)):
            c = self.ids[i]
            for j in range(i):
                if A[j,c] == 1:
                    A = (LinSolver._get_row_adder(i, j, self.N)@A)%2
                    self.traf = (LinSolver._get_row_adder(i, j, self.N)@self.traf)%2
    
    def get_solution(self, b : NDArray) -> NDArray:
        q = (self.traf@b)%2
        s = np.zeros((self.M,), dtype=ITYPE)
        for i, e in enumerate(self.ids):
            s[e] = q[i]
        return s
                