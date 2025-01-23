from __future__ import annotations
from enum import Enum
from typing import Tuple, List, Optional

from .symplectic_rep.clifford_gate import Clifford
from ._global_vars import ITYPE

import numpy as np
from numpy.typing import NDArray

def get_circuit(inp, qubits : int):
    return Circuit.get_circuit_from_string(inp, qubits)

# Identical to stim circuit language
class Operation(Enum):
    CZ = "CZ"
    I = "I"
    S = "S"
    H = "H"
    R = "C_ZYX" #R = S H, X<-Y<-Z<-X
    R_DAG = "C_XYZ" #R_DAG = H S_DAG, X->Y->Z->X
    SQRT_X_DAG = "SQRT_X_DAG" #SQRT_X_DAG = H S_DAG H
    SWAP = "SWAP"
    BARRIER = ""
    X = "X"
    Y = "Y"
    Z = "Z"

def contract_single_qubit_clifford(ops : List[Operation]) -> List[Operation]:
    if len(ops) == 0:
        return [Operation.I]
    c = get_clifford_of_operation(ops[0], [0], 1)
    for op in ops[1:]:
        c = get_clifford_of_operation(op, [0], 1)@c
    circ = Circuit.get_SCL_as_circuit(c)
    return [i[0] for i in circ._circ]
    

def get_clifford_of_operation(op : Operation, ts : List[int], N : int):
    if op == Operation.CZ:
        assert(len(ts) == 2)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N,ts[1]] = m[ts[1]+N,ts[0]] = 1
        return Clifford.from_matrix(m)
    if op == Operation.SWAP:
        assert(len(ts) == 2)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0],ts[0]] = m[ts[0]+N,ts[0]+N] = 0
        m[ts[1],ts[1]] = m[ts[1]+N,ts[1]+N] = 0
        m[ts[0],ts[1]] = m[ts[0]+N,ts[1]+N] = 1
        m[ts[1],ts[0]] = m[ts[1]+N,ts[0]+N] = 1
        return Clifford.from_matrix(m)
    if op == Operation.I or op == Operation.BARRIER:
        m = np.identity(2*N, dtype=ITYPE)
        return Clifford.from_matrix(m)
    if op == Operation.S:
        assert(len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N,ts[0]] = 1
        return Clifford.from_matrix(m)
    if op == Operation.H:
        assert(len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N,ts[0]] = m[ts[0],ts[0]+N] = 1
        m[ts[0],ts[0]] = m[ts[0]+N,ts[0]+N] = 0
        return Clifford.from_matrix(m)
    if op == Operation.R:
        assert(len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N,ts[0]] = m[ts[0],ts[0]+N] = 1
        m[ts[0],ts[0]] = 0
        return Clifford.from_matrix(m)
    if op == Operation.R_DAG:
        assert(len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N,ts[0]] = m[ts[0],ts[0]+N] = 1
        m[ts[0]+N,ts[0]+N] = 0
        return Clifford.from_matrix(m)
    if op == Operation.SQRT_X_DAG:
        assert(len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0],ts[0]+N] = 1
        return Clifford.from_matrix(m)
    if op == Operation.X:
        assert(len(ts) == 1)
        p = np.zeros((2*N,), dtype=ITYPE)
        p[ts[0]+N] = 1
        return Clifford.from_matrix_with_phase(np.identity(2*N, dtype=ITYPE), p)
    if op == Operation.Y:
        assert(len(ts) == 1)
        p = np.zeros((2*N,), dtype=ITYPE)
        p[ts[0]+N] = p[ts[0]] = 1
        return Clifford.from_matrix_with_phase(np.identity(2*N, dtype=ITYPE), p)
    if op == Operation.Z:
        assert(len(ts) == 1)
        p = np.zeros((2*N,), dtype=ITYPE)
        p[ts[0]] = 1
        return Clifford.from_matrix_with_phase(np.identity(2*N, dtype=ITYPE), p)
    raise ValueError(f"Operation '{op.value}' not known!")

Gate = Tuple[Operation, List[int]]

class Circuit:
    @staticmethod
    def get_CZL_as_circuit(c : Clifford) -> Circuit:
        circ = Circuit(c.N)
        for i in range(c.N):
            for j in range(i+1, c.N):
                if c.symplectic_matrix[i+c.N,j] == 1:
                    circ.append((Operation.CZ, [i,j]))
        return circ
    
    @staticmethod
    def get_SCL_as_circuit(c : Clifford, only_paulis : bool = False) -> Circuit:
        circ = Circuit(c.N, [])
        for i in range(c.N):
            if c.phase[i] == 1:
                circ.append((Operation.Z, [i]))
            if c.phase[i+c.N] == 1:
                circ.append((Operation.X, [i]))
        if only_paulis:
            return circ
        for i in range(c.N):
            m = (c.symplectic_matrix[i,i],c.symplectic_matrix[i,i+c.N],
                 c.symplectic_matrix[i+c.N,i],c.symplectic_matrix[i+c.N,i+c.N])
            if m == (1,0,0,1):
                pass
                #circ.append((Operation.I, [i]))
            elif m == (1,0,1,1):
                circ.append((Operation.S, [i]))
            elif m == (1,1,0,1):
                circ.append((Operation.SQRT_X_DAG, [i]))
            elif m == (0,1,1,0):
                circ.append((Operation.H, [i]))
            elif m == (1,1,1,0):
                circ.append((Operation.R_DAG, [i]))
            elif m == (0,1,1,1):
                circ.append((Operation.R, [i]))
            else:
                raise ValueError(f"Gate with signature {str(m)} at qubit {str(i)} is not Clifford!")
        return circ
    
    @staticmethod
    def get_perm_as_circuit(c : Clifford) -> Circuit:
        circ = Circuit(c.N, [])
        for i in range(c.N):
            if c.symplectic_matrix[i,i] == 0:
                for j in range(c.N):
                    if c.symplectic_matrix[j,i] == 1:
                        circ.append((Operation.SWAP, [i,j]))
                        c = c@get_clifford_of_operation(Operation.SWAP, [i,j], c.N)
                        break
        return circ
    
    @staticmethod
    def get_paulis_as_circuit(b : NDArray, invert : bool = False) -> Circuit:
        circ = Circuit(len(b)//2)
        o1, o2 = (0, len(b)//2) if invert else (len(b)//2, 0)
        for i in range(len(b)//2):
            if b[i+o1] == 1:
                circ.append((Operation.Z, [i]))
            if b[i+o2] == 1:
                circ.append((Operation.X, [i]))
        return circ
    
    @staticmethod
    def get_circuit_from_string(circ : str, n : int = -1) -> Circuit:
        c : List[Gate] = []
        m = 0
        try:
            for j, l in enumerate(circ.splitlines()):
                parts = l.strip().split()
                c.append((Operation(parts[0].upper()), [int(i) for i in parts[1:]]))
                m = max(max(c[-1][1])+1, m)   
        except ValueError:
            raise ValueError(f"Invalid instruction in line {str(j)}: '{l}'")
        if n == -1:
            return Circuit(m, c)
        else:
            if m > n:
                raise ValueError(f"Circuit is defined on at least {m} qubits but only {n} were given")
            return Circuit(n, c)

    @staticmethod
    def decompose_clifford(c : Clifford) -> Circuit:
        pass

    def __init__(self, n : int, circ : List[Gate] = None):
        if circ is None:
            self._circ = []
        else:
            self._circ = circ
        self._n = n

    def __add__(self, other : Circuit) -> Circuit:
        assert(self.N == other.N)
        return Circuit(self.N, self._circ + other._circ)

    @property
    def N(self):
        return self._n

    def shallow_optimize(self):
        ### Contract single-qubit Cliffords
        for i in range(self.N):
            tars = [[]]
            ops = [[]]
            for j, gate in enumerate(self._circ):
                if gate[1] == [i]:
                    tars[-1].append(j)
                    ops[-1].append(gate[0])
                elif len(gate) == 2 and i in gate[1]:
                    tars.append([])
                    ops.append([])
            for ts, os in zip(reversed(tars), reversed(ops)):
                if len(ts) == 0:
                    continue
                o = contract_single_qubit_clifford(os)
                for j in reversed(ts):
                    self._circ.pop(j)
                if len(o) == 1 and o[0] == Operation.I:
                    continue
                for el in reversed(o):
                    self._circ.insert(ts[0],(el,[i]))
        ### Collect Paulis

    def map_qubits(self, mp : dict, N : int) -> Circuit:
        nc = Circuit(N)
        for op, ts in self._circ:
            nc.append((op, [mp.get(i,i) for i in ts]))
        return nc


    def append(self, a : Gate):
        self._circ.append(a)

    def insert(self, index : int, a : Gate):
        self._circ.insert(index, a)
    
    def get_as_string(self) -> str:
        s = ""
        for op, ts in self._circ:
            s += op.value
            for t in ts:
                s += f" {str(t)}"
            s += "\n"
        return s
    
    def get_as_clifford(self) -> Clifford:
        c = Clifford.from_matrix(np.identity(2*self.N, dtype=ITYPE))
        for gate, ts in reversed(self._circ):
            c = c@get_clifford_of_operation(gate, ts, self.N)
        return c