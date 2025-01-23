from __future__ import annotations
from enum import Enum
from typing import Tuple, List, Optional, Union

from .symplectic_rep.clifford_gate import Clifford
from ._global_vars import ITYPE

import numpy as np
from numpy.typing import NDArray


def get_circuit(inp, qubits: int):
    return Circuit.get_circuit_from_string(inp, qubits)

# Identical to stim circuit language


class Operation(Enum):
    CZ = "CZ"
    CX = "CX"
    I = "I"
    S = "S"
    SDG = "SDG"
    H = "H"
    R = "C_ZYX"  # R = S H, X<-Y<-Z<-X
    R_DAG = "C_XYZ"  # R_DAG = H S_DAG, X->Y->Z->X
    SXDG = "SQRT_X_DAG"  # SQRT_X_DAG = H S_DAG H
    SWAP = "SWAP"
    BARRIER = ""
    X = "X"
    Y = "Y"
    Z = "Z"


def contract_single_qubit_clifford(ops: List[Operation]) -> List[Operation]:
    if len(ops) == 0:
        return [Operation.I]
    c = get_clifford_of_operation(ops[0], [0], 1)
    for op in ops[1:]:
        c = get_clifford_of_operation(op, [0], 1)@c
    circ = Circuit.get_SCL_as_circuit(c)
    return [i[0] for i in circ._gates]


def get_clifford_of_operation(op: Operation, ts: List[int], N: int):
    if op == Operation.CZ:
        assert (len(ts) == 2)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N, ts[1]] = m[ts[1]+N, ts[0]] = 1
        return Clifford.from_matrix(m)
    if op == Operation.SWAP:
        assert (len(ts) == 2)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0], ts[0]] = m[ts[0]+N, ts[0]+N] = 0
        m[ts[1], ts[1]] = m[ts[1]+N, ts[1]+N] = 0
        m[ts[0], ts[1]] = m[ts[0]+N, ts[1]+N] = 1
        m[ts[1], ts[0]] = m[ts[1]+N, ts[0]+N] = 1
        return Clifford.from_matrix(m)
    if op == Operation.I or op == Operation.BARRIER:
        m = np.identity(2*N, dtype=ITYPE)
        return Clifford.from_matrix(m)
    if op == Operation.S:
        assert (len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N, ts[0]] = 1
        return Clifford.from_matrix(m)
    if op == Operation.H:
        assert (len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N, ts[0]] = m[ts[0], ts[0]+N] = 1
        m[ts[0], ts[0]] = m[ts[0]+N, ts[0]+N] = 0
        return Clifford.from_matrix(m)
    if op == Operation.R:
        assert (len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N, ts[0]] = m[ts[0], ts[0]+N] = 1
        m[ts[0], ts[0]] = 0
        return Clifford.from_matrix(m)
    if op == Operation.R_DAG:
        assert (len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0]+N, ts[0]] = m[ts[0], ts[0]+N] = 1
        m[ts[0]+N, ts[0]+N] = 0
        return Clifford.from_matrix(m)
    if op == Operation.SXDG:
        assert (len(ts) == 1)
        m = np.identity(2*N, dtype=ITYPE)
        m[ts[0], ts[0]+N] = 1
        return Clifford.from_matrix(m)
    if op == Operation.X:
        assert (len(ts) == 1)
        p = np.zeros((2*N,), dtype=ITYPE)
        p[ts[0]+N] = 1
        return Clifford.from_matrix_with_phase(np.identity(2*N, dtype=ITYPE), p)
    if op == Operation.Y:
        assert (len(ts) == 1)
        p = np.zeros((2*N,), dtype=ITYPE)
        p[ts[0]+N] = p[ts[0]] = 1
        return Clifford.from_matrix_with_phase(np.identity(2*N, dtype=ITYPE), p)
    if op == Operation.Z:
        assert (len(ts) == 1)
        p = np.zeros((2*N,), dtype=ITYPE)
        p[ts[0]] = 1
        return Clifford.from_matrix_with_phase(np.identity(2*N, dtype=ITYPE), p)
    raise ValueError(f"Operation '{op.value}' not known!")


Gate = Tuple[Operation, List[int]]


class Circuit:
    @staticmethod
    def get_CZL_as_circuit(c: Clifford) -> Circuit:
        circ = Circuit(c.N)
        for i in range(c.N):
            for j in range(i+1, c.N):
                if c.symplectic_matrix[i+c.N, j] == 1:
                    circ.append((Operation.CZ, [i, j]))
        return circ

    @staticmethod
    def get_SCL_as_circuit(c: Clifford, only_paulis: bool = False) -> Circuit:
        circ = Circuit(c.N)
        for i in range(c.N):
            if c.phase[i] == 1:
                circ.append((Operation.Z, [i]))
            if c.phase[i+c.N] == 1:
                circ.append((Operation.X, [i]))
        if only_paulis:
            return circ
        for i in range(c.N):
            m = (c.symplectic_matrix[i, i], c.symplectic_matrix[i, i+c.N],
                 c.symplectic_matrix[i+c.N, i], c.symplectic_matrix[i+c.N, i+c.N])
            if m == (1, 0, 0, 1):
                pass
                # circ.append((Operation.I, [i]))
            elif m == (1, 0, 1, 1):
                circ.append((Operation.S, [i]))
            elif m == (1, 1, 0, 1):
                circ.append((Operation.SXDG, [i]))
            elif m == (0, 1, 1, 0):
                circ.append((Operation.H, [i]))
            elif m == (1, 1, 1, 0):
                circ.append((Operation.R_DAG, [i]))
            elif m == (0, 1, 1, 1):
                circ.append((Operation.R, [i]))
            else:
                raise ValueError(
                    f"Gate with signature {str(m)} at qubit {str(i)} is not Clifford!")
        return circ

    @staticmethod
    def get_perm_as_circuit(c: Clifford) -> Circuit:
        circ = Circuit(c.N)
        for i in range(c.N):
            if c.symplectic_matrix[i, i] == 0:
                for j in range(c.N):
                    if c.symplectic_matrix[j, i] == 1:
                        circ.append((Operation.SWAP, [i, j]))
                        c = c@get_clifford_of_operation(Operation.SWAP, [i, j], c.N)
                        break
        return circ

    @staticmethod
    def get_paulis_as_circuit(b: NDArray, invert: bool = False) -> Circuit:
        circ = Circuit(len(b)//2)
        o1, o2 = (0, len(b)//2) if invert else (len(b)//2, 0)
        for i in range(len(b)//2):
            if b[i+o1] == 1:
                circ.append((Operation.Z, [i]))
            if b[i+o2] == 1:
                circ.append((Operation.X, [i]))
        return circ

    @staticmethod
    def get_circuit_from_string(circ: str, n: int = -1) -> Circuit:
        c: List[Gate] = []
        m = 0
        try:
            for j, l in enumerate(circ.splitlines()):
                parts = l.strip().split()
                c.append((Operation(parts[0].upper()), [int(i) for i in parts[1:]]))
                m = max(max(c[-1][1])+1, m)
        except ValueError:
            raise ValueError(f"Invalid instruction in line {str(j)}: '{l}'")
        if n == -1:
            circuit = Circuit(m)
            circuit.append(c)
            return circuit
        else:
            if m > n:
                raise ValueError(
                    f"Circuit is defined on at least {m} qubits but only {n} were given")
            circuit = Circuit(n)
            circuit.append(c)
            return circuit

    @staticmethod
    def decompose_clifford(c: Clifford) -> Circuit:
        pass
    
    def __init__(self, num_qubits: int):
        self._num_qubits = num_qubits
        self._gates = []

    def h(self, qubit: int):
        self.append((Operation.H, [qubit]))

    def s(self, qubit: int):
        self.append((Operation.S, [qubit]))

    def sdg(self, qubit: int):
        self.append((Operation.SDG, [qubit]))

    def sxdg(self, qubit: int):
        self.append((Operation.SXDG, [qubit]))

    def cx(self, control: int, target: int):
        self.append((Operation.CX, [control, target]))

    def cz(self, qubit1: int, qubit2):
        self.append((Operation.CZ, [qubit1, qubit2]))

    def swap(self, qubit1: int, qubit2):
        self.append((Operation.SWAP, [qubit1, qubit2]))

    def x(self, qubit: int):
        self.append((Operation.X, [qubit]))

    def y(self, qubit: int):
        self.append((Operation.Y, [qubit]))

    def z(self, qubit: int):
        self.append((Operation.Z, [qubit]))

    def id(self, qubit: int):
        self.append((Operation.I, [qubit]))

    def __add__(self, other: Circuit) -> Circuit:
        assert (self.num_qubits == other.num_qubits)
        circuit = Circuit(self.num_qubits)
        circuit.append(self._gates + other._gates)
        return circuit

    @property
    def num_qubits(self):
        return self._num_qubits

    def shallow_optimize(self):
        # Contract single-qubit Cliffords
        for i in range(self.num_qubits):
            tars = [[]]
            ops = [[]]
            for j, gate in enumerate(self._gates):
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
                    self._gates.pop(j)
                if len(o) == 1 and o[0] == Operation.I:
                    continue
                for el in reversed(o):
                    self._gates.insert(ts[0], (el, [i]))
        # Collect Paulis

    def map_qubits(self, mp: dict, N: int) -> Circuit:
        nc = Circuit(N)
        for op, ts in self._gates:
            nc.append((op, [mp.get(i, i) for i in ts]))
        return nc

    def append(self, a: Union[Gate, List[Gate]]):
        if isinstance(a, list):
            self._gates += a
        else:
            self._gates.append(a)

    def insert(self, index: int, a: Gate):
        self._gates.insert(index, a)

    def __str__(self) -> str:
        s = ""
        for op, ts in self._gates:
            s += op.value
            for t in ts:
                s += f" {str(t)}"
            s += "\n"
        return s

    def to_qiskit(self):
        import qiskit
        QC = qiskit.QuantumCircuit

        gates = {
            Operation.X: QC.x, Operation.Y: QC.y, Operation.Z: QC.z, Operation.H: QC.h,
            Operation.SDG: QC.sdg, Operation.S: QC.s, Operation.CX: QC.cx,
            Operation.CZ: QC.cz, Operation.SXDG: QC.sxdg,
            Operation.SWAP: QC.swap, Operation.I: QC.id
        }

        circuit = QC(self.num_qubits)

        for op, qubits in self._gates:
            if op in gates:
                gates[op](circuit, *qubits)
            elif op == Operation.R_DAG:
                circuit.sdg(*qubits)
                circuit.h(*qubits)
            elif op == Operation.R:
                circuit.h(*qubits)
                circuit.s(*qubits)
            elif op == Operation.BARRIER:
                circuit.barrier()
            else:
                assert False, f"Unknown op {op}"
        return circuit

    def to_clifford(self) -> Clifford:
        c = Clifford.from_matrix(np.identity(2*self.num_qubits, dtype=ITYPE))
        for gate, ts in reversed(self._gates):
            c = c@get_clifford_of_operation(gate, ts, self.num_qubits)
        return c
