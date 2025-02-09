from __future__ import annotations
from enum import Enum
from typing import Tuple, List, Union, overload
import numpy as np
from numpy.typing import NDArray

try:
    from qiskit import QuantumCircuit as qiskitQuantumCircuit
except ImportError:
    _has_qiskit = False
finally:
    _has_qiskit = True

from .symplectic_rep.clifford_gate import Clifford
from .symplectic_rep.helper import matrix_rank, get_AMG_decomposition
from ._utility import _argument_assignment, MissingOptionalLibraryError

# Identical to stim circuit language


class Operation(Enum):
    CZ = "CZ"
    CX = "CX"
    I = "I"
    S = "S"
    SDG = "SDG"
    H = "H"
    C_ZYX = "C_ZYX"  # S H, X<-Y<-Z<-X
    C_XYZ = "C_XYZ"  # H S_DAG, X->Y->Z->X
    SXDG = "SQRT_X_DAG"  # SQRT_X_DAG = H S_DAG H
    SWAP = "SWAP"
    BARRIER = ""
    X = "X"
    Y = "Y"
    Z = "Z"


Gate = Tuple[Operation, List[int]]


def contract_single_qubit_clifford(ops: List[Operation]) -> List[Operation]:
    if len(ops) == 0:
        return [Operation.I]
    c = gate_to_clifford(ops[0], [0], 1)
    for op in ops[1:]:
        c = gate_to_clifford(op, [0], 1) @ c
    circ = Circuit.from_SCL(c)
    return [i[0] for i in circ._gates]


def gate_to_clifford(op: Operation, qubits: List[int], num_qubits: int):
    n = num_qubits
    if op == Operation.CZ:
        assert (len(qubits) == 2)
        m = np.identity(2*n, dtype=np.int32)
        m[qubits[0]+n, qubits[1]] = m[qubits[1]+n, qubits[0]] = 1
        return Clifford(m)
    if op == Operation.CX:
        assert (len(qubits) == 2)
        m = np.identity(2*n, dtype=np.int32)
        m[qubits[1], qubits[0]] = m[qubits[0]+n, qubits[1]+n] = 1
        return Clifford(m)
    if op == Operation.SWAP:
        assert (len(qubits) == 2)
        m = np.identity(2*n, dtype=np.int32)
        m[qubits[0], qubits[0]] = m[qubits[0]+n, qubits[0]+n] = 0
        m[qubits[1], qubits[1]] = m[qubits[1]+n, qubits[1]+n] = 0
        m[qubits[0], qubits[1]] = m[qubits[0]+n, qubits[1]+n] = 1
        m[qubits[1], qubits[0]] = m[qubits[1]+n, qubits[0]+n] = 1
        return Clifford(m)
    if op == Operation.I or op == Operation.BARRIER:
        m = np.identity(2*n, dtype=np.int32)
        return Clifford(m)
    if op == Operation.S:
        assert (len(qubits) == 1)
        m = np.identity(2*n, dtype=np.int32)
        m[qubits[0]+n, qubits[0]] = 1
        return Clifford(m)
    if op == Operation.H:
        assert (len(qubits) == 1)
        m = np.identity(2*n, dtype=np.int32)
        m[qubits[0]+n, qubits[0]] = m[qubits[0], qubits[0]+n] = 1
        m[qubits[0], qubits[0]] = m[qubits[0]+n, qubits[0]+n] = 0
        return Clifford(m)
    if op == Operation.C_ZYX:
        assert (len(qubits) == 1)
        m = np.identity(2*n, dtype=np.int32)
        m[qubits[0]+n, qubits[0]] = m[qubits[0], qubits[0]+n] = 1
        m[qubits[0], qubits[0]] = 0
        return Clifford(m)
    if op == Operation.C_XYZ:
        assert (len(qubits) == 1)
        m = np.identity(2*n, dtype=np.int32)
        m[qubits[0]+n, qubits[0]] = m[qubits[0], qubits[0]+n] = 1
        m[qubits[0]+n, qubits[0]+n] = 0
        return Clifford(m)
    if op == Operation.SXDG:
        assert (len(qubits) == 1)
        m = np.identity(2*n, dtype=np.int32)
        m[qubits[0], qubits[0]+n] = 1
        return Clifford(m)
    if op == Operation.X:
        assert (len(qubits) == 1)
        p = np.zeros((2*n,), dtype=np.int32)
        p[qubits[0]+n] = 1
        return Clifford(np.identity(2*n, dtype=np.int32), p)
    if op == Operation.Y:
        assert (len(qubits) == 1)
        p = np.zeros((2*n,), dtype=np.int32)
        p[qubits[0]+n] = p[qubits[0]] = 1
        return Clifford(np.identity(2*n, dtype=np.int32), p)
    if op == Operation.Z:
        assert (len(qubits) == 1)
        p = np.zeros((2*n,), dtype=np.int32)
        p[qubits[0]] = 1
        return Clifford(np.identity(2*n, dtype=np.int32), p)
    raise ValueError(f"Operation '{op.value}' not known")


class Circuit:
    @overload
    def __init__(self, num_qubits: int):
        """
        Constructs an empty circuit.

        Parameters
        ----------
        num_qubits: int
            Number of qubits the circuit acts on.
        """
        pass

    @overload
    def __init__(self, init_string: str):
        """
        Constructs a circuit from a string.

        Parameters
        ----------
        init_string: str
            String containing the circuit.

        Examples
        ----------
            >>> Circuit("H 0\\nS 1\\nCZ 0 1")

        """
        pass

    @overload
    def __init__(self, init_string: str, num_qubits: int):
        """
        Constructs a circuit from a string defined on a number of qubits. The number of
        qubits must be large enough to contain all gates from the string.

        Parameters
        ----------
        init_string: str
            String containing the circuit.
        num_qubits: int
            Number of qubits the circuit acts on.

        Examples
        ----------
            >>> Circuit("H 0\\nS 1\\nCZ 0 1", 4)
        """
        pass

    @overload
    def __init__(self, cliff: Clifford):
        """
        Construct a circuit from a Clifford gate using the algorithm
        from <https://arxiv.org/pdf/quant-ph/0406196>.

        Parameters
        ----------
        cliff: Clifford
            Clifford element to create a circuit from.
        """
        pass

    def __init__(self, *args, **kwargs):
        options = [{"num_qubits": int},
                   {"init_string": str},
                   {"init_string": str, "num_qubits": int},
                   {"cliff": Clifford}]
        i, a = _argument_assignment(
            options, "Circuit()", *args, **kwargs)
        if i == 0:
            self._num_qubits = a["num_qubits"]
            self._gates: List[Gate] = []
        elif i == 1 or i == 2:
            self._gates: List[Gate] = []
            m = 0
            try:
                for j, l in enumerate(a["init_string"].splitlines()):
                    l = l.split("#", 1)[0]
                    parts = l.strip().split()
                    if len(parts) == 0:
                        continue
                    elif len(parts) == 1:
                        for operation in Operation:
                            if operation.value in parts[0].upper():
                                self._gates.append(
                                    (operation, [int(parts[0].upper().replace(operation.value, ""))]))
                                m = max(max(self._gates[-1][1])+1, m)
                                break
                    else:
                        self._gates.append((Operation(parts[0].upper()), [
                            int(i) for i in parts[1:]]))
                        m = max(max(self._gates[-1][1])+1, m)
            except ValueError:
                raise ValueError(
                    f"Invalid instruction in line {str(j)}: '{l}'")
            if i == 2:
                self._num_qubits = a["num_qubits"]
                if m > a["num_qubits"]:
                    raise ValueError(
                        f"Circuit is defined on at least '{m}' qubits but only '{a['num_qubits']}' were given")
            else:
                self._num_qubits = m
        elif i == 3:
            # from <https://arxiv.org/pdf/quant-ph/0406196> Theorem 8
            cliff: Clifford = a["cliff"]
            self._num_qubits = cliff.num_qubits
            self._gates: List[Gate] = []
            n = cliff.num_qubits

            # Step 1
            def get_rank(x): return matrix_rank(
                x.symplectic_matrix[0:n, n:2*n])
            prev_rank = get_rank(cliff)
            for i in range(n):
                h = gate_to_clifford(Operation.H, [i], n)
                if (new_rank := get_rank(h@cliff)) > prev_rank:
                    cliff = h@cliff
                    self.insert(0, (Operation.H, [i]))
                    prev_rank = new_rank

            # Step 2
            for i in range(n):
                if cliff.symplectic_matrix[0:n, n:2*n][i, i] % 2 == 0:
                    for j in range(i+1, n):
                        if cliff.symplectic_matrix[0:n, n:2*n][j, i] % 2 == 1:
                            cliff = gate_to_clifford(
                                Operation.CX, [j, i], n)@cliff
                            self.insert(0, (Operation.CX, [j, i]))
                            break
                for j in range(i+1, n):
                    if cliff.symplectic_matrix[0:n, n:2*n][j, i] % 2 == 1:
                        cliff = gate_to_clifford(Operation.CX, [i, j], n)@cliff
                        self.insert(0, (Operation.CX, [i, j]))
            for i in reversed(range(n)):
                for j in range(0, i):
                    if cliff.symplectic_matrix[0:n, n:2*n][j, i] % 2 == 1:
                        cliff = gate_to_clifford(Operation.CX, [i, j], n)@cliff
                        self.insert(0, (Operation.CX, [i, j]))

            # Step 3
            M, G = get_AMG_decomposition(cliff.symplectic_matrix[n:2*n, n:2*n])
            for i in range(n):
                if G[i, i] % 2 == 1:
                    cliff = gate_to_clifford(Operation.S, [i], n)@cliff
                    self.insert(0, (Operation.S, [i]))

            # Step 4
            for i in range(n):
                for j in range(0, i):
                    if M[i, j] % 2 == 1:
                        cliff = gate_to_clifford(Operation.CX, [i, j], n)@cliff
                        self.insert(0, (Operation.CX, [i, j]))

            # Step 5
            for i in range(n):
                cliff = gate_to_clifford(Operation.S, [i], n)@cliff
                self.insert(0, (Operation.S, [i]))

            # Step 6
            for i in reversed(range(n)):
                for j in range(0, i):
                    if cliff.symplectic_matrix[:n, n:2*n][j, i]:
                        cliff = gate_to_clifford(Operation.CX, [i, j], n)@cliff
                        self.insert(0, (Operation.CX, [i, j]))

            # Step 7
            for i in range(n):
                cliff = gate_to_clifford(Operation.H, [i], n)@cliff
                self.insert(0, (Operation.H, [i]))

            # Step 8
            M, G = get_AMG_decomposition(cliff.symplectic_matrix[n:2*n, :n])
            for i in range(n):
                if G[i, i] % 2 == 1:
                    cliff = gate_to_clifford(Operation.S, [i], n)@cliff
                    self.insert(0, (Operation.S, [i]))

            # Step 9
            for i in range(n):
                for j in range(0, i):
                    if M[i, j] % 2 == 1:
                        cliff = gate_to_clifford(Operation.CX, [i, j], n)@cliff
                        self.insert(0, (Operation.CX, [i, j]))

            # Step 10
            for i in range(n):
                cliff = gate_to_clifford(Operation.S, [i], n)@cliff
                self.insert(0, (Operation.S, [i]))

            # Step 11
            for i in reversed(range(n)):
                for j in range(0, i):
                    if M[i, j] % 2 == 1:
                        cliff = gate_to_clifford(Operation.CX, [i, j], n)@cliff
                        self.insert(0, (Operation.CX, [i, j]))

            # Fix Pauli frame
            circuit_clifford = self.to_clifford()
            target_clifford = a["cliff"]
            phases = (circuit_clifford.phase + target_clifford.phase) % 2
            for i in range(n):
                if phases[i] == 1:
                    self.insert(0, (Operation.Z, [i]))
                if phases[i + n] % 2 == 1:
                    self.insert(0, (Operation.X, [i]))

    def h(self, qubit: int):
        """
        Adds a Hadamard gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.H, [qubit]))

    def s(self, qubit: int):
        """
        Adds a phase gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.S, [qubit]))

    def sdg(self, qubit: int):
        """
        Adds the adjoint of the phase gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.SDG, [qubit]))

    def sxdg(self, qubit: int):
        """
        Adds a sqrt(X)=(h sdg h) gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.SXDG, [qubit]))

    def c_xyz(self, qubit: int):
        """
        Adds a c_xyz=(h sdg) gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.C_XYZ, [qubit]))

    def c_zyx(self, qubit: int):
        """
        Adds a c_zyx=(s h) gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.C_ZYX, [qubit]))

    def x(self, qubit: int):
        """
        Adds a Pauli-X gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.X, [qubit]))

    def y(self, qubit: int):
        """
        Adds a Pauli-Y gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.Y, [qubit]))

    def z(self, qubit: int):
        """
        Adds a Pauli-Z gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.Z, [qubit]))

    def id(self, qubit: int):
        """
        Adds an identity gate to the end of circuit.

        Parameters
        ----------
        qubit: int
            Qubit the gate acts on.
        """
        self.append((Operation.I, [qubit]))

    def cx(self, control: int, target: int):
        """
        Adds a controlled-X gate to the end of circuit.

        Parameters
        ----------
        control: int
            Control qubit.
        target: int
            Target qubit.
        """
        self.append((Operation.CX, [control, target]))

    def cz(self, qubit1: int, qubit2):
        """
        Adds a controlled-Z gate to the end of circuit.

        Parameters
        ----------
        qubit1: int
            First qubit the gate acts on.
        qubit2: int
            Second qubit the gate acts on.
        """
        self.append((Operation.CZ, [qubit1, qubit2]))

    def swap(self, qubit1: int, qubit2):
        """
        Adds a swapping operation to the end of circuit.

        Parameters
        ----------
        qubit1: int
            First qubit the gate acts on.
        qubit2: int
            Second qubit the gate acts on.
        """
        self.append((Operation.SWAP, [qubit1, qubit2]))

    def __add__(self, other: Circuit) -> Circuit:
        assert (self.num_qubits == other.num_qubits)
        circuit = Circuit(self.num_qubits)
        circuit.append(self._gates + other._gates)
        return circuit

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits the Clifford is defined on.

        Returns
        ----------
        int
            Number of qubits.
        """
        return self._num_qubits

    def shallow_optimize(self):
        """
        Performs a small optimization of single-qubit gates of the circuit.
        This operation is in-place. 
        """
        for i in range(self.num_qubits):
            targets = [[]]
            ops = [[]]
            for j, gate in enumerate(self._gates):
                if gate[1] == [i]:
                    targets[-1].append(j)
                    ops[-1].append(gate[0])
                elif len(gate) == 2 and i in gate[1]:
                    targets.append([])
                    ops.append([])
            for qubits, os in zip(reversed(targets), reversed(ops)):
                if len(qubits) == 0:
                    continue
                o = contract_single_qubit_clifford(os)
                for j in reversed(qubits):
                    self._gates.pop(j)
                if len(o) == 1 and o[0] == Operation.I:
                    continue
                for el in reversed(o):
                    self._gates.insert(qubits[0], (el, [i]))
        # Collect Paulis

    def permuted(self, map: dict, num_qubits: Union[int, None] = None) -> Circuit:
        if num_qubits is None:
            num_qubits = self.num_qubits
        circuit = Circuit(num_qubits)
        for op, qubits in self._gates:
            circuit.append((op, [map.get(i, i) for i in qubits]))
        return circuit

    def two_qubit_gate_count(self) -> int:
        """
        Returns the number of two-qubit gates in the circuit.

        Returns
        ----------
        int
            Number of two-qubit gates.
        """
        i = 0
        for op, qubits in self._gates:
            if len(qubits) == 2:
                i += 1
        return i

    def gate_count(self) -> int:
        """
        Returns the number of gates in the circuit.

        Returns
        ----------
        int
            Number of gates.
        """
        return len(self._gates)

    def append(self, gate: Union[Gate, List[Gate]]):
        """
        Adds one or multiple gates to the circuit.
        In order to compose two circuits, use `Circuit("H 0") + Circuit("S 0")`.

        Parameters
        ----------
        gate: Union[Gate, List[Gate]]
            Gate or list of gates to add to the end of the circuit.
        """
        if isinstance(gate, list):
            self._gates += gate
        else:
            self._gates.append(gate)

    def insert(self, index: int, gate: Gate):
        """
        Adds a gate to a specified position in the circuit.
        In order to compose two circuits, use `Circuit("H 0") + Circuit("S 0")`.

        Parameters
        ----------
        index: int
            Position of the gate.
        gate: Gate
            Gate to add to the circuit.
        """
        self._gates.insert(index, gate)

    def __str__(self) -> str:
        s = ""
        for op, qubits in self._gates:
            s += op.value
            for t in qubits:
                s += f" {str(t)}"
            s += "\n"
        return s

    def to_qiskit(self):
        """
        Transforms the circuit into a qiskit circuit.

        Raises

        Returns
        ----------
        qiskit.QuantumCircuit
            Circuit as QuantumCircuit from qiskit.

        Raises
        ----------
        MissingOptionalLibraryError
            If qiskit is not installed
        """
        if not _has_qiskit:
            raise MissingOptionalLibraryError(
                "to_qiskit() requires 'qiskit' to be installed")
        QC = qiskitQuantumCircuit
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
            elif op == Operation.C_XYZ:
                circuit.sdg(*qubits)
                circuit.h(*qubits)
            elif op == Operation.C_ZYX:
                circuit.h(*qubits)
                circuit.s(*qubits)
            elif op == Operation.BARRIER:
                circuit.barrier()
            else:
                assert False, f"Unknown op {op}"
        return circuit

    def to_clifford(self) -> Clifford:
        """
        Returns the circuit as a Clifford object.

        Returns
        ----------
        Clifford
            Clifford object of the circuit.
        """
        c = Clifford(np.identity(
            2*self.num_qubits, dtype=np.int32))
        for gate, ts in reversed(self._gates):
            c = c@gate_to_clifford(gate, ts, self.num_qubits)
        return c

    @staticmethod
    def from_CZL(clifford: Clifford) -> Circuit:
        circ = Circuit(clifford.num_qubits)
        for i in range(clifford.num_qubits):
            for j in range(i+1, clifford.num_qubits):
                if clifford.symplectic_matrix[i+clifford.num_qubits, j] == 1:
                    circ.append((Operation.CZ, [i, j]))
        return circ

    @staticmethod
    def from_SCL(clifford: Clifford, only_paulis: bool = False) -> Circuit:
        circ = Circuit(clifford.num_qubits)
        for i in range(clifford.num_qubits):
            if clifford.phase[i] == 1:
                circ.append((Operation.Z, [i]))
            if clifford.phase[i+clifford.num_qubits] == 1:
                circ.append((Operation.X, [i]))
        if only_paulis:
            return circ
        for i in range(clifford.num_qubits):
            m = (clifford.symplectic_matrix[i, i], clifford.symplectic_matrix[i, i+clifford.num_qubits],
                 clifford.symplectic_matrix[i+clifford.num_qubits, i], clifford.symplectic_matrix[i+clifford.num_qubits, i+clifford.num_qubits])
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
                circ.append((Operation.C_XYZ, [i]))
            elif m == (0, 1, 1, 1):
                circ.append((Operation.C_ZYX, [i]))
            else:
                raise ValueError(
                    f"Gate with signature {str(m)} at qubit {str(i)} is not Clifford")
        return circ

    @staticmethod
    def from_permutation(clifford: Clifford) -> Circuit:
        circ = Circuit(clifford.num_qubits)
        for i in range(clifford.num_qubits):
            if clifford.symplectic_matrix[i, i] == 0:
                for j in range(clifford.num_qubits):
                    if clifford.symplectic_matrix[j, i] == 1:
                        circ.append((Operation.SWAP, [i, j]))
                        clifford = clifford @ gate_to_clifford(
                            Operation.SWAP, [i, j], clifford.num_qubits
                        )
                        break
        return circ

    @staticmethod
    def from_paulis(paulis: NDArray, invert: bool = False) -> Circuit:
        circ = Circuit(len(paulis) // 2)
        o1, o2 = (0, len(paulis) // 2) if invert else (len(paulis) // 2, 0)
        for i in range(len(paulis) // 2):
            if paulis[i + o1] == 1:
                circ.append((Operation.Z, [i]))
            if paulis[i + o2] == 1:
                circ.append((Operation.X, [i]))
        return circ
