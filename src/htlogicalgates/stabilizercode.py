import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, overload
from copy import deepcopy
from itertools import chain, combinations
from qsalto import M as MacWilliams

from .symplectic_rep.helper import pauli_string_to_list, max_index_of_pauli
from .resources.resources import load_stabilizercode
from .symplectic_rep.helper import matrix_rank
from ._utility import _argument_assignment


class StabilizerCode:
    @overload
    def __init__(self, name: str): ...
    @overload
    def __init__(self, name: str, num_qubits: int): ...

    @overload
    def __init__(self, paulis: Tuple[List[str], List[str],
                 List[str]], skip_tests: bool = False): ...

    @overload
    def __init__(self, x_logicals: List[str], z_logicals: List[str],
                 stabilizers: List[str], skip_tests: bool = False): ...

    def __init__(self, *args, **kwargs):
        options = [{"name": str},
                   {"name": str, "num_qubits": int},
                   {"paulis": Tuple},
                   {"paulis": Tuple, "skip_tests": bool},
                   {"x_logicals": List, "z_logicals": List, "stabilizers": List},
                   {"x_logicals": List, "z_logicals": List, "stabilizers": List, "skip_tests": bool}]
        i, a = _argument_assignment(
            options, "StabilizerCode()", *args, **kwargs)

        def _get_qecc_e_from_paulis(x_logicals, z_logicals, stabilizers, st) -> NDArray:
            k = len(x_logicals)
            n = len(stabilizers) + k
            if not st:
                if k != len(z_logicals):
                    raise ValueError(
                        "Different number of logical Pauli-X and Pauli-Z operators")
                if n != (m := max([max_index_of_pauli(i) for i in x_logicals + z_logicals + stabilizers])):
                    raise ValueError(
                        f"Wrong number of stabilizer generators for code defined on '{m}' qubits")
            els = [pauli_string_to_list(i, n)
                   for i in x_logicals + z_logicals + stabilizers]
            return np.array(els, dtype=np.int32).T

        if i == 0:
            self._e_mat = load_stabilizercode(a["name"])
        elif i == 1:
            self._e_mat = load_stabilizercode(a["name"], a["num_qubits"])
        elif i == 2:
            if len(a["paulis"]) != 3:
                raise ValueError(f"StabilizerCode() argument got invalid value '{
                                 str(a['paulis'])}'")
            self._e_mat = _get_qecc_e_from_paulis(
                a["paulis"][0], a["paulis"][1], a["paulis"][2], False)
            self._check_validity()
        elif i == 3:
            self._e_mat = _get_qecc_e_from_paulis(
                a["paulis"][0], a["paulis"][1], a["paulis"][2], a["skip_tests"])
            if not a["skip_tests"]:
                self._check_validity()
        elif i == 4:
            self._e_mat = _get_qecc_e_from_paulis(
                a["x_logicals"], a["z_logicals"], a["stabilizers"], False)
            self._check_validity()
        elif i == 5:
            self._e_mat = _get_qecc_e_from_paulis(
                a["x_logicals"], a["z_logicals"], a["stabilizers"], a["skip_tests"])
            if not a["skip_tests"]:
                self._check_validity()
        self._distance = -1

    def _check_validity(self):
        if matrix_rank(self.get_e_matrix()[:, 2*self.k:]) != self.n - self.k:
            raise ValueError(
                "Given stabilizer generators are not independent. Get an independent set by calling `reduce_to_stabilizer_generators`")
        e_xz = self.get_e_matrix()
        e_zx = np.roll(self.get_e_matrix(), shift=self.n, axis=0)
        if np.count_nonzero((e_xz[:, 2*self.k:].T@e_zx[:, 2*self.k:]) % 2) != 0:
            raise ValueError("Given stabilizer generators do not commute")
        if np.count_nonzero((e_xz[:, self.k:2*self.k].T@e_zx[:, self.k:2*self.k]) % 2) != 0:
            raise ValueError("Logical Pauli-Z do not commute with themselves")
        if np.count_nonzero((e_xz[:, self.k:].T@e_zx[:, self.k:]) % 2) != 0:
            raise ValueError("Logical Pauli-Z do not commute with stabilizers")
        if np.count_nonzero((e_xz[:, :self.k].T@e_zx[:, :self.k]) % 2) != 0:
            raise ValueError("Logical Pauli-X do not commute with themselves")
        if np.count_nonzero((np.delete(e_xz, range(self.k, 2*self.k), axis=1).T@np.delete(e_zx, range(self.k, 2*self.k), axis=1)) % 2) != 0:
            raise ValueError("Logical Pauli-X do not commute with themselves")
        if np.count_nonzero((e_xz[:, self.k:2*self.k].T@e_zx[:, :self.k]) % 2 - np.eye(self.k, dtype=np.int32)) != 0:
            raise ValueError(
                "Logical Pauli operators do not (anti-)commute correctly")

    def get_e_matrix(self) -> NDArray:
        return self._e_mat

    @ property
    def n(self) -> int:
        return np.shape(self._e_mat)[0]//2

    @ property
    def k(self) -> int:
        return np.shape(self._e_mat)[1] - self.n

    @ property
    def d(self) -> int:
        if self._distance == -1:
            self._compute_distance()
        return self._distance

    @ property
    def nkd(self) -> Tuple[int, int, int]:
        return (self.n, self.k, self.d)

    def _compute_distance(self):
        sle_a=np.zeros(self.n + 1, dtype=np.int32)
        for stabs in chain.from_iterable(combinations(self.get_e_matrix()[:, 2*self.k:].T, r)
                                         for r in range(self.n - self.k + 1)):
            if len(stabs) == 0:
                sle_a[0] += 1
                continue
            stab=np.sum(stabs, axis=0) % 2
            sle_a[np.count_nonzero(stab[:self.n] + stab[self.n])] += 1
        sle_b=MacWilliams(self.n)@sle_a
        self._distance=int(
            np.min(np.nonzero(np.round(sle_b*2**self.k).astype(sle_a.dtype) - sle_a)))


def reduce_to_stabilizer_generators(stabilizers: List[str]) -> List[str]:
    num_qubits=max([max_index_of_pauli(i) for i in stabilizers])
    stabs=np.array([pauli_string_to_list(i, num_qubits)
                     for i in stabilizers], dtype=np.int32)
    rank=matrix_rank(stabs)
    index=[]
    num=np.shape(stabs)[0]
    i=0
    while i < num - len(index):
        if rank == matrix_rank(a := np.delete(stabs, i, axis=0)):
            stabs=a
            index.append(i)
        else:
            i += 1
    new_stabs=deepcopy(stabilizers)
    for j in index:
        new_stabs.pop(j)
    return new_stabs


def are_independent_generators(stabilizers: List[str]) -> bool:
    return stabilizers == reduce_to_stabilizer_generators(stabilizers)
