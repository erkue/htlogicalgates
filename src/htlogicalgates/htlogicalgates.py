from typing import Union, Optional, Iterable, Dict, Any
from datetime import datetime
import json
import numpy as np
from numpy.typing import NDArray

try:
    from tqdm import tqdm
except ImportError:
    _has_tqdm = False
finally:
    _has_tqdm = True

from .grb_interface.grb_enviroment import Enviroment
from .grb_interface.grb_gates import *
from .grb_interface.grb_math_interface import *
from .symplectic_rep import *
from .symplectic_rep.random_symplectic import symplectic_t
from .symplectic_rep.helper import LinSolver
from .connectivity import Connectivity
from .stabilizercode import StabilizerCode
from ._utility import MissingOptionalLibraryError


def tailor_logical_gate(
    stab_code: StabilizerCode,
    connectivity: Connectivity,
    logical_gate: Union[Circuit, int],
    num_cz_layers: int,
    time_limit: float = -1,
    log_to_console: bool = False,
    log_file: str = "",
    optimize: bool = True,
    gurobi: Dict = {},
    perm: Tuple[bool, bool] = (False, False)
) -> Tuple[Optional[Circuit], str]:
    """
    Find a circuit implementation for a Clifford gate of a given quantum
    error-correcting code tailored to a specified hardware connectivity. Some
    codes and connectivities can be accessed by their names. For more information,
    see `htlogicalgates.codes`.

    Parameters
    ----------
    * `stab_code: StabilizerCode` \\
        Stabilizer code for which a logical circuit should be tailored.
    * `connectivity: Connectivity` \\
        Connectivity to tailor circuit to. 
    * `logical_gate: Union[Circuit, int]` \\
        Representation of the logical gate in form of a circuit or integer.
    * `num_cz_layers: int` \\
        Number of controlled-Z gate layers of the ansatz with which the circuit should
        be compiled.
    * `time_limit: float, optional` \\
        Time in seconds until the programm aborts regardless of whether or not a circuit
        implementation has been found A value of -1 removes the time limit, by default -1.
    * `log_to_console: bool, optional` \\
        Whether or not Gurobi should log its progress to the console, by default False.
    * `log_file: str, optional` \\
        File path of the log created by Gurobi. An emptry string removes the log-file, 
        by default "".
    * `optimize: bool, optional` \\
        Collapse single-qubit Clifford gates after compilation, by default True.
    * `gurobi: Dict, optional` \\
        Arguments to pass to the gurobi optimizer, by default {}.
    * `perm: Tuple[bool, bool]` \\
       If true, add a permutation layer to the start (index 0) or end (index 1) to the circuit,
        by default [False, False].

    Returns
    -------
    Tuple[Optional[Circuit], str]
        A representation of the circuit in form of a \
        member the circuit class and a string containing the final status message. \
        If a circuit has not been found, `None` is returned instead of the circuit.

    Examples
    --------
        >>> circ = find_one_logical_gate("4_2_2", "circular", "H 0", 2, -1, True)

    """
    if not isinstance(stab_code, StabilizerCode):
        raise TypeError("Create qecc object via function 'get_qecc'!")
    stabilizer_matrix = stab_code.get_e_matrix()
    if not isinstance(connectivity, Connectivity):
        raise TypeError("Create connectivity object via function 'get_conn'!")
    if not isinstance(logical_gate, Circuit) and not isinstance(logical_gate, int):
        raise TypeError("Expected a `Circuit` object for `logical_gate`")

    if isinstance(logical_gate, Circuit):
        logical_gate = logical_gate.to_clifford()
        add_phases = np.roll(logical_gate.phase,
                             len(stabilizer_matrix[0]) - len(stabilizer_matrix[:, 0]) // 2)
        logical_gate = logical_gate.symplectic_matrix
    else:
        add_phases = None
    if isinstance(logical_gate, int):
        logical_gate = symplectic_t(logical_gate, len(stabilizer_matrix[:, 0]) // 2)

    gate_finder = GateFinder(num_cz_layers, connectivity.matrix, stabilizer_matrix, log_to_console,
                             log_file, gurobi, perm)
    if time_limit >= 0:
        gate_finder.set_time_limit(time_limit)
    gate_finder.set_logical_gate(logical_gate)
    gate_finder.set_target_function()
    gate_finder.find_gate()
    if gate_finder.has_solution():
        if add_phases is not None and np.count_nonzero(add_phases) != 0:
            print(add_phases)
            ps = Circuit.from_paulis(
                sum([stabilizer_matrix[:, i] for i in np.nonzero(add_phases)]))
            circuit = ps + gate_finder.get_circuit_implementation()
            if optimize:
                circuit.shallow_optimize()
            return circuit, gate_finder.get_status()
        else:
            circuit = gate_finder.get_circuit_implementation()
            if optimize:
                circuit.shallow_optimize()
            return circuit, gate_finder.get_status()
    else:
        return None, gate_finder.get_status()


def tailor_multiple_logical_gates(
    stab_code: StabilizerCode,
    connectivity: Connectivity,
    logical_gates: Iterable[int],
    num_cz_layers: int,
    output_file: str = "",
    time_limit: float = -1,
    log_file: str = "",
    progress_bar: bool = False,
    save_every: int = 1,
    optimize: bool = True,
    gurobi: Dict = {},
    perm: Tuple[bool, bool] = (False, False)
) -> dict:
    """
    Find a circuit implementations for multiple Clifford gates of a given quantum
    error-correcting code tailored to a specified hardware connectivity. Some
    codes and connectivities can be accessed by their names. For more information,
    see `htlogicalgates.codes`.

    Parameters
    ----------
    * `stab_code: StabilizerCode` \\
        Stabilizer code for which a logical circuit should be tailored.
    * `connectivity: Union[str, NDArray]` \\
        Name of connectivity or numpy array of shape (n,n) representing the
          connectivity matrix. 
    * `logical_gates: Iterable[int]` \\
        Integers representing logical Clifford gates.
    * `num_cz_layers: int` \\
        Number of controlled-Z gate layers of the ansatz with which the circuit
          should be compiled.
    * `time_limit: float, optional` \\
        Time in seconds until the programm aborts regardless of whether or not a
        circuit implementation has been found A value of -1 removes the time limit,
        by default -1.
    * `log_to_console: bool, optional` \\
        Whether or not Gurobi should log its progress to the console, by default False.
    * `log_file: str, optional` \\
        File path of the log created by Gurobi. An emptry string removes the log-file,
        by default "".
    * `progress_bar: bool, optional` \\
        Whether or not to show a progress bar. Requires tqdm to be installed, by default False.
    * `optimize: bool, optional` \\
        Collapse single-qubit Clifford gates after compilation, by default True.
    * `gurobi: Dict, optional` \\
        Arguments to pass to the gurobi optimizer, by default {}.
    * `perm: Tuple[bool, bool]` \\
       If true, add a permutation layer to the start (index 0) or end (index 1) to the circuit,
       by default [False, False].

    Returns
    -------
    """
    if not isinstance(stab_code, StabilizerCode):
        raise TypeError("Create qecc object via function 'get_code'!")
    stabilizer_matrix = stab_code.get_e_matrix()
    if not isinstance(connectivity, Connectivity):
        raise TypeError("Create connectivity object via function 'get_conn'!")
    connectivity_matrix = connectivity.matrix

    def iterate(x):
        if progress_bar:
            if not _has_tqdm:
                raise MissingOptionalLibraryError(
                    "progress_bar requires 'qiskit' to be installed")

            return tqdm(x, smoothing=0)  # type: ignore
        return x

    gate_finder = GateFinder(num_cz_layers, connectivity_matrix, stabilizer_matrix,
                             False, log_file, gurobi, perm)
    if time_limit >= 0:
        gate_finder.set_time_limit(time_limit)
    gate_finder.set_target_function()
    stor = {
        "Meta": {
            "Connectivity": str(connectivity_matrix),
            "Code": str(stabilizer_matrix),
            "n": gate_finder.n,
            "k": gate_finder.k,
            "Number CZ layers": num_cz_layers,
            "Time limit": time_limit,
            "Started": str(datetime.now())
        },
        "Gates": {}
    }
    for num, i in enumerate(iterate(logical_gates)):
        gate_finder.set_logical_gate(symplectic_t(i, gate_finder.k))
        gate_finder.find_gate()
        if gate_finder.has_solution():
            circuit = gate_finder.get_circuit_implementation()
            if optimize:
                circuit.shallow_optimize()
            stor["Gates"][i] = {"Circuit": circuit.__str__(),
                                "Status": gate_finder.get_status(),
                                "Runtime": gate_finder.get_runtime()}
        else:
            stor["Gates"][i] = {"Circuit": None,
                                "Status": gate_finder.get_status(),
                                "Runtime": gate_finder.get_runtime()}
        if (num + 1) % save_every == 0:
            if output_file != "":
                with open(output_file, 'w') as file:
                    json.dump(stor, file)
    stor["Meta"]["Ended"] = str(datetime.now())
    if output_file != "":
        with open(output_file, 'w') as file:
            json.dump(stor, file)
    return stor


class GateFinder:
    def __init__(
        self,
        num_CZL: int,
        con: NDArray,
        enc: NDArray,
        log_to_console: bool = False,
        log_file: str = "",
        gurobi: Dict[Any, Any] = {},
        perm: Tuple[bool, bool] = (False, False)
    ):
        self.NUM_CZL = num_CZL
        self.CON = con
        self.ENC = Clifford.from_matrix(enc)
        self.n = len(self.CON)
        self.k = self.ENC.m - self.n
        self.active_gate = False

        self.env = Enviroment(log_to_console, log_file, gurobi)
        self.SCLs = [create_SCL(self.n, self.env)] + \
            [create_cons_SCL(self.n, self.env) for _ in range(self.NUM_CZL)]
        self.CZLs = [create_CZL(self.CON, self.env) for _ in range(self.NUM_CZL)]
        self.LOGICAL, self.LOG_IDS = self.env.create_predef_bin_matrix(2*self.k, 2*self.k)
        self.FREEDOM = create_reduced_freedom_matrix(self.n, self.k, self.env)
        self.Perms: List[Union[None, Clifford, ExprMatrix]] = [None, None]
        self.ANSATZ = self.SCLs[0]
        if perm[1]:
            self.Perms[1] = create_Perm(self.n, self.env)
            self.ANSATZ = self.Perms[1] @ self.ANSATZ
        for scl, czl in zip(self.SCLs[1:], self.CZLs):
            self.ANSATZ = self.ANSATZ @ czl @ scl
        if perm[0]:
            self.Perms[0] = create_Perm(self.n, self.env)
            self.ANSATZ = self.ANSATZ @ self.Perms[0]
        enc_expr = ExprMatrix.create_from_array(self.ENC.symplectic_matrix)
        self.env.add_equality_constraint_mat(
            enc_expr @ (self.LOGICAL.create_expanded_dims(self.n +
                        self.k, self.n + self.k) + self.FREEDOM),
            self.ANSATZ @ enc_expr,
            True
        )
        self.lin_solv = LinSolver(enc.T)

    def set_time_limit(self, time_limit: float):
        self.env.set_time_limit(time_limit)

    def set_logical_gate(self, logical_gate: NDArray):
        self.env.set_many_predef_var(logical_gate % 2, self.LOG_IDS)
        self.active_gate = True

    def set_target_function(self):
        e = Expression.create_const(0)
        for czl in self.CZLs:
            for i in range(self.n):
                for j in range(i+1, self.n):
                    e = e + czl[self.n+i, j]
        self.env.set_target_function(e)

    def has_solution(self) -> bool:
        return self.env.has_solution()

    def get_status(self) -> str:
        return self.env.get_status()

    def get_runtime(self) -> float:
        return self.env.get_runtime()

    def get_work(self) -> float:
        return self.env.get_work()

    def find_gate(self):
        assert (self.active_gate)
        self.env.solve()

    def get_circuit_implementation(self) -> Circuit:
        cliffs: List[Clifford] = [None] * (2*self.NUM_CZL + 1)
        circs: List[Circuit] = [None] * (2*self.NUM_CZL + 1)
        cliffs[::2] = [Clifford.from_matrix(
            self.env.evaluate_matrix(i)) for i in self.SCLs]
        cliffs[1::2] = [Clifford.from_matrix(
            self.env.evaluate_matrix(i)) for i in self.CZLs]
        tot_cliff: Clifford = cliffs[0]
        circs[::2] = [Circuit.from_SCL(i) for i in cliffs[::2][::-1]]
        circs[1::2] = [Circuit.from_CZL(i) for i in cliffs[1::2][::-1]]

        for c in cliffs[1:]:
            tot_cliff = c @ tot_cliff
        if self.Perms[0] != None:
            self.Perms[0] = Clifford.from_matrix(
                self.env.evaluate_matrix(self.Perms[0]))
            tot_cliff = tot_cliff @ self.Perms[0]
            circs.insert(0, Circuit.from_permutation(self.Perms[0]))
        if self.Perms[1] != None:
            self.Perms[1] = Clifford.from_matrix(
                self.env.evaluate_matrix(self.Perms[1]))
            tot_cliff = self.Perms[1] @ tot_cliff
            circs.append(Circuit.from_permutation(self.Perms[1]))

        paulis = Circuit.from_paulis(
            self.lin_solv.get_solution((tot_cliff@self.ENC).phase), invert=True)
        return sum(circs, start=paulis)
