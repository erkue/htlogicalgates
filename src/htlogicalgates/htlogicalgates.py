from typing import Union, Optional, Iterable, Dict, Any
from datetime import datetime
from enum import Enum
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
from .symplectic_rep.integer_symplectic import symplectic_matrix
from .symplectic_rep.helper import LinSolver
from .connectivity import Connectivity
from .stabilizercode import StabilizerCode
from ._utility import MissingOptionalLibraryError


class CostFunction(Enum):
    cz_count = "cz_count"
    h_count = "h_count"


def tailor_logical_gate(
    stab_code: StabilizerCode,
    connectivity: Connectivity,
    logical_gate: Union[Circuit, Clifford, int],
    num_cz_layers: int,
    time_limit: float = -1,
    log_to_console: bool = False,
    log_file: str = "",
    optimize: bool = True,
    gurobi: Dict = {},
    perm: Tuple[bool, bool] = (False, False),
    cost_function: CostFunction = CostFunction.cz_count
) -> Tuple[Optional[Circuit], str]:
    """
    Finds a circuit implementation for a Clifford gate of a given quantum
    error-correcting code tailored to a specified hardware connectivity.

    Parameters
    ----------
    stab_code: StabilizerCode
        Stabilizer code for which a logical circuit should be tailored.
    connectivity: Connectivity
        Connectivity to tailor the circuit to.
    logical_gate: Union[Circuit, Clifford, int]
        Representation of the logical gate in form of a circuit, Clifford or integer.
    num_cz_layers: int
        Number of controlled-Z gate layers for the ansatz with which the circuit should
        be compiled.
    time_limit: float, optional
        Time in seconds until the program aborts regardless of whether or not a circuit
        implementation has been found. A value of -1 removes the time limit, by default -1.
    log_to_console: bool, optional
        Whether or not Gurobi should log its progress to the console, by default False.
    log_file: str, optional
        File path of the log created by Gurobi. An empty string removes the log file,
        by default "".
    optimize: bool, optional
        Whether to collapse single-qubit Clifford gates after compilation, by default True.
    gurobi: Dict, optional
        Arguments to pass to the Gurobi optimizer, by default {}.
    perm: Tuple[bool, bool], optional
        If true, a permutation layer is added to the start (index 0) or end (index 1) of the circuit,
        by default [False, False].
    cost_function: CostFunction, optional
        The optimization metric used to evaluate and optimize circuits,
        by default CostFunction.cz_count.

    Returns
    -------
    Tuple[Optional[Circuit], str]
        A circuit (if one has been found) and a status message.

    Examples
    --------
        >>> conn = htlg.Connectivity("circular", n=4)
        >>> stab_code = htlg.StabilizerCode("4_2_2")
        >>> log_gate = htlg.Circuit("H 0", 2)
        >>> circ, status = tailor_logical_gate(stab_code, conn, log_gate, 2)

    """
    if not isinstance(stab_code, StabilizerCode):
        raise TypeError("Create qecc object via function 'get_qecc'.")
    stabilizer_matrix = stab_code.get_e_matrix()
    if not isinstance(connectivity, Connectivity):
        raise TypeError("Create connectivity object via function 'get_conn'")
    if not isinstance(logical_gate, Circuit) and not isinstance(logical_gate, int) and not isinstance(logical_gate, Clifford):
        raise TypeError("Wrong type for argument `logical_gate`")

    if isinstance(logical_gate, Circuit):
        logical_gate = logical_gate.to_clifford()
    if isinstance(logical_gate, Clifford):
        add_phases = np.roll(logical_gate.phase,
                             len(stabilizer_matrix[0]) - len(stabilizer_matrix[:, 0]) // 2)
        logical_gate = logical_gate.symplectic_matrix
    else:
        add_phases = None
    if isinstance(logical_gate, int):
        logical_gate = symplectic_matrix(
            logical_gate, len(stabilizer_matrix[:, 0]) // 2)

    if not stab_code.n == connectivity.num_qubits:
        raise ValueError(f"Stabilizer code is defined on '{stab_code.n}' qubits but connectvity on '{connectivity.num_qubits}'")
    if not stab_code.k == len(logical_gate) // 2:
        raise ValueError(f"Stabilizer code has '{stab_code.k}' logical qubits but logical gate is defined on '{len(logical_gate) // 2}'")

    gate_finder = GateFinder(num_cz_layers, connectivity.matrix, stabilizer_matrix, log_to_console,
                             log_file, gurobi, perm)
    if time_limit >= 0:
        gate_finder.set_time_limit(time_limit)
    gate_finder.set_logical_gate(logical_gate)
    gate_finder.set_target_function(cost_function)
    gate_finder.find_gate()
    if gate_finder.has_solution():
        if add_phases is not None and np.count_nonzero(add_phases) != 0:
            ps = Circuit.from_paulis(
                sum([stabilizer_matrix[:, i] for i in np.nonzero(add_phases)[0]]))
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
    perm: Tuple[bool, bool] = (False, False),
    cost_function: CostFunction = CostFunction.cz_count
) -> Dict:
    """
    Finds circuit implementations for multiple Clifford gates of a given quantum
    error-correcting code tailored to a specified hardware connectivity. The
    results can be saved to a file. 

    Parameters
    ----------
    stab_code: StabilizerCode
        Stabilizer code for which a logical circuit should be tailored.
    connectivity: Connectivity
        Connectivity to tailor the circuit to.
    logical_gates: Iterable[int]
        Iterable of integers identifying the Clifford gates that should be constructed.
    num_cz_layers: int
        Number of controlled-Z gate layers for the ansatz with which the circuit should
        be compiled.
    output_file: str, optional
        Filepath where the results dictionary is stored as a .json-file, by default "".
    time_limit: float, optional
        Time in seconds until the program aborts regardless of whether or not a circuit
        implementation has been found. A value of -1 removes the time limit, by default -1.
    log_to_console: bool, optional
        Whether or not Gurobi should log its progress to the console, by default False.
    log_file: str, optional
        File path of the log created by Gurobi. An empty string removes the log file,
        by default "".
    progress_bar: bool, optianl
        Whether or not to show a progress bar. Requires the tqdm package, by default False.
    save_every: int, optional
        If given an output file, continually save gates after constructing this many gates,
        by default 1.
    optimize: bool, optional
        Whether to collapse single-qubit Clifford gates after compilation, by default True.
    gurobi: Dict, optional
        Arguments to pass to the Gurobi optimizer, by default {}.
    perm: Tuple[bool, bool]
        If true, a permutation layer is added to the start (index 0) or end (index 1) of the circuit,
        by default [False, False].
    cost_function: CostFunction, optional
        The optimization metric used to evaluate and optimize circuits,
        by default CostFunction.cz_count.


    Returns
    -------
    Dict
        Dictionary containing the constructed circuits and runtime information.
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
    
    if not stab_code.n == connectivity.num_qubits:
        raise ValueError(f"Stabilizer code is defined on '{stab_code.n}' qubits but connectvity on '{connectivity.num_qubits}'")

    gate_finder = GateFinder(num_cz_layers, connectivity_matrix, stabilizer_matrix,
                             False, log_file, gurobi, perm)
    if time_limit >= 0:
        gate_finder.set_time_limit(time_limit)
    gate_finder.set_target_function(cost_function)
    stor = {
        "Meta": {
            "Connectivity": connectivity,
            "Code": stab_code,
            "Number CZ layers": num_cz_layers,
            "Time limit": time_limit,
            "Started": str(datetime.now())
        },
        "Gates": {}
    }
    for num, i in enumerate(iterate(logical_gates)):
        gate_finder.set_logical_gate(symplectic_matrix(i, gate_finder.k))
        gate_finder.find_gate()
        if gate_finder.has_solution():
            circuit = gate_finder.get_circuit_implementation()
            if optimize:
                circuit.shallow_optimize()
            stor["Gates"][i] = {"Circuit": circuit,
                                "Status": gate_finder.get_status(),
                                "Runtime": gate_finder.get_runtime()}
        else:
            stor["Gates"][i] = {"Circuit": None,
                                "Status": gate_finder.get_status(),
                                "Runtime": gate_finder.get_runtime()}
        if (num + 1) % save_every == 0:
            if output_file != "":
                save_results_dictionary(stor, output_file)
    stor["Meta"]["Ended"] = str(datetime.now())
    if output_file != "":
        save_results_dictionary(stor, output_file)
    return stor


def save_results_dictionary(results: Dict, filepath: str):
    """
    Saves the result dictionary returned by `tailor_multiple_logical_gates`
    to a .json-file.

    Parameters
    ----------
    results: Dict
        Results dictionary.
    filepath: str
        File path to save the .json-file.
    """
    results = deepcopy(results)
    if (c := results.get("Meta", {}).get("Connectivity", "N/A")) != "N/A":
        results["Meta"]["Connectivity"] = c.matrix.tolist()
    if (c := results.get("Meta", {}).get("Code", "N/A")) != "N/A":
        results["Meta"]["Code"] = c.get_e_matrix().tolist()
    for key, val in results.get("Gates", {}).items():
        circ = val.get("Circuit", None)
        if isinstance(circ, Circuit):
            results["Gates"][key]["Circuit"] = circ.__str__()
    with open(filepath, 'w') as file:
        json.dump(results, file)


def load_results_dictionary(filepath: str):
    """
    Loads the result dictionary saved by `save_results_dictionary`.

    Parameters
    ----------
    filepath: str
        File path to load the results dictionary.
    
    Returns
    ----------
    Dict:
        Dictionary containing the information stored in the .json-file.
    """
    with open(filepath, "r") as file:
        results = json.load(file)
    if (c := results.get("Meta", {}).get("Connectivity", "N/A")) != "N/A":
        results["Meta"]["Connectivity"] = Connectivity(np.array(c, dtype=np.int32)) 
    if (c := results.get("Meta", {}).get("Code", "N/A")) != "N/A":
        results["Meta"]["Code"] = StabilizerCode(np.array(c, dtype=np.int32)) 
    for key, val in results.get("Gates", {}).items():
        circ = val.get("Circuit", None) 
        if isinstance(circ, str) and circ.capitalize() != "NONE":
            results["Gates"][key]["Circuit"] = Circuit(circ)
        else:
            results["Gates"][key]["Circuit"] = None
    for key in list(results.get("Gates", {}).keys()):
        results["Gates"][int(key)] = results["Gates"].pop(key)
    return results


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
        self.ENC = Clifford(enc)
        self.n = len(self.CON)
        self.k = self.ENC.m - self.n
        self.active_gate = False

        self.env = Enviroment(log_to_console, log_file, gurobi)
        self.SCLs = [create_SCL(self.n, self.env)] + \
            [create_cons_SCL(self.n, self.env) for _ in range(self.NUM_CZL)]
        self.CZLs = [create_CZL(self.CON, self.env)
                     for _ in range(self.NUM_CZL)]
        self.LOGICAL, self.LOG_IDS = self.env.create_predef_bin_matrix(
            2*self.k, 2*self.k)
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

    def set_target_function(self, cost_function: CostFunction = CostFunction.cz_count):
        e = Expression.create_const(0)
        if cost_function == CostFunction.cz_count:
            for czl in self.CZLs:
                for i in range(self.n):
                    for j in range(i+1, self.n):
                        e = e + czl[self.n+i, j]
        elif cost_function == CostFunction.h_count:
            for scl in self.SCLs:
                for i in range(self.n):
                    e = e + scl[i, self.n+i]
        else:
            raise ValueError(f"Unsupported cost function '{cost_function}'")
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
        cliffs[::2] = [Clifford(
            self.env.evaluate_matrix(i)) for i in self.SCLs]
        cliffs[1::2] = [Clifford(
            self.env.evaluate_matrix(i)) for i in self.CZLs]
        tot_cliff: Clifford = cliffs[0]
        circs[::2] = [Circuit.from_SCL(i) for i in cliffs[::2][::-1]]
        circs[1::2] = [Circuit.from_CZL(i) for i in cliffs[1::2][::-1]]

        for c in cliffs[1:]:
            tot_cliff = tot_cliff @ c
        if self.Perms[0] != None:
            self.Perms[0] = Clifford(
                self.env.evaluate_matrix(self.Perms[0]))
            tot_cliff = tot_cliff @ self.Perms[0]
            circs.insert(0, Circuit.from_permutation(self.Perms[0]))
        if self.Perms[1] != None:
            self.Perms[1] = Clifford(
                self.env.evaluate_matrix(self.Perms[1]))
            tot_cliff = self.Perms[1] @ tot_cliff
            circs.append(Circuit.from_permutation(self.Perms[1]))

        paulis = Circuit.from_paulis(
            self.lin_solv.get_solution((tot_cliff@self.ENC).phase), invert=True)
        return sum(circs, start=paulis)
