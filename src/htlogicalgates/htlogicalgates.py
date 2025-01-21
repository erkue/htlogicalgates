from typing import Union, Optional, Iterable, Dict, Any
from datetime import datetime
import json
import sys

import numpy as np
from numpy.typing import NDArray

from .grb_interface.grb_enviroment import Enviroment
from .grb_interface.grb_gates import *
from .grb_interface.grb_math_interface import *
from .symplectic_rep import *
from .symplectic_rep.random_symplectic import symplectic_t
from .symplectic_rep.helper import LinSolver
from .codes.codes import get_encoding_of_code, get_connectivity

def tailor_logical_gate(code : Union[str, NDArray],
                      connectivity : Union[str, NDArray],
                      logical_gate : Union[str, Circuit, int, NDArray],
                      num_CZL : int, time_limit : float = -1,
                      log_to_console : bool = False,
                      log_file : str = "",
                      **kwargs) -> Tuple[Optional[Circuit], str]:
    """Find a circuit implementation for a Clifford gate of a given quantum
    error-correcting code tailored to a specified hardware connectivity. Some
    codes and connectivities can be accesed by their names. For more information,
    see `htlogicalgates.codes`.

    Args:
        code (Union[str, NDArray]): Name of code or numpy array of shape `(2n,n+k)` \
        consisting logical Pauli operators and stabilizers of a code. 
        connectivity (Union[str, NDArray]): Name of connectivity or numpy array \
        of shape (n,n) representing the connectivity matrix. 
        logical_gate (Union[str, int, NDArray]): Representation of the \
        logical gate in form of a string, circuit, integer, or numpy array.
        num_CZL (int): Number of controlled-Z gate layers of the ansatz with which \
        the circuit should be compiled.
        time_limit (float, optional): Time in seconds until the programm aborts \
        regardless of whether or not a circuit implementation has been found A value \
        of -1 removes the time limit. Defaults to -1.
        log_to_console (bool, optional): Whether or not Gurobi should log its progress \
        to the console. Defaults to False.
        log_file (str, optional): File path of the log created by Gurobi. An emptry \
        string removes the log-file. Defaults to "".
        gurobi (Dict[Any, Any], optional): Additional arguments for the Gurobi solver.
        optimize (bool, optional): Perform a slight optimization of single-qubit gates. \
        Defaults to True.
        perm (Tuple[bool, bool], optional): Defaults to [False, False] 
    Returns:
        Tuple[Optional[Circuit], str]: A representation of the circuit in form of a \
        member the circuit class and a string containing the final status message. \
        If a circuit has not been found, `None` is returned instead of the circuit.

    Examples:
        >>> circ = find_one_logical_gate("4_2_2", "circular", "H 0", 2, -1, True)
    """
    if isinstance(code, str): code = get_encoding_of_code(code)
    if not isinstance(code, np.ndarray): raise TypeError("Invalid code argument!")
    if type(connectivity) == str: connectivity = get_connectivity(connectivity, len(code[:,0])//2)
    if not isinstance(connectivity, np.ndarray): raise TypeError("Invalid connectivity argument!")
    if isinstance(logical_gate, str): logical_gate = Circuit.get_circuit_from_string(logical_gate, len(code[0])-len(code[:,0])//2)
    if isinstance(logical_gate, Circuit):
        logical_gate = logical_gate.get_as_clifford()
        add_phases = np.roll(logical_gate.phase, len(code[0])-len(code[:,0])//2)
        logical_gate = logical_gate.symplectic_matrix
    else:
        add_phases = None
    if isinstance(logical_gate, int): logical_gate = symplectic_t(logical_gate, len(code[:,0])//2)
    gf = GateFinder(num_CZL, connectivity, code, log_to_console, log_file, kwargs.get("gurobi", {}), kwargs.get("perm", [False, False]))
    if time_limit >= 0:
        gf.set_time_limit(time_limit)
    gf.set_logical_gate(logical_gate)
    gf.set_target_function()
    gf.find_gate()
    if gf.has_solution():
        if add_phases is not None and np.count_nonzero(add_phases) != 0:
            print(add_phases)
            ps = Circuit.get_paulis_as_circuit(sum([code[:,i] for i in np.nonzero(add_phases)]))
            c = ps + gf.get_circuit_implementation()
            if kwargs.get("optimize", True):
                c.shallow_optimize()
            return c, gf.get_status()
        else:
            c = gf.get_circuit_implementation()
            if kwargs.get("optimize", True):
                c.shallow_optimize()
            return c, gf.get_status()
    else:
        return None, gf.get_status()

def tailor_multiple_logical_gates(code : Union[str, NDArray],
                      connectivity : Union[str, NDArray],
                      logical_gates : Iterable[int],
                      num_CZL : int,
                      output_file : str = "",
                      time_limit : float = -1,
                      log_file : str = "",
                      progress_bar : bool = False,
                      save_every : int = 1,
                      **kwargs):
    """Find a circuit implementations for multiple Clifford gates of a given quantum
    error-correcting code tailored to a specified hardware connectivity. Some
    codes and connectivities can be accesed by their names. For more information,
    see `htlogicalgates.codes`.

    Args:
        code (Union[str, NDArray]): Name of code or numpy array of shape `(2n,n+k)` \
        consisting logical Pauli operators and stabilizers of a code. 
        connectivity (Union[str, NDArray]): Name of connectivity or numpy array \
        of shape (n,n) representing the connectivity matrix. 
        logical_gates (Iterable[int]): Integers representing logical Clifford gates.
        num_CZL (int): Number of controlled-Z gate layers of the ansatz with which \
        the circuit should be compiled.
        time_limit (float, optional): Time in seconds until the programm aborts \
        regardless of whether or not a circuit implementation has been found A value \
        of -1 removes the time limit. Defaults to -1.
        log_to_console (bool, optional): Whether or not Gurobi should log its progress \
        to the console. Defaults to False.
        log_file (str, optional): File path of the log created by Gurobi. An emptry \
        string removes the log-file. Defaults to "".
        gurobi (Dict[Any, Any], optional): Additional arguments for the Gurobi solver.
        optimize (bool, optional): Perform a slight optimization of single-qubit gates. \
        Defaults to True.
        perm (Tuple[bool, bool], optional): Defaults to [False, False] 
    Raises:
        TypeError: _description_
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    if isinstance(code, str): code = get_encoding_of_code(code)
    if not isinstance(code, np.ndarray): raise TypeError("Invalid code argument!")
    if type(connectivity) == str: connectivity = get_connectivity(connectivity, len(code[:,0])//2)
    if not isinstance(connectivity, np.ndarray): raise TypeError("Invalid connectivity argument!")
    if progress_bar:
        try:
            from tqdm import tqdm
            f = lambda x : tqdm(x, smoothing=0)
        except ImportError:
            print("WARNING: Package 'tqdm' is not installed. Continuing without progress bar.")
            f = lambda x : x
    else: 
        f = lambda x : x
    gf = GateFinder(num_CZL, connectivity, code, False, log_file, kwargs.get("gurobi", {}), kwargs.get("perm", [False, False]))
    if time_limit >= 0:
        gf.set_time_limit(time_limit)
    gf.set_target_function()
    stor = {"Meta" : {"Connectivity" : str(connectivity),
                      "Code" : str(code),
                      "n" : gf.n,
                      "k" : gf.k,
                      "Number CZL" : num_CZL,
                      "Time limit" : time_limit,
                      "Started" : str(datetime.now())
                      },
            "Gates" : {}}
    for num, i in enumerate(f(logical_gates)):
        gf.set_logical_gate(symplectic_t(i, gf.k))
        gf.find_gate()
        if gf.has_solution():
            c = gf.get_circuit_implementation()
            if kwargs.get("optimize", True):
                c.shallow_optimize()
            stor["Gates"][i] = {"Circuit" : c.get_as_string(),
                                "Status" : gf.get_status(),
                                "Runtime" : gf.get_runtime()}
        else:
            stor["Gates"][i] = {"Circuit" : None,
                                "Status" : gf.get_status(),
                                "Runtime" : gf.get_runtime()}
        if (num+1)%save_every == 0:
            if output_file != "":
                with open(output_file, 'w') as f:
                    json.dump(stor, f)
    stor["Meta"]["Ended"] = str(datetime.now())
    if output_file != "":
        with open(output_file, 'w') as f:
            json.dump(stor, f)
    return stor

class GateFinder:
    def __init__(self, num_CZL : int, con : NDArray, enc : NDArray,
                 log_to_console : bool = False, log_file : str = "",
                 gurobi : Dict[Any, Any] = {},
                 perm : Tuple[bool, bool] = [False, False]):
        self.NUM_CZL = num_CZL
        self.CON = con
        self.ENC = Clifford.from_matrix(enc)
        self.n = len(self.CON)
        self.k = self.ENC.M - self.n
        self.active_gate = False

        self.env = Enviroment(log_to_console, log_file, gurobi)
        self.SCLs = [create_SCL(self.n, self.env)] + [create_cons_SCL(self.n, self.env) for _ in range(self.NUM_CZL)]
        self.CZLs = [create_CZL(self.CON, self.env) for _ in range(self.NUM_CZL)]
        self.LOGICAL, self.LOG_IDS = self.env.create_predef_bin_matrix(2*self.k, 2*self.k)
        self.FREEDOM = create_reduced_freedom_matrix(self.n, self.k, self.env)
        self.Perms = [None, None]
        self.ANSATZ = self.SCLs[0]
        if perm[1]:
            self.Perms[1] = create_Perm(self.n, self.env)
            self.ANSATZ = self.Perms[1]@self.ANSATZ
        for scl, czl in zip(self.SCLs[1:], self.CZLs):
            self.ANSATZ = self.ANSATZ@czl@scl
        if perm[0]:
            self.Perms[0] = create_Perm(self.n, self.env)
            self.ANSATZ = self.ANSATZ@self.Perms[0]
        enc_expr = ExprMatrix.create_from_array(self.ENC.symplectic_matrix)
        self.env.add_equality_constraint_mat(enc_expr@(self.LOGICAL.create_expanded_dims(self.n + self.k, self.n + self.k) + self.FREEDOM),
                                             self.ANSATZ@enc_expr, True)
        
        self.lin_solv = LinSolver(enc.T)
        
    def set_time_limit(self, time_limit : float):
        self.env.set_time_limit(time_limit)

    def set_logical_gate(self, logical_gate : NDArray):
        self.env.set_many_predef_var(logical_gate%2, self.LOG_IDS)
        self.active_gate = True

    def set_target_function(self):
        e = Expression.create_const(0)
        for czl in self.CZLs:
            for i in range(self.n):
                for j in range(i+1, self.n):
                    e = e + czl[self.n+i,j]
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
        assert(self.active_gate)
        self.env.solve()

    def get_circuit_implementation(self) -> Circuit:
        cliffs : List[Clifford] = [None] * (2*self.NUM_CZL + 1)
        circs : List[Circuit] = [None] * (2*self.NUM_CZL + 1)
        cliffs[::2] = [Clifford.from_matrix(self.env.evaluate_matrix(i)) for i in self.SCLs]
        cliffs[1::2] = [Clifford.from_matrix(self.env.evaluate_matrix(i)) for i in self.CZLs]
        tot_cliff : Clifford = cliffs[0]
        circs[::2] = [Circuit.get_SCL_as_circuit(i) for i in cliffs[::2][::-1]]
        circs[1::2] = [Circuit.get_CZL_as_circuit(i) for i in cliffs[1::2][::-1]]
        
        for c in cliffs[1:]:
            tot_cliff = c@tot_cliff
        if self.Perms[0] != None:
            self.Perms[0] = Clifford.from_matrix(self.env.evaluate_matrix(self.Perms[0]))
            tot_cliff = tot_cliff@self.Perms[0]
            circs.insert(0, Circuit.get_perm_as_circuit(self.Perms[0]))
        if self.Perms[1] != None:
            self.Perms[1] = Clifford.from_matrix(self.env.evaluate_matrix(self.Perms[1]))
            tot_cliff = self.Perms[1]@tot_cliff
            circs.append(Circuit.get_perm_as_circuit(self.Perms[1]))

        paulis = Circuit.get_paulis_as_circuit(self.lin_solv.get_solution((tot_cliff@self.ENC).phase), invert=True)
        return sum(circs, start=paulis)