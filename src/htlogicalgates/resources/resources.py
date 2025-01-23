import json
import os
from typing import Dict, List, Optional, Any

from numpy.typing import NDArray
import numpy as np

from .._global_vars import ITYPE, ENCODING_FILE, CON_FILE, ENCODING_KEY, DESCR_KEY, CON_KEY

_automated_cons = {"linear" : "Connections exist between qubits $i$ and $i+1$.",
                  "circular" : "Connections exist between qubits $i$ and $(i+1)%n$.",
                  "all" : "Connections exist between all qubits."}

_automated_qeccs = {"trivial n" : "The trivial [[n,n,1]] code."}

def get_json_resource(name : str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    name = os.path.join(script_dir, name)
    with open(name, "r") as f:
        data = json.load(f)
        return data

def get_internal_qeccs() -> Dict:
    return get_json_resource(ENCODING_FILE)

def get_internal_connectivities() -> Dict:
    return get_json_resource(CON_FILE)

def read_external_json(path : str, *loc : Any) -> NDArray:
    with open(path, "r") as f:
        data = json.load(f)
        for l in loc:
            data = data[l]
        return np.array(data, dtype=ITYPE)

def available_qeccs() -> Dict:
    return {key : val[DESCR_KEY] for key, val in get_internal_qeccs().items()} + _automated_qeccs

def load_qecc(name : str) -> NDArray:
    if name in get_internal_qeccs().keys():
        return np.array(get_internal_qeccs()[name][ENCODING_KEY], dtype=ITYPE).T
    if "trivial" in name:
        n = int(name.split()[1])
        return np.eye(2*n, dtype=ITYPE)
    raise ValueError(f"No code found under name '{str(name)}'.")


def load_connectivity(name : str, n : Optional[int] = None) -> NDArray:
    if name in get_internal_connectivities().keys():
        return np.array(get_internal_connectivities()[name][CON_KEY], dtype=ITYPE)
    if n == None:
        raise ValueError("Please pass a qubit count 'n'!")
    elif name in _automated_cons.keys():
        if name in ["all-to-all", "all"]:
            return np.full((n,n), 1, dtype=ITYPE) - np.identity(n, dtype=ITYPE)
        if name in ["circular", "circle", "circ"]:
            return np.roll(np.identity(n, dtype=ITYPE), shift=1, axis=0) +\
                   np.roll(np.identity(n, dtype=ITYPE), shift=-1, axis=0)
        if name in ["linear", "line"]:
            return np.eye(n, n, 1, dtype=ITYPE) + np.eye(n, n, -1, dtype=ITYPE)
    raise ValueError(f"No connectivity found under name '{str(name)}'.")

def available_connectivities() -> Dict:
    return {key : val[DESCR_KEY] for key, val in get_internal_connectivities().items()} + _automated_cons