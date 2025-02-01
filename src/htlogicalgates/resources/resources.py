import json
import os
from typing import Dict, List, Optional, Any

from numpy.typing import NDArray
import numpy as np

from .._utility import ENCODING_FILE, CON_FILE, ENCODING_KEY, DESCR_KEY, CON_KEY

_automated_cons = {"linear": "Connections exist between qubits $i$ and $i+1$.",
                   "circular": "Connections exist between qubits $i$ and $(i+1)%n$.",
                   "all": "Connections exist between all qubits."}

_automated_qeccs = {r"trivial": "The trivial [[n,n,1]] code."}


def get_json_resource(name: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    name = os.path.join(script_dir, name)
    with open(name, "r") as f:
        data = json.load(f)
        return data


def get_internal_qeccs() -> Dict:
    return get_json_resource(ENCODING_FILE)


def get_internal_connectivities() -> Dict:
    return get_json_resource(CON_FILE)


def read_external_json(path: str, *loc: Any) -> NDArray:
    with open(path, "r") as f:
        data = json.load(f)
        for l in loc:
            data = data[l]
        return np.array(data, dtype=np.int32)


def load_stabilizercode(name: str, num_qubits: Optional[int] = None) -> NDArray:
    if name in get_internal_qeccs().keys():
        return np.array(get_internal_qeccs()[name][ENCODING_KEY], dtype=np.int32).T
    if num_qubits == None:
        raise ValueError("Please pass a qubit count 'n'!")
    if "trivial" in name:
        return np.eye(2*num_qubits, dtype=np.int32)
    raise ValueError(f"No code found under name '{str(name)}'.")


def load_connectivity(name: str, num_qubits: Optional[int] = None) -> NDArray:
    if name in get_internal_connectivities().keys():
        return np.array(get_internal_connectivities()[name][CON_KEY], dtype=np.int32)
    if num_qubits == None:
        raise ValueError("Please pass a qubit count 'n'")
    elif name in _automated_cons.keys():
        if name in ["all-to-all", "all"]:
            return np.full((num_qubits, num_qubits), 1, dtype=np.int32) - np.identity(num_qubits, dtype=np.int32)
        if name in ["circular", "circle", "circ"]:
            return np.roll(np.identity(num_qubits, dtype=np.int32), shift=1, axis=0) +\
                np.roll(np.identity(num_qubits, dtype=np.int32), shift=-1, axis=0)
        if name in ["linear", "line"]:
            return np.eye(num_qubits, num_qubits, 1, dtype=np.int32) + np.eye(num_qubits, num_qubits, -1, dtype=np.int32)
    raise ValueError(f"No connectivity found under name '{str(name)}'.")


def available_stabilizercodes() -> Dict:
    """
    Returns all precompiled stabilizers codes as a dictionary.

    Returns
    ----------
    Dict
        Dictionary of all precompiled stabilizers codes. Keys are the names of the
        the stabilizers codes and values are explanations.
    """
    return {key: val[DESCR_KEY] for key, val in get_internal_qeccs().items()} | _automated_qeccs


def available_connectivities() -> Dict:
    """
    Returns all precompiled connectivies as a dictionary.

    Returns
    ----------
    Dict
        Dictionary of all precompiled connectivites. Keys are the names of the
        the connectivities and values are explanations.
    """
    return {key: val[DESCR_KEY] for key, val in get_internal_connectivities().items()} | _automated_cons
