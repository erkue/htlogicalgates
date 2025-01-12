import json
import os
from typing import Dict, List, Optional, Any

from numpy.typing import NDArray
import numpy as np

from .._global_vars import ITYPE, ENCODING_FILE, CON_FILE, ENCODING_KEY, DESCR_KEY, CON_KEY

_automated_cons = {"linear" : "Connections exist between qubits $i$ and $i+1$.",
                  "circular" : "Connections exist between qubits $i$ and $(i+1)%n$.",
                  "all-to-all" : "Connections exist between all qubits.",
                  "all" : "Connections exist between all qubits."}

_automated_codes = {"trivial" : "The trivial [[n,n,1]] code. Get code with argument 'trivial n'."}

def get_json_resource(name : str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    name = os.path.join(script_dir, name)
    with open(name, "r") as f:
        data = json.load(f)
        return data

def get_internal_encodings() -> Dict:
    return get_json_resource(ENCODING_FILE)

def get_internal_connectivities() -> Dict:
    return get_json_resource(CON_FILE)

def read_external_json(path : str, *loc : Any) -> NDArray:
    with open(path, "r") as f:
        data = json.load(f)
        for l in loc:
            data = data[l]
        return np.array(data, dtype=ITYPE)

def available_encodings() -> Dict:
    return {key : val[DESCR_KEY] for key, val in get_internal_encodings().items()} + _automated_codes

def get_encoding_of_code(name : str) -> NDArray:
    if name in get_internal_encodings().keys():
        return np.array(get_internal_encodings()[name][ENCODING_KEY], dtype=ITYPE).T
    if "trivial" in name:
        n = int(name.split()[1])
        return np.eye(2*n, dtype=ITYPE)
    raise ValueError(f"No code found under name '{str(name)}'.")


def get_connectivity(name : str, n : Optional[int] = None):
    if name in get_internal_connectivities().keys():
        return np.array(get_internal_connectivities()[name][CON_KEY], dtype=ITYPE)
    elif name in _automated_cons.keys():
        if name == "all-to-all" or name == "all":
            return np.full((n,n), 1, dtype=ITYPE) - np.identity(n, dtype=ITYPE)
        if name == "circular":
            return np.roll(np.identity(n, dtype=ITYPE), shift=1, axis=0) +\
                   np.roll(np.identity(n, dtype=ITYPE), shift=-1, axis=0)
        if name == "linear":
            return np.eye(n, n, 1, dtype=ITYPE) + np.eye(n, n, -1, dtype=ITYPE)
    raise ValueError(f"No connectivity found under name '{str(name)}'.")

def available_connectivities() -> Dict:
    return {key : val[DESCR_KEY] for key, val in get_internal_connectivities().items()} + _automated_cons