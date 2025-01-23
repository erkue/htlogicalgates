# Hardware-Tailored Logical Gates

This package can be used to compile circuit implementations for **logical Clifford gates** of **quantum error-correcting codes**. The main features are

- works for **every stabilizer code** (runtime increases with code size),
- works for **every logical Clifford gate**, and
- by providing a connectivity map of qubits, **hardware-tailored** circuits can be obtained.

During circuit compilation, the number of two-qubit gates is minimized. By constructing **hardware-tailored** circuits, further qubit permutations are avoided altogether.  

## Requirements

A list of Python package dependencies is included in [pyproject.toml](pyproject.toml) and are automatically installed together with the package.

Furthermore, a valid [Gurobi](https://www.gurobi.com/) license is neccesary for the circuit compilation. There exists a wide-range of licences, including one for academic use that is free of charge. You can find further information on the [Gurobi downloads page](https://www.gurobi.com/downloads/).

## Installation

This Python package is available on [PyPi]() and can be installed using `pip` via

```
pip install htlogicalgates
```
Alternatively, you can clone this repository and include it in your project.

## Basic example ##

The package can be used in the following way

```py
from htlogicalgates import *

conn = get_conn("circular", n=4)
qecc = get_qecc("4_2_2")
log_gate = get_circuit("H 0", 2)

circ = tailor_logical_gate(qecc, conn, log_gate, 2)
```

Here, we construct a logical gate for the $⟦4,2,2⟧$-code, as indicated by the argument `"4_2_2"`. We use a `"circular"`-connectivity and search a circuit implementation for `"H 0"`, the Hadamard gate on logical qubit 0. The ansatz consists of `2` controlled-Z gate layers.