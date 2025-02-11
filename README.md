# Hardware-Tailored Logical Gates
![PyPI - Version](https://img.shields.io/pypi/v/htlogicalgates)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/erkue/htlogicalgates/blob/main/LICENSE)
[![Tests](https://github.com/erkue/htlogicalgates/actions/workflows/run-tests.yml/badge.svg)](https://github.com/erkue/htlogicalgates/actions/workflows/run-tests.yml)
[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-UNITARY%20FOUNDATION-brightgreen.svg)](https://unitary.foundation)



This package can be used to compile circuit implementations for **logical Clifford gates** of **quantum error-correcting codes**. The main features are:

- works for **every stabilizer code** (runtime increases with code size),
- works for **every logical Clifford gate**, and
- by providing a connectivity map of qubits, **hardware-tailored** circuits can be obtained.

During circuit compilation, the number of two-qubit gates is minimized. By constructing **hardware-tailored** circuits, further qubit permutations are avoided altogether.  

## Requirements

A list of Python package dependencies is included in [pyproject.toml](pyproject.toml) and are automatically installed together with the package.

Furthermore, a valid [Gurobi](https://www.gurobi.com/) license is neccesary for the circuit compilation. There exists a wide-range of licences, including one for academic use that is free of charge. You can find further information on the [Gurobi downloads page](https://www.gurobi.com/downloads/).

## Installation

This Python package is available on [PyPI](https://pypi.org/project/htlogicalgates/) and can be installed using `pip` via

```
pip install htlogicalgates
```
Alternatively, you can clone this repository and include it in your project.

## License 

This package is distributed under the MIT License.

If you want to support work like this, please cite: _our paper (tba)_

## Tutorials

The following sections of this readme, along with more examples and guiding information, can be found as Jupyter notebook tutorials:
- [Basic Tutorial](doc/tutorial.ipynb)
- [Advanced Tutorial](doc/tutorial_tailor_multiple_gates.ipynb)




## How to tailor logical circuits

The main workflow for tailoring a circuit starts with creating three central objects:

```py
import htlogicalgates as htlg

stab_code = htlg.StabilizerCode("4_2_2")
connectivity = htlg.Connectivity("circular", num_qubits=4)
logical_gate = htlg.Circuit(2)
logical_gate.h(0)
```

After importing the package, we create three objects:

- First, we create a `StabilizerCode` for which we wish to find a logical circuit. In this example, we pass `"4_2_2"`, which selects the $⟦4,2,2⟧$ color code. Some common codes are predefined (see below), but custom codes can also be specified through a set of stabilizer generators. 

- Next, we create a `Connectivity` that stores connections between qubits on the target hardware. Two-qubit gates will only be allowed between connected qubits. For this example, we use a `"circular"` connectivity on `num_qubits=4` qubits. Other predefined connectivities can be queried via `htlg.available_connectivities()`. Moreover, a custom connectivity can be created from an adjacency matrix. 

- Finally, we initialize a `Circuit` with the number of logical qubits and add a Hadamard gate on the first qubit (note that we count **qubits starting at 0**). In the following we will tailor a circuit that implements the action of this circuit on the logical level of the stabilizer code.

Note that we created a `Connectivity` for $n=4$ qubits and a logical `Circuit` for $k=2$ qubits since we are using the $⟦n=4,k=2,2⟧$-code.

Now we can pass these objects to the function `tailor_logical_gate`:
```py
circ, status = htlg.tailor_logical_gate(stab_code, connectivity, logical_gate, num_cz_layers=2)
```
The parameter `num_cz_layers` determines the number of CZ gate layers in the ansatz circuit. Generally speaking, more CZ layers make the ansatz more expressive and can lead to circuits with less two-qubit gates in total, while increasing runtime. If you can not find a specific gate, try to increase the number of CZ gate layers.

The return value `status` indicates the state of the optimization:

- `"Optimal"` : The returned circuit is optimal in terms of two-qubit gates.
- `"Bound {n}"` : The returned circuit is not optimal in terms of two-qubit games but there is no circuit with less than $n$ two-qubit gates.
- `"Infeasible"` : There is no physical circuit for the given stabilizer code, connectivity, logical circuit, and number of CZ gate layer.
- `"Time out"` : A physical circuit was not found in the given time limit.

If the status message is `"Optimal"` or `"Bound {n}"`, then `circ` contains the physical circuit implementation. Otherwise, it is `None`.
