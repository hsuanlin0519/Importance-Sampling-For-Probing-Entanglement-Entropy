import random
import numpy as np
from qiskit import (
    execute
)
from qiskit import Aer
from qiskit.extensions import UnitaryGate
from cmath import log as clog
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, coherent_unitary_error
from dotenv import load_dotenv
import os
import qiskit.quantum_info as qi

from RMclassdef import *
from RMfunctions import *

# Set experiment parameters, you can always manually change these to fit your demand
n_qbits = 3
n_cbits = 3
n_shots = 1000
n_circuits = 100000

# Declare Behavior Oriented Class Object RMManager for X(u) and u with a parameter for number of circuits to measure
Manager = RMManger(n_circuits)

# Declare Circuit object and prepare certain Quantum state
# Append random Haar Measure unitary and measurements - options:(bipartite/full system)
backend = Aer.get_backend("qasm_simulator")
for n in range(0, n_circuits):
    Cir_q = Circuit(n_qbits, n_cbits)
    Cir_q.run_one_product()
    Cir_q.prepare_measurements()
    x, uList = randomized_measurement(Cir_q, backend, n_shots)
    Manager.cirObj_lst.append(Cir_q)
    Manager.rmResult_lst.append(x)
    Manager.rmU_lst.append(uList)

# purity calculation
Manager.purity_calculation()
print(Manager.get_purity())

# Create experiment log path and output result files
if Manager.create_log_path():
    Manager.finalize_output_files(Manager.cirObj_lst[0].get_circuit(), n_shots)


