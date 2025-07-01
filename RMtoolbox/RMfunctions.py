import json

import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from cmath import log as clog
from cmath import exp as cexp
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
import qiskit.quantum_info as qi
from collections import Counter
import os


def x_u(cir_obj, counter_measured_strings, n_shots):
    sigma = 0
    sub_system_size = cir_obj.get_cal_size()
    for i, m_str in enumerate(counter_measured_strings):
        for k, next_str in enumerate(counter_measured_strings):
            ham = int(-1 * (hamming(list(m_str), list(next_str))) * len(list(m_str)))
            sigma += (pow(-2, ham) * (counter_measured_strings[m_str]/n_shots) * (counter_measured_strings[next_str]/n_shots))
    value = (pow(2, sub_system_size)) * sigma
    return value


def randomized_measurement(cir_obj, backend, n_shots):
    # Perform measurements (execute circuit)
    job = execute(cir_obj.get_circuit(), backend, shots=n_shots, memory=True)
    data = job.result().get_memory()
    str_counter = Counter(data)
    x_u_value = x_u(cir_obj, str_counter, n_shots)
    return x_u_value, cir_obj.get_u()

