import math
# import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from scipy.spatial.distance import hamming
import torch
import torch.nn as nn
from qiskit import (
    IBMQ, Aer, execute,
    QuantumRegister, ClassicalRegister, QuantumCircuit, quantum_info
)
from qiskit import transpile, assemble
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.quantum_info import random_unitary
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control
from trot_state import *
import operator
import os
import json
from RMdatavisual import *


class Circuit:
    def __init__(self, q=0, c=0):
        self.n_qubits = q
        self.n_cbits = c
        q_reg = QuantumRegister(self.n_qubits)
        c_reg = ClassicalRegister(self.n_cbits)
        self.qc = QuantumCircuit(q_reg, c_reg)
        self.cal_size = None
        self.circuit_u = []

    def run_ghz(self):
        self.qc.h(0)
        for p in range(0, (self.n_qubits-1)):
            self.qc.cx(p, p+1)

    def run_product(self):
        return 0

    def run_plus_product(self):
        self.qc.h(0)
        return 0

    def run_one_product(self):
        for i in range(0, self.n_qubits):
            self.qc.x(i)
        return 0

    def run_topological(self):
        for p in range(0, self.n_cbits):
            self.qc.cz(p*2, p*2+1)
        for p in range(0, self.n_cbits-1):
            self.qc.cz(p * 2+1, p * 2+2)
        self.qc.cz(0, (self.n_qubits-1))

    def run_quench_bell(self, delta_t, time_step):
        self.qc = get_trot_state(delta_t, time_step)

    def get_circuit(self):
        return self.qc

    def get_cal_size(self):
        return self.cal_size

    def get_u(self):
        return self.circuit_u

    def prepare_measurements(self):
        if self.n_cbits == self.n_qubits:
            self.prepare_full_system()
        elif self.n_cbits == self.n_qubits/2:
            self.prepare_bipartite()

    def prepare_bipartite(self):
        self.cal_size = int(self.n_qubits / 2)
        if self.cal_size % 2 == 1 or self.n_cbits != self.cal_size:
            print("Error, This function doesn't support this system size setting.")
            return -1
        self.qc.barrier()
        for i in range(self.cal_size, self.cal_size*2):
            # Pull a random unitary
            u = random_unitary(2)
            self.circuit_u.append(u.data)
            randUnitary = UnitaryGate(u, label='Unitary')
            self.qc.append(randUnitary, [i])
        for i in range(self.cal_size, self.cal_size*2):
            # apply bipartite measurements (bottom half)
            self.qc.measure([i], [i-self.cal_size])

    def prepare_full_system(self):
        self.cal_size = self.n_qubits
        if self.n_cbits != self.cal_size:
            print("Error, Function doesn't support systems with different sizes of qubits and cbits.")
            return -1
        self.qc.barrier()
        for i in range(0, self.cal_size):
            # Pull a random unitary
            u = random_unitary(2)
            self.circuit_u.append(u.data)
            randUnitary = UnitaryGate(u, label='Unitary')
            self.qc.append(randUnitary, [i])
            # apply full system measurements
        for i in range(0, self.cal_size):
            # apply bipartite measurements (bottom half)
            self.qc.measure([i], [i])


class RMManger:
    def __init__(self, n_s):
        self.Ns = n_s
        self.rmResult_lst = []
        self.rmU_lst = []
        self.cirObj_lst = []
        self.purity = None
        self.path = None
        self.deviation = None

    def purity_calculation(self):
        self.purity = sum(self.rmResult_lst)/len(self.rmResult_lst)
        return self.purity

    def get_deviation(self):
        self.deviation = np.std(self.rmResult_lst)
        return self.deviation

    def get_purity(self):
        return self.purity

    def create_log_path(self):
        self.path = './resultsRM_{}'
        for i in range(1, 100):
            if not os.path.exists(self.path.format(i)):
                os.mkdir(self.path.format(i))
                self.path = self.path.format(i)
                print("Experiment Directory Successfully Created.")
                return True
        print("Experiment Directory Creation Failed (Reached max index number of directory automatic creation).")
        return False

    def finalize_output_files(self, qc, n_shots):
        # Set file path for output files
        u_file = open(self.path + "/unitary.json", "a")
        x_file = open(self.path + "/xResult.json", "a")
        log_file = open(self.path + "/expLog.txt", "a")

        # Output the X file
        j_str = json.dumps(self.rmResult_lst, indent=2)
        for i in j_str:
            x_file.write(i)

        # Output the U file
        dict_u = {}
        for index, i in enumerate(self.rmU_lst):
            tmp = []
            for k in i:
                u = np.array2string(k)
                tmp.append(u)
            dict_u[str(index)] = tmp
        j_str2 = json.dumps(dict_u, indent=2)
        u_file.write(j_str2)

        # Output Experiment Log
        log_file.write("Circuits: " + str(self.Ns) + "\n")
        log_file.write("shots: " + str(n_shots) + "\n")
        log_file.write("Purity: " + str(self.get_purity()) + "\n")
        log_file.write("Standard Deviation of X(u): " + str(self.get_deviation()) + "\n")
        plot_circuit(qc, self.path + "/Circuit.png")
