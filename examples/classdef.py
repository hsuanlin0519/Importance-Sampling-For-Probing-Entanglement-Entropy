import math
# import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from scipy.spatial.distance import hamming
import torch
import torch.nn as nn
from qiskit import (
    Aer, execute, IBMQ,
    QuantumRegister, ClassicalRegister, QuantumCircuit, quantum_info
)
from qiskit import transpile, assemble
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit_ibm_provider import IBMProvider
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.quantum_info import random_unitary
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control
from trot_state import *
import operator


class AngleParameters(Dataset):
    def __init__(self, label, angle, phase_part):
        self.label = torch.from_numpy(label)
        self.angle = angle
        self.phase_part = phase_part
        self.n_labels = len(label)

    def __getitem__(self, index):
        return self.label[index], self.angle[index], self.phase_part[index]

    def __len__(self):
        return self.n_labels  # how many examples

    def get_all_data(self):
        return self.label, self.angle, self.phase_part


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 200)
        self.l2 = nn.Linear(200, 100)
        self.l3 = nn.Linear(100, 10)
        self.l4 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(200)
        self.norm2 = nn.BatchNorm1d(100)
        self.norm3 = nn.BatchNorm1d(10)

    def forward(self, x):
        # no activation and no softmax at the end
        x_out = self.l1(x)
        x_out = self.norm1(x_out)
        x_out = self.relu(x_out)
        x_out = self.l2(x_out)
        x_out = self.norm2(x_out)
        x_out = self.relu(x_out)
        x_out = self.l3(x_out)
        x_out = self.norm3(x_out)
        x_out = self.relu(x_out)
        out = self.l4(x_out)
        return out


class Predictions:
    def __init__(self):
        self.angle = None
        self.answers = None
        self.prob = None
        self.raw_label = None
        self.raw_label_prob = None
        self.phase = None

    def get_answers(self):
        return self.answers

    def get_angle(self):
        return self.angle

    def get_phase(self):
        return self.phase

    def get_prob(self):
        return self.prob

    def get_raw_label(self):
        return self.raw_label

    def get_raw_prob(self):
        return self.raw_label_prob


class PredictionManager:
    def __init__(self, circuit_type):
        self.predict_obj = []
        self.metro_index_list = []
        self.metro_data_list = []
        self.acceptance_rate_list = []
        self.acceptance_rate = 0
        self.top_metro_dict = {}
        self.dict_metro = {}
        self.avg_x_is = 0
        self.circuit_type = circuit_type
        # judge the type of circuit
        if circuit_type == 0:
            # product state
            self.absolute = 1
        elif circuit_type == 1:
            # GHZ state
            self.absolute = 0.5
        elif circuit_type == 2:
            # Trot state with time step = 0
            self.absolute = 1
        elif circuit_type == 3:
            # Trot state with time step = pi/8
            self.absolute = 0.5
        elif circuit_type == 4:
            # Trot state with time step = pi/4
            self.absolute = 0.25
        else:
            self.absolute = 1

    def find_top_metro(self, top):
        # 取得所需data之index以便取用各種資料
        dict_u = {}
        for index in self.metro_index_list:
            dict_u[index] = dict_u.get(index, 0) + 1
        self.top_metro_dict = dict(sorted(dict_u.items(), key=operator.itemgetter(1), reverse=True)[:top])
        print(self.top_metro_dict)
        return self.top_metro_dict

    def calculate_acceptance_rate(self):
        self.acceptance_rate = len(self.dict_metro)/len(self.metro_data_list)
        return self.acceptance_rate

    def update_metro_dict(self):
        self.dict_metro = {}
        for index in self.metro_index_list:
            self.dict_metro[index] = self.dict_metro.get(index, 0) + 1
        return self.dict_metro

    def calculate_avg_prediction(self):
        value = 0
        for u in self.predict_obj:
            value += u.get_answers()
        self.avg_x_is = value/len(self.predict_obj)
        return self.avg_x_is


class Circuit:
    def __init__(self, q=0, c=0, shots=0, circuit_type=None):
        self.n_qubits = q
        self.n_cbits = c
        self.shots = shots
        self.circuit_type = circuit_type
        q_reg = QuantumRegister(self.n_qubits)
        c_reg = ClassicalRegister(self.n_cbits)
        self.qc = QuantumCircuit(q_reg, c_reg)
        # detect circuit type and build certain quantum states
        if circuit_type == 0:
            self.run_product()
        elif circuit_type == 1:
            self.run_ghz()
        elif circuit_type == 2:
            self.run_trot(0.05, 0)
        elif circuit_type == 3:
            self.run_trot(0.05, 2.5)
        elif circuit_type == 4:
            self.run_trot(0.05, 5)
        else:
            self.run_single_delay(circuit_type)

    def run_ghz(self):
        self.qc.h(0)
        for p in range(0, (self.n_qubits-1)):
            self.qc.cx(p, p+1)

    def run_product(self):
        return 0

    def run_trivial(self):
        for p in range(0, self.n_qubits):
            self.qc.h(p)

    def run_topological(self):
        for p in range(0, self.n_cbits):
            self.qc.cz(p*2, p*2+1)
        for p in range(0, self.n_cbits-1):
            self.qc.cz(p * 2+1, p * 2+2)
        self.qc.cz(0, (self.n_qubits-1))

    def run_trot(self, delta_t, time_step):
        self.qc = get_trot_state(delta_t, time_step)

    def run_single_delay(self, delay_t):
        self.qc.x(0)
        self.qc.x(1)
        self.qc.x(2)
        if delay_t > 0:
            self.qc.delay(delay_t, unit='s')
        return 0

    def get_circuit(self):
        return self.qc


class CircuitManager:
    def __init__(self, provider):
        self.cir_set = []
        self.job_set_id_list = []
        self.job_manager = IBMQJobManager()
        self.job_set = None
        self.job_set_result = None
        self.provider = provider

    def add_to_set(self, circuit_obj):
        self.cir_set.append(circuit_obj)

    def transpile_and_run(self, shot, device):
        self.cir_set = transpile(self.cir_set, backend=device, initial_layout=[0, 1, 124], scheduling_method="alap")
        self.job_set = self.job_manager.run(self.cir_set, backend=device, memory=True, shots=shot)
        self.job_set_result = self.job_set.results()
        self.job_set_id_list.append(self.job_set.job_set_id())
        return self.job_set_result

    def query_job_result(self, index):
        retrieved_job = self.job_manager.retrieve_job_set(job_set_id=self.job_set_id_list[index], provider=self.provider)
        return retrieved_job.results()

