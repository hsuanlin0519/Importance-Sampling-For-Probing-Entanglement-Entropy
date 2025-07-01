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
from qiskit_ibm_provider import IBMProvider
from qiskit.extensions import UnitaryGate
from datavisual import *
from classdef import *
from cmath import log as clog
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, coherent_unitary_error
import qiskit.quantum_info as qi
from decomp import compare, udecomp, recover_u, reconstruct


def data_preprocess(lab_f, max_f, c_bit):
    data2 = json.load(lab_f)
    m_data = json.load(max_f)
    label = np.array(data2, dtype=float)
    angle = np.zeros((len(label), 2 * c_bit), dtype=float)
    phase_part = np.zeros((len(label), 2 * c_bit), dtype=complex)
    for k in range(0, len(label)):
        c = str(k)
        if c_bit != len(m_data[c]):
            print("Size of input data is invalid. (Should be the same with c_bit size)")
            return -1
        for i in range(0, len(m_data[c])):
            t = matrix_to_angles(m_data[c][i]).data
            pha, alpha, theta, phi = compare(t)
            angle[k][i] = theta
            angle[k][i+c_bit] = phi
            phase_part[k][i] = pha
            phase_part[k][i+c_bit] = alpha
    return label, angle, phase_part


def matrix_to_angles(data):
    a = data.replace('[', '')
    a = a.replace(']', '')
    a = a.replace('\n', '')
    a = a.replace(' ', '')
    a = a.replace('j', "j ")
    a = a.split(' ')
    m = Operator([[complex(a[0]), complex(a[1])], [complex(a[2]), complex(a[3])]])
    return m


def metropolis(manager, input_arr, burn_in):
    # initialize metropolis first step
    current = input_arr[0]
    current_index = 0
    accept_count = 0
    # start looping
    for i in range(1, len(input_arr)):
        # produce random uniform number
        rand = random.uniform(0, 1)
        value = min(1, (input_arr[i] / current))
        # accept new step
        if rand <= value:
            current = input_arr[i]
            current_index = i
            accept_count += 1
            if i >= burn_in:
                manager.metro_data_list.append(current)
                manager.metro_index_list.append(current_index)
        # reject new step
        else:
            if i >= burn_in:
                manager.metro_data_list.append(current)
                manager.metro_index_list.append(current_index)
    # store acceptance rate to calculate acceptance rate for multiple steps of Metropolis Sampling
    manager.acceptance_rate_list.append(accept_count/len(input_arr))
    return 0


def prepare_circuit(ans_obj, circuit_type, q, c, shots):
    n_qub = q
    n_cb = c
    CirObj = Circuit(n_qub, n_cb, shots, circuit_type)
    qc = CirObj.get_circuit()
    # Unitary recovered then assigned on to the circuit
    for p in range(0, n_cb):
        # 取出角度和phase，重組unitary
        angle_part = ans_obj.get_angle()
        phase_part = ans_obj.get_phase()
        phase = phase_part[p]
        alpha = phase_part[p + n_cb]
        theta = angle_part[p]
        phi = angle_part[p + n_cb]
        # assign unitary to circuit
        final, u = reconstruct(phase, alpha, theta, phi)
        randUnitary = UnitaryGate(u, label='Unitary')
        qc.append(randUnitary, [p])
    # check system q_bit c_bit setups to decide executing full-system or bipartite measuring
    if n_qub == n_cb:
        for g in range(0, n_cb):
            qc.measure([g], g)
    elif n_cb == n_qub/2:
        for g in range(0, n_cb):
            qc.measure([g+n_cb], g)
    else:
        print("Invalid q_bit & c_bit combination")
        return -1
    # perform qiskit job execution
    return qc


def p_is(x_is, avg_u):
    value = x_is / avg_u
    return value


def p2_is(manager, q, c, shots, backend):
    sigma = 0
    if len(manager.dict_metro) == 0:
        print("Empty Dictionary ERROR!")
    else:
        # declare a new class object Circuit Manager
        provider = IBMQ.get_provider(hub='ibm-q-hub-ntu', group='ntu-internal', project='default')
        cir_manager = CircuitManager(provider)

        # prepare and run all circuits with job manager
        for u in manager.top_metro_dict:
            qc = prepare_circuit(manager.predict_obj[u], manager.circuit_type, q, c, shots)
            cir_manager.add_to_set(qc)
        job_result = cir_manager.transpile_and_run(shots, backend)

        # calculation of purity
        for index, r in enumerate(manager.top_metro_dict):
            p_is_val = p_is(manager.predict_obj[r].get_answers(), manager.avg_x_is)
            sigma += (manager.top_metro_dict[r] * x_e(job_result.get_memory(index), manager.predict_obj[r], shots)) / p_is_val
            print(job_result.get_counts(index))
        value = (1 / sum(manager.top_metro_dict.values())) * sigma
        ab_err = np.abs((manager.absolute - value))
        write_exp_log_metro(backend, shots, len(manager.top_metro_dict), value, ab_err)
        return ab_err


def x_e(result_memory, ans_obj, shots):
    sigma = 0
    sub_system_size = int(len(ans_obj.get_angle()) / 2)
    measured_strings = result_memory
    for i in range(0, shots):
        for k in range(0, shots):
            if i != k:
                ham = int(-1 * (hamming(list(measured_strings[i]), list(measured_strings[k])) * len(
                    list(measured_strings[i]))))
                sigma += pow(-2, ham)
    value = (pow(2, sub_system_size) / (shots * (shots - 1))) * sigma
    return value


def p2_uniform(manager, q, c, shots, rand_pick_num, backend):
    sigma = 0
    if len(manager.dict_metro) == 0:
        print("Empty Dictionary ERROR!")
    else:
        # randomly pick unitary
        raw_rand = []
        for i in range(0, rand_pick_num):
            raw_rand.append(random.choice(manager.predict_obj))

        # declare a new class object Circuit Manager
        provider = IBMQ.get_provider(hub='ibm-q-hub-ntu', group='ntu-internal', project='default')
        cir_manager = CircuitManager(provider)
        # prepare and run all circuits with job manager
        for u in raw_rand:
            qc = prepare_circuit(u, manager.circuit_type, q, c, shots)
            cir_manager.add_to_set(qc)

        job_result = cir_manager.transpile_and_run(shots, backend)

        # calculation of purity
        for index, r in enumerate(raw_rand):
            sigma += (x_e(job_result.get_memory(index), r, shots)) / 1
        value = (1 / len(raw_rand)) * sigma
        ab_err = np.abs((manager.absolute - value))
        write_exp_log_uniform(backend, shots, rand_pick_num, value, ab_err)
        return ab_err


def p2_specify_uniform(manager, q, c, shots, rand_list, backend):
    sigma = 0
    if len(manager.dict_metro) == 0:
        print("Empty Dictionary ERROR!")
    else:
        # randomly pick unitary
        raw_rand = []
        for i in rand_list:
            raw_rand.append(manager.predict_obj[i])
        print(raw_rand)
        # declare a new class object Circuit Manager
        provider = IBMQ.get_provider(hub='ibm-q-hub-ntu', group='ntu-internal', project='default')
        cir_manager = CircuitManager(provider)
        # prepare and run all circuits with job manager
        for u in raw_rand:
            qc = prepare_circuit(u, manager.circuit_type, q, c, shots)
            cir_manager.add_to_set(qc)

        job_result = cir_manager.transpile_and_run(shots, backend)

        # calculation of purity
        for index, r in enumerate(raw_rand):
            sigma += (x_e(job_result.get_memory(index), r, shots)) / 1
        value = (1 / len(raw_rand)) * sigma
        ab_err = np.abs((manager.absolute - value))
        write_exp_log_uniform(backend, shots, len(rand_list), value, ab_err)
        return ab_err


def p2_uniform_local(manager, q, c, shots, rand_pick_num, backend_str):
    sigma = 0
    if len(manager.dict_metro) == 0:
        print("Empty Dictionary ERROR!")
    else:
        raw_rand = []
        for i in range(0, rand_pick_num):
            raw_rand.append(random.choice(manager.predict_obj))
        for r in raw_rand:
            sigma += (x_e_local(r, manager.circuit_type, q, c, shots, backend_str)) / 1
        value = (1 / len(raw_rand)) * sigma
        ab_err = np.abs((manager.absolute - value))
        write_exp_log_uniform(backend_str, shots, rand_pick_num, value, ab_err)
        return ab_err


def x_e_local(ans_obj, circuit_type, q, c, shots, backend_str):
    sigma = 0
    sub_system_size = int(len(ans_obj.get_angle()) / 2)
    qc = prepare_circuit(ans_obj, circuit_type, q, c, shots)
    job = execute(qc, backend_str, shots=shots, memory=True)
    measured_strings = job.result().get_memory()
    for i in range(0, shots):
        for k in range(0, shots):
            if i != k:
                ham = int(-1 * (hamming(list(measured_strings[i]), list(measured_strings[k])) * len(
                    list(measured_strings[i]))))
                sigma += pow(-2, ham)
    value = (pow(2, sub_system_size) / (shots * (shots - 1))) * sigma
    print(value)
    return value


def p2_is_local(manager, q, c, shots, backend_str):
    sigma = 0
    if len(manager.dict_metro) == 0:
        print("Empty Dictionary ERROR!")
    else:
        for r in manager.top_metro_dict:
            p_is_val = p_is(manager.predict_obj[r].get_answers(), manager.avg_x_is)
            print(manager.top_metro_dict[r], " ", manager.predict_obj[r].get_answers(), " ", p_is_val, "raw label= ",
                  manager.predict_obj[r].get_raw_label())
            sigma += (manager.top_metro_dict[r] * x_e_local(manager.predict_obj[r], manager.circuit_type, q, c, shots, backend_str)) / p_is_val
        value = (1 / sum(manager.top_metro_dict.values())) * sigma
        ab_err = np.abs((manager.absolute - value))
        write_exp_log_metro(backend_str, shots, len(manager.top_metro_dict), value, ab_err)
        return ab_err


def p2_special(manager, q, c, shots, rand_list, backend):
    sigma = 0
    sigma2 = 0
    sigma3 = 0
    sigma4 = 0
    sigma5 = 0
    if len(manager.dict_metro) == 0:
        print("Empty Dictionary ERROR!")
    else:
        # randomly pick unitary
        raw_rand = []
        for i in rand_list:
            raw_rand.append(manager.predict_obj[i])
        print(raw_rand)
        # declare a new class object Circuit Manager
        provider = IBMQ.get_provider(hub='ibm-q-hub-ntu', group='ntu-internal', project='default')
        cir_manager = CircuitManager(provider)
        # prepare and run all circuits with job manager
        for u in raw_rand:
            qc = prepare_circuit(u, manager.circuit_type, q, c, shots)
            cir_manager.add_to_set(qc)
        job_result = cir_manager.transpile_and_run(shots, backend)

        # calculation of purity
        for index, r in enumerate(raw_rand):
            sigma += (x_e_special(job_result.get_memory(index), 0, 1, shots)) / 1
        value = (1 / len(raw_rand)) * sigma
        ab_err = np.abs((manager.absolute - value))
        write_exp_log_uniform(backend, shots, len(rand_list), value, ab_err)

        # calculation of purity
        for index, r in enumerate(raw_rand):
            sigma2 += (x_e_special(job_result.get_memory(index), 1, 1, shots)) / 1
        value2 = (1 / len(raw_rand)) * sigma2
        ab_err = np.abs((manager.absolute - value2))
        write_exp_log_uniform(backend, shots, len(rand_list), value2, ab_err)

        # calculation of purity
        for index, r in enumerate(raw_rand):
            sigma3 += (x_e_special(job_result.get_memory(index), 2, 1, shots)) / 1
        value3 = (1 / len(raw_rand)) * sigma3
        ab_err = np.abs((manager.absolute - value3))
        write_exp_log_uniform(backend, shots, len(rand_list), value3, ab_err)

        # calculation of purity
        for index, r in enumerate(raw_rand):
            nl = parse_arr(job_result.get_memory(index), [0, 1], shots)
            print(job_result.get_memory(index))
            print(nl)
            sigma4 += (x_e_group(nl, r, shots)) / 1
        value4 = (1 / len(raw_rand)) * sigma4
        ab_err = np.abs((manager.absolute - value4))
        write_exp_log_uniform(backend, shots, len(rand_list), value4, ab_err)

        # calculation of purity
        for index, r in enumerate(raw_rand):
            nl = parse_arr(job_result.get_memory(index), [0, 2], shots)
            print(job_result.get_memory(index))
            print(nl)
            sigma5 += (x_e_group(nl, r, shots)) / 1
        value5 = (1 / len(raw_rand)) * sigma5
        ab_err = np.abs((manager.absolute - value5))
        write_exp_log_uniform(backend, shots, len(rand_list), value5, ab_err)

        return ab_err


def x_e_special(result_memory, index, size, shots):
    sigma = 0
    sub_system_size = size
    measured_strings = result_memory
    for i in range(0, shots):
        for k in range(0, shots):
            if i != k:
                ham = int(-1 * (hamming(list(measured_strings[i][index]), list(measured_strings[k][index])) * len(
                    list(measured_strings[i][index]))))
                sigma += pow(-2, ham)
    value = (pow(2, sub_system_size) / (shots * (shots - 1))) * sigma
    return value


def x_e_group(result_memory, ans_obj, shots):
    sigma = 0
    sub_system_size = 2
    measured_strings = result_memory
    for i in range(0, shots):
        for k in range(0, shots):
            if i != k:
                ham = int(-1 * (hamming(list(measured_strings[i]), list(measured_strings[k])) * len(
                    list(measured_strings[i]))))
                sigma += pow(-2, ham)
    value = (pow(2, sub_system_size) / (shots * (shots - 1))) * sigma
    return value


def parse_arr(memory, index_list, shots):
    new_list = []
    for k in range(0, shots):
        temp = ""
        for i in index_list:
            temp = temp + memory[k][i]
        new_list.append(temp)
    return new_list
