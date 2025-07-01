import json
import os
import numpy as np
from decomp import *

f = open("./resultsRM_3/unitary.json")
lf = open("./resultsRM_3/xResult.json")


def data_preprocess(lab_f, mat_f):
    data2 = json.load(lab_f)
    m_data = json.load(mat_f)
    half_q = len(m_data['0'])
    label = np.array(data2, dtype=float)
    angle = np.zeros((len(label), 2 * half_q), dtype=float)
    phase_part = np.zeros((len(label), 2 * half_q), dtype=complex)
    for k in range(0, len(label)):
        c = str(k)
        for i in range(0, len(m_data[c])):
            t = matrix_to_angles(m_data[c][i]).data
            pha, alpha, theta, phi = compare(t)
            angle[k][i] = theta
            angle[k][i+half_q] = phi
            phase_part[k][i] = pha
            phase_part[k][i+half_q] = alpha
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


label_arr, angle_arr, phase_arr = data_preprocess(lf, f)
print(label_arr)
print(angle_arr)