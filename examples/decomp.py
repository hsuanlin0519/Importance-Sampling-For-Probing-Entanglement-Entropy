import json

import numpy as np
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control
# Loading your IBM Quantum account(s)
from math import pi
from qiskit import (
    IBMQ, Aer, execute,
    QuantumRegister, ClassicalRegister, QuantumCircuit, quantum_info
)
import matplotlib.pyplot as plt
from cmath import log as clog
from cmath import exp as cexp
import scipy
from math import atan, sin, cos


def wrong_distribution(n):
    # A Random matrix with the wrong distribution
    z = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    return q


def reconstruct(phase, alpha, theta, beta):
    Rz = np.array([[cexp(1.j*alpha/2), 0], [0, cexp(-1.j*alpha/2)]])
    tmp1 = np.dot(np.array([[1, 0], [0, phase]]), Rz)
    tmp2 = np.dot(np.array([[cos(theta/2), sin(theta/2)], [-sin(theta/2), cos(theta/2)]]), np.array([[cexp(1.j*beta/2), 0], [0, cexp(-1.j*beta/2)]]))
    final = np.dot(tmp1, tmp2)
    return final, tmp2


def get_atan(x, y):
    theta = atan(y/x) # this is in between -pi/2, pi/2
    if y > 0 and x < 0:
        theta = pi+theta
    elif y < 0 and x < 0:
        theta = pi+theta
    elif y < 0 and x > 0:
        theta = 2*pi+theta
    return theta


def udecomp(u):
    u1 = u[0, 0]
    u2 = u[0, 1]
    u3 = u[1, 0]
    u4 = u[1, 1]
    phase = -u3/np.conjugate(u2)
    su = np.dot(np.array([[1, 0], [0, np.conjugate(phase)]]), u)
    x = su[0, 0].real
    y = su[0, 0].imag
    p = su[0, 1].real
    q = su[0, 1].imag
    alpha = get_atan(x, y)+get_atan(p, q)
    beta = get_atan(x, y)-get_atan(p, q)
    tmpx = su[0, 0]*cexp(-1.j*(alpha+beta)/2)
    tmpy = su[0, 1]*cexp(-1.j*(alpha-beta)/2)
    theta = 2*get_atan(tmpx.real, tmpy.real)
    return phase, alpha, theta, beta


def checkdist():
    Num = 1000
    data = []
    odata = []
    for _ in range(Num):
        tu = random_unitary(2)
        ev = np.linalg.eig(tu.data)
        odata.append(clog(ev[0][0]).imag)
        odata.append(clog(ev[0][1]).imag)
        phase, alpha, beta, theta = udecomp(tu.data)
        final = reconstruct(phase, alpha, beta, theta)
        #tu=wrong_distribution(2)
        #ax,ay,az,ai=np.trace(np.dot(tu,px))/2,np.trace(np.dot(tu,py))/2,np.trace(np.dot(tu,pz))/2,np.trace(np.dot(tu,pi))/2
        #u1,u2,u3,u4=recoveru(ax,ay,az)
        #u1,u2,u3,u4,phase=u2su(tu.data)
        #supart=np.array([[u1,u2],[u3,u4]])
        #print(supart,np.linalg.det(supart))
        #u=np.sqrt(phase)*np.dot(np.array([[1/np.sqrt(phase),0],[0,np.sqrt(phase)]]),np.array([[u1,u2],[u3,u4]]))
        #if abs(abs(np.linalg.det(u))-1)>1e-6:print('error det')
        #if np.max(abs(tu.data-u))>1e-6:print('different err')
        ev = np.linalg.eig(final)
        data.append(clog(ev[0][0]).imag)
        data.append(clog(ev[0][1]).imag)
    np.array(data).flatten()
    np.array(odata).flatten()
    plt.hist(data)
    plt.hist(odata, histtype=u'step')
    plt.show()


def compare(tu):
    phase, alpha, theta, phi = udecomp(tu)
    final, tmp2 = reconstruct(phase, alpha, theta, phi)

    if tu.all() == final.all():
        return phase, alpha, theta, phi
    else:
        print("Error ! Different")


def recover_u(ax, ay, az):
    u2 = ax - 1.j * ay
    u3 = ax + 1.j * ay
    phase = -u3 / np.conjugate(u2)
    lena = np.sqrt(1 - abs(u2) ** 2)
    tmp_beta = az / np.sqrt(phase) / 1.j / lena

    criteria = 0
    test = 0
    while abs(criteria - 1) > 1e-6:
        if test == 0:
            tmp_beta = np.arcsin(tmp_beta)
            # if test==1:
        #    tmpbeta=np.arcsin(tmpbeta)
        #   if tmpbeta>0:tmpbeta=np.pi-tmpbeta;print('1')
        #   if tmpbeta <0 :tmpbeta=-np.pi-tmpbete;print('2')

        test += 1

        tmp = clog(np.sqrt(phase)).imag

        beta = tmp + tmp_beta

        u1 = lena * cexp(1.j * beta)

        u4 = np.conjugate(u1) * phase
        recovered = np.array([[u1, u2], [u3, u4]])
        criteria = abs(np.linalg.det(recovered))
        # print(criteria)

    return recovered
