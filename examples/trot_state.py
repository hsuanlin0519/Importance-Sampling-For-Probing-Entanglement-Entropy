import argparse, time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit import IBMQ, QuantumCircuit, execute,  Aer
from math import pi
from qiskit.circuit import Parameter
from qiskit import Aer
font = {'family': 'sans',
        'color': 'black',
        'weight': 'normal',
        'size': 18,
        }
params = {'legend.fontsize': 16,
          'legend.handlelength': 1,
          'xtick.labelsize': 'large',
          'font.size': 18,}
plt.rcParams.update(params)


backend_sim = Aer.get_backend('qasm_simulator')
vectorsim=BasicAer.get_backend('statevector_simulator')
dt = Parameter('dt')


def ZZbase(J=1):
    ZZ_qr = QuantumRegister(2)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

    ZZ_qc.cnot(0,1)
    ZZ_qc.rz( 2*J*dt, 1)
    ZZ_qc.cnot(0, 1)
    ZZ = ZZ_qc.to_instruction()
    return ZZ


def YYbase(ZZ):
    YY_qr = QuantumRegister(2)
    YY_qc = QuantumCircuit(YY_qr, name='YY')
    YY_qc.s(YY_qr)
    YY_qc.h(YY_qr)
    YY_qc.append(ZZ, [0,1])
    YY_qc.h(YY_qr)
    YY_qc.sdg(YY_qr)
    YY = YY_qc.to_instruction()
    return YY


def XXbase(ZZ):
    XX_qr = QuantumRegister(2)
    XX_qc = QuantumCircuit(XX_qr, name='XX')
    XX_qc.h(XX_qr)
    XX_qc.append(ZZ, [0,1])
    XX_qc.h(XX_qr)
    XX = XX_qc.to_instruction()
    return XX




def createeven(J=1, tfactor=1,PBC=False):
    ZZ=ZZbase(J=J*tfactor)

    XX = XXbase(ZZ)
    YY=YYbase(ZZ)
    even_qr = QuantumRegister(num_qubits)
    even_qc = QuantumCircuit(even_qr, name='even')
    if PBC:
        for i in range(0,num_qubits,2):
            even_qc.append(YY,[i,(i+1)%num_qubits])
            even_qc.append(XX,[i,(i+1)%num_qubits])
    else:
        for i in range(0,num_qubits-1,2):
            even_qc.append(YY,[i,i+1])
            even_qc.append(XX,[i,i+1])
    even_gate = even_qc.to_instruction()
    return even_gate


def createodd(Jp=1,PBC=False):
    ZZo=ZZbase(J=Jp)
    XXo = XXbase(ZZo)
    YYo=YYbase(ZZo)
    odd_qr = QuantumRegister(num_qubits)
    odd_qc = QuantumCircuit(odd_qr, name='odd')
    if PBC:
        for i in range(1,num_qubits,2):
            odd_qc.append(YYo,[i,(i+1)%num_qubits])
            odd_qc.append(XXo,[i,(i+1)%num_qubits])
    else:
        for i in range(1,num_qubits-1,2):
            odd_qc.append(YYo,[i,i+1])
            odd_qc.append(XXo,[i,i+1])
    odd_gate = odd_qc.to_instruction()
    return odd_gate


def onetrotstep(J,Jp,PBC):
    tfactor=1
    if args.order==2: tfactor=1/2
    even_gate=createeven(J=J, tfactor=tfactor,PBC=PBC)
    odd_gate=createodd(Jp=Jp,PBC=PBC)

    Trot_tb_qr = QuantumRegister(num_qubits)
    Trot_tb_qc = QuantumCircuit(Trot_tb_qr)

    Trot_tb_qc.append(even_gate,[i for i in range(num_qubits)])
    Trot_tb_qc.append(odd_gate,[i for i in range(num_qubits)])
    if args.order==2:
        Trot_tb_qc.append(even_gate,[i for i in range(num_qubits)])
    #Trot_tb_qc.draw()
    Trot_tb_gate = Trot_tb_qc.to_instruction()
    return Trot_tb_gate


def create_exactcircls(exls=[0] ,J=1,Jp=1,delta_t=0.1,time_steps=np.arange(1,40,1)):
    circuits=[]
    PBC=True
    if args.bc == 'obc' or args.bc == 'OBC':
        PBC=False
    Trot_tb_gate = onetrotstep(J=J, Jp=Jp, PBC=PBC)
    for n_steps in time_steps:
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits/2)
        qc = QuantumCircuit(qr, cr)
        if exls==None:
            ## prepare Bell basis as initial state
            qc.x(qr)
            for i in range(num_qubits):
                if i%2 == 0:qc.h(qr[i])
                if PBC:
                    qc.cx(qr[i], qr[(i+1)%num_qubits])
                else:
                    if i+1<num_qubits and i%2==0:
                        qc.cx(qr[i], qr[(i+1)])
        else:
            ## prepare Neel state as initial state
            qc.x(exls)
        qc.append(Trot_tb_gate, [i for i in range(num_qubits)])
        #qc.measure(qr, cr)
        qc = qc.bind_parameters({dt: delta_t*n_steps})
        circuits.append(qc)
    return circuits


## To run,
## command line
## python trot_state.py
## It returns a list of circuits and show the drawing
t1=time.time()
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--N', type=int, help='num_qubits', default=8)
parser.add_argument('--J', type=float, help='even bond strength', default=0)
parser.add_argument('--Jp', type=float, help='bond strength', default=-1)
parser.add_argument('--bc', type=str, help='PBC or OBC', default='obc')
parser.add_argument('--exls', type=list, help='initial excitation, which qubits set to |1>', default='N')
parser.add_argument('--order', type=int, help='Trotter order', default=2)
args = parser.parse_args()
num_qubits = args.N
subddate='{:02.0f}{:02.0f}{}'.format(time.localtime().tm_mon, time.localtime().tm_mday, time.localtime().tm_year)
if args.exls == ['N']:
    exls = None # this is for creating Bell basis for initial state
else:
    exls = [int(i) for i in args.exls]
#twisttime(exls=exls,subdate=subddate)


def get_trot_state(delta_t, time_value):
    time_steps = np.array([time_value])
    circuits = create_exactcircls(exls, args.J, args.Jp, delta_t*pi, time_steps)
    return circuits[0]

"""
IBMQ.load_account()
provider = IBMQ.get_provider(
    hub='ibm-q-hub-ntu', group='ntu-internal', project='default')
backend = Aer.get_backend('qasm_simulator')
qc = get_state(0.05, 0)
cir_set = transpile(qc, backend=backend)
result = backend.run(cir_set, shots=10000).result()
counts = result.get_counts(0)
print(counts)
"""