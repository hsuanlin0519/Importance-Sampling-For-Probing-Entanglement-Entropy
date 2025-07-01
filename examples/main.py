import math
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datavisual import *
from func import *
from dotenv import load_dotenv
import os
from decomp import udecomp, compare, recover_u, reconstruct
from classdef import AngleParameters, NeuralNet, PredictionManager, Predictions, Circuit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import random


# Load Datafiles for ML/Calculation
pur_f = open('../RMtoolbox/resultsRM_6/xResult.json')
matrix_f = open('../RMtoolbox/resultsRM_6/unitary.json')

# Training Parameters , Please Manually Insert System Size
n_qub = 3  # total system size
n_cb = 3  # amount of classical bits
half_qub = math.ceil(n_qub / 2)
n_epoch = 200
n_batchSize = 100
learning_rate = 0.001
n_metropolis_steps = 5

# set Qiskit configuration
IBMQ.save_account(os.getenv("IBM_KEY"), overwrite=True)

IBMQ.load_account()
# Manually Set Boolean of circuit type 0 for product state , 1 for GHZ State
provider = IBMQ.get_provider(hub='ibm-q-hub-ntu', group='ntu-internal', project='default')
backend = provider.get_backend("ibm_sherbrooke")
b_circuit_state = backend.properties().t1(0)/10

# initialize a experiment log
init_log(b_circuit_state, n_qub)

# feed data file to preprocess
label, angle, phase = data_preprocess(pur_f, matrix_f, n_cb)


# Dataset Declaration , split into train & validation set
dataset = AngleParameters(label, angle, phase)
train_set, val_set = torch.utils.data.random_split(dataset, [80000, 20000])
train_loader = DataLoader(dataset=train_set, batch_size=n_batchSize, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=n_batchSize, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = NeuralNet(2 * n_cb).to(device)
# Set Loss L1Loss() => Mean Absolute Error, MSELoss() => Mean Square Error
LossFunc = nn.L1Loss()
train_loss = []
valid_loss = []
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
answers = np.empty(0, dtype=float)

# training loop
for epoch in range(n_epoch):
    epo_loss = 0
    for i, (labels, angles, phase) in enumerate(train_loader):
        # forward
        outputs = net(angles.float())
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        loss = LossFunc(outputs.float(), labels.float())
        epo_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % len(train_loader) == 0:
            print(f'Epoch [{epoch + 1}/{n_epoch}], Training Loss: {(epo_loss / (i + 1)):.4f}')
            train_loss.append((epo_loss.detach().numpy() / (i + 1)))

    with torch.no_grad():
        val_epo_loss = 0
        for c, (labels, angles, phase) in enumerate(val_loader):
            predicted = net(angles.float())
            # Collect predictions of the last Epoch
            if (epoch + 1) == n_epoch:
                answers = np.append(answers, predicted)
            predicted = predicted.view(-1)
            labels = labels.view(-1)
            loss = LossFunc(predicted.float(), labels.float())
            val_epo_loss += loss
            if (c + 1) % len(val_loader) == 0:
                print(f'Epoch [{epoch + 1}/{n_epoch}], Validation Loss: {(val_epo_loss / (c + 1)):.4f}')
                valid_loss.append((val_epo_loss / (c + 1)))

if len(valid_loss) == len(train_loss):
    write_log("final train loss is: ", train_loss[-1])
    write_log("final valid loss is: ", valid_loss[-1])
    plot_loss(train_loss, valid_loss)
else:
    print("Loss List Error.")

if len(answers) == len(val_set):
    plot_label(label)
    plot_predictions(answers)
else:
    print("Answers amount Error.")

# Create a behavior oriented class to organize all the Predictions
Ans_Manager = PredictionManager(b_circuit_state)

for k in range(0, len(answers)):
    p = Predictions()
    Ans_Manager.predict_obj.append(p)
    Ans_Manager.predict_obj[k].answers = answers[k]

    labels, angles, phases = val_loader.dataset.__getitem__(k)
    Ans_Manager.predict_obj[k].raw_label = labels
    Ans_Manager.predict_obj[k].angle = angles
    Ans_Manager.predict_obj[k].phase = phases

# check if class object amounts are correct , then print a example randomly
if len(Ans_Manager.predict_obj) == len(val_set):
    r = random.randint(0, len(val_set)-1)

# metropolis sample
for _ in range(0, n_metropolis_steps):
    metropolis(Ans_Manager, answers, 50)
    Ans_Manager.update_metro_dict()
Ans_Manager.calculate_avg_prediction()
write_log("Acceptance rate is: ", sum(Ans_Manager.acceptance_rate_list)/len(Ans_Manager.acceptance_rate_list))
plot_metropolis(answers, Ans_Manager.metro_data_list)


# set purity estimation

rl = []
for i in range(0, 100):
    rl.append(random.randrange(0, len(Ans_Manager.predict_obj)))

Ans_Manager.circuit_type = -1
n_pick_u = 100
n_shots = 50
Ans_Manager.find_top_metro(n_pick_u)
p2_special(Ans_Manager,  n_qub, n_cb, n_shots, rl, backend)


end_log()
