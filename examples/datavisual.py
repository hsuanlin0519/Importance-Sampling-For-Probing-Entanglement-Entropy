import matplotlib.pyplot as plt
import numpy as np
import operator
import math
from matplotlib.ticker import PercentFormatter
import datetime
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_loss(train_loss, valid_loss):
    plt.plot(train_loss, color='red', linewidth=3, label="訓練損失")
    plt.plot(valid_loss, color='orange', linewidth=3, label="測試損失")
    plt.legend(loc='upper right', fontsize=18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("訓練代", fontsize=12)
    plt.ylabel('${\u03B4}$', fontsize=18)
    plt.show()
    plt.clf()


def plot_predictions(answers):
    plt.hist(answers, bins=np.arange(min(answers), max(answers) + 0.1, 0.1), edgecolor='black', color='green',
             linewidth='1.2', weights=np.ones(len(answers)) / len(answers))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('X')
    plt.ylabel('Proportion of Counts')
    plt.title("Distribution of ML Predictions")
    plt.show()
    plt.clf()
    # zoom in 10X
    plt.hist(answers, bins=np.arange(min(answers), max(answers) + 0.01, 0.01), edgecolor='black', color='pink',
             linewidth='1.2', weights=np.ones(len(answers)) / len(answers))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('X')
    plt.ylabel('Proportion of Counts')
    plt.title("Distribution of ML Predictions (Zoomed In)")
    plt.show()
    plt.clf()


def plot_label(label):
    plt.hist(label, bins=np.arange(min(label), max(label) + 0.1, 0.1), edgecolor='black', color='red', linewidth='1.2',
             weights=np.ones(len(label)) / len(label))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('X')
    plt.ylabel('Proportion of Counts')
    plt.title("Distribution of Dataset Label")
    plt.show()
    plt.clf()


def plot_metropolis(raw, metro):
    plt.hist(raw, bins=np.arange(min(raw), max(raw) + 0.1, 0.1), color='red', linewidth='1.2',
             weights=np.ones(len(raw)) / len(raw), label='raw')
    plt.hist(metro, bins=np.arange(min(metro), max(metro) + 0.1, 0.1), edgecolor='black', facecolor="None",
             linewidth='3', weights=np.ones(len(metro)) / len(metro), alpha=0.3, label='Metropolis')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('X')
    plt.ylabel('Proportion of Counts')
    plt.title("Distribution of Dataset Label")
    plt.legend(loc='upper right')
    plt.show()
    plt.clf()


def plot_circuit(circuit):
    circuit.draw(output='mpl', filename='C:/Users/josep/OneDrive/桌面/論文/piccir.png')
    plt.clf()


exp_f_root = 'C:/Users/josep/OneDrive/桌面/論文/論文圖/single_qubit_product/sim_logs/sher_0_1_124_excited_state.txt'


def init_log(cir_type, sys_size):
    exp_f = open(exp_f_root, "a")
    exp_f.write("========================================\n")
    exp_f.write(f'Experiment start time: {[datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).ctime()]}\n')
    exp_f.write(f'Circuit Type: {[cir_type]}\n')
    exp_f.write(f'System size: {[sys_size]}\n')
    exp_f.write("========================================\n")
    exp_f.close()


def end_log():
    exp_f = open(exp_f_root, "a")
    exp_f.write("========================================\n")
    exp_f.write(f'Experiment end time: {[datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).ctime()]}\n')
    exp_f.write("========================================\n\n\n\n")
    exp_f.close()


def write_log(info, value):
    exp_f = open(exp_f_root, "a")
    exp_f.write(f'{info}:{[value]}\n')
    exp_f.close()


def write_exp_log_uniform(backend, nm, nu, p2, ab_err):
    exp_f = open(exp_f_root, "a")
    write_log("backend is: ", backend)
    write_log("Nm is(uniform): ", nm)
    write_log("Nu is(uniform): ", nu)
    write_log("p2_is value is(uniform): ", p2)
    write_log("Absolute Error is(uniform): ", ab_err)
    exp_f.close()


def write_exp_log_metro(backend, nm, nu, p2, ab_err):
    exp_f = open(exp_f_root, "a")
    write_log("backend is: ", backend)
    write_log("Nm is(IS): ", nm)
    write_log("Nu is(IS): ", nu)
    write_log("p2_is value is(IS): ", p2)
    write_log("Absolute Error is(IS): ", ab_err)
    exp_f.close()