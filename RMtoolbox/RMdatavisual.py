import matplotlib.pyplot as plt
import numpy as np
import operator
import math
from matplotlib.ticker import PercentFormatter
import datetime
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_circuit(circuit, f_name):
    circuit.draw(output='mpl', filename=f_name)
    plt.clf()
