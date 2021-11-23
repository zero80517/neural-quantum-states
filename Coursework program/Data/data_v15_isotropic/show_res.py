import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib

matplotlib.rcParams.update({'font.size': 20})


def energy_mean(num, filename, color, label, exact=None, usetex=False):
    with open(filename + '.log', 'r') as file:
        training = json.load(file)['training']

    iteration = np.zeros(len(training))
    en_mean = np.zeros_like(iteration)

    for i in range(len(iteration) ):
        iteration[i] = training[i]['iteration']
        en_mean[i] = training[i]['sampler results']['energy mean']

    if usetex:
        rc('text', usetex=True)

    fig = plt.figure(num=num)
    plt.plot(iteration, en_mean, color=color, label=label)
    plt.xlabel('Итерация')
    plt.ylabel('Энергия (K)')

    if exact:
        plt.plot([iteration[0], iteration[-1]], [exact, exact], color='black', label='Точное значение')
        plt.legend()
    else:
        plt.legend()

    ax = fig.gca()
    ax.tick_params(axis='x', direction='in', length=10, top=True, bottom=True)
    ax.tick_params(axis='y', direction='in', length=10, left=True, right=True)


# the name of data of the results of training and NQS
filename = "sgd"

# plot training result
energy_mean(num=1, filename=filename, color="g", label="SGD")

# the name of data of the results of training and NQS
filename = "momentum"
energy_mean(num=1, filename=filename, color="r", label="Momentum")

# the name of data of the results of training and NQS
filename = "adam"
energy_mean(num=1, filename=filename, color="b", label="Adam")

# def energy_mean(num, filename, marker, label, exact=None, usetex=False):
#     with open(filename + '.log', 'r') as file:
#         training = json.load(file)['training']
#
#     iteration = np.zeros(len(training))
#     en_mean = np.zeros_like(iteration)
#
#     for i in range(len(iteration) ):
#         iteration[i] = training[i]['iteration']
#         en_mean[i] = training[i]['sampler results']['energy mean']
#
#     if usetex:
#         rc('text', usetex=True)
#
#     fig = plt.figure(num=num)
#     plt.plot(iteration, en_mean, marker=marker, color='black', label=label)
#     plt.xlabel('Итерация')
#     plt.ylabel('Энергия')
#
#     if exact:
#         plt.plot([iteration[0], iteration[-1]], [exact, exact], color='black', label='Точное значение')
#         plt.legend()
#     else:
#         plt.legend()
#
#     ax = fig.gca()
#     ax.tick_params(axis='x', direction='in', length=10, top=True, bottom=True)
#     ax.tick_params(axis='y', direction='in', length=10, left=True, right=True)
#
# # the name of data of the results of training and NQS
# filename = "sgd"
#
# # plot training result
# energy_mean(num=1, filename=filename, marker='o', label="SGD")
#
# # the name of data of the results of training and NQS
# filename = "momentum"
# energy_mean(num=1, filename=filename, marker="v", label="Momentum")
#
# # the name of data of the results of training and NQS
# filename = "adam"
# energy_mean(num=1, filename=filename, marker="s", label="Adam")

plt.show()
