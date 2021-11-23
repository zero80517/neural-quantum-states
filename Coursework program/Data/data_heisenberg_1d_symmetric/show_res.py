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
    plt.ylabel('Энергия')

    if exact:
        plt.plot([iteration[0], iteration[-1]], [exact, exact], color='black', label='Точное значение')
        plt.legend()
    else:
        plt.legend()

    ax = fig.gca()
    ax.tick_params(axis='x', direction='in', length=10, top=True, bottom=True)
    ax.tick_params(axis='y', direction='in', length=10, left=True, right=True)


def relative_error(num, filename, color, label, exact=None, usetex=False):
    if exact is None:
        raise ValueError("enter the value of the exact solution")

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
    plt.plot(iteration, np.abs((en_mean-exact)/exact), color=color, label=label)
    plt.yscale('log')
    plt.xlabel('Итерация')
    plt.ylabel('Относительная погрешность')
    plt.legend()

    ax = fig.gca()
    ax.tick_params(axis='x', direction='in', length=10, top=True, bottom=True)
    ax.tick_params(axis='y', which='minor', direction='in', length=5, left=True, right=True)
    ax.tick_params(axis='y', direction='in', length=10, left=True, right=True)


# the exact value of the ground state
exact = -35.6175461195

# the name of data of the results of training and NQS
filename = "sgd"

# plot training result
energy_mean(num=1, filename=filename, color="g", label="SGD")
relative_error(num=2, filename=filename, color="g", label="SGD", exact=exact)

# the name of data of the results of training and NQS
filename = "momentum"
energy_mean(num=1, filename=filename, color="r", label="Momentum")
relative_error(num=2, filename=filename, color="r", label="Momentum", exact=exact)

# the name of data of the results of training and NQS
filename = "adam"
energy_mean(num=1, filename=filename, color="b", label="Adam", exact=exact)
relative_error(num=2, filename=filename, color="b", label="Adam", exact=exact)

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
#
# def relative_error(num, filename, marker, label, exact=None, usetex=False):
#     if exact is None:
#         raise ValueError("enter the value of the exact solution")
#
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
#     plt.plot(iteration, np.abs((en_mean-exact)/exact), marker=marker, color='black', label=label)
#     plt.yscale('log')
#     plt.xlabel('Итерация')
#     plt.ylabel('Относительная погрешность')
#     plt.legend()
#
#     ax = fig.gca()
#     ax.tick_params(axis='x', direction='in', length=10, top=True, bottom=True)
#     ax.tick_params(axis='y', which='minor', direction='in', length=5, left=True, right=True)
#     ax.tick_params(axis='y', direction='in', length=10, left=True, right=True)
#
#
# # the exact value of the ground state
# exact = -35.6175461195
#
# # the name of data of the results of training and NQS
# filename = "sgd"
#
# # plot training result
# energy_mean(num=1, filename=filename, marker='o', label="SGD")
# relative_error(num=2, filename=filename, marker="o", label="SGD", exact=exact)
#
# # the name of data of the results of training and NQS
# filename = "momentum"
# energy_mean(num=1, filename=filename, marker="v", label="Momentum")
# relative_error(num=2, filename=filename, marker="v", label="Momentum", exact=exact)
#
# # the name of data of the results of training and NQS
# filename = "adam"
# energy_mean(num=1, filename=filename, marker="s", label="Adam", exact=exact)
# relative_error(num=2, filename=filename, marker="s", label="Adam", exact=exact)

plt.show()
