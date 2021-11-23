import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def energy_mean(num, filename, title, exact=None, usetex=False):
    with open(filename + '.log', 'r') as file:
        training = json.load(file)['training']

    iteration = np.zeros(len(training))
    en_mean = np.zeros_like(iteration)
    en_error = np.zeros_like(iteration)

    for i in range(len(iteration) ):
        iteration[i] = training[i]['iteration']
        en_mean[i] = training[i]['sampler results']['energy mean']
        en_error[i] = training[i]['sampler results']['energy error']

    if usetex:
        rc('text', usetex=True)

    plt.figure(num=num)
    plt.errorbar(iteration, en_mean, yerr=en_error, fmt='-', color="blue",
                 ecolor='red', label='Results of training')
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    plt.title(title)

    if exact:
        plt.plot([iteration[0], iteration[-1]], [exact, exact], color='black', label='Exact value')
        plt.legend()
    else:
        plt.legend()


def relative_error(num, filename, title, exact=None, usetex=False):
    if exact is None:
        raise ValueError("enter the value of the exact solution")

    with open(filename + '.log', 'r') as file:
        training = json.load(file)['training']

    iteration = np.zeros(len(training))
    en_mean = np.zeros_like(iteration)
    en_error = np.zeros_like(iteration)

    for i in range(len(iteration) ):
        iteration[i] = training[i]['iteration']
        en_mean[i] = training[i]['sampler results']['energy mean']
        en_error[i] = training[i]['sampler results']['energy error']

    if usetex:
        rc('text', usetex=True)

    plt.figure(num=num)
    yerr = np.array(
        [np.abs(en_error/exact), np.abs(en_error/exact)]
    ).reshape((2, int(iteration[-1]) ) )
    plt.errorbar(iteration, np.abs((en_mean-exact)/exact),
                 yerr=yerr, fmt='-', color="blue", ecolor='red',
                 label='Results of training')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Relative error')
    plt.title(title)
    plt.legend()


def sgd(num, filename, title, usetex=False):
    with open(filename + '.log', 'r') as file:
        data = json.load(file)
        if "SGD" not in data:
            raise ValueError(f"the {filename} doesn't have results of SGD optimizer")
        training = data['training']

    iteration = np.zeros(len(training))
    sum_abs_w = np.zeros_like(iteration)
    mean_abs_w = np.zeros_like(iteration)
    mean_abs_dw = np.zeros_like(iteration)

    for i in range(len(iteration) ):
        iteration[i] = training[i]['iteration']
        sum_abs_w[i] = training[i]["optimizer results"]['sum[|w|]']
        mean_abs_w[i] = training[i]["optimizer results"]["mean[|w|]"]
        mean_abs_dw[i] = training[i]["optimizer results"]["mean[|dw|]"]

    if usetex:
        rc('text', usetex=True)

    fig = plt.figure(num=num)
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(iteration, sum_abs_w, color="black")
    ax1.set_ylabel('Sum of |weights|')
    ax1.set_xlabel('Iterations')
    ax1.set_title(title)

    ax2 = fig.add_axes([0.3, 0.2, 0.5, 0.45])
    yerr = np.array(
        [np.heaviside(mean_abs_w - mean_abs_dw, 0) * mean_abs_dw,
         mean_abs_dw]
    ).reshape((2, int(iteration[-1]) ) )
    ax2.errorbar(iteration, mean_abs_w, yerr=yerr, color="blue", ecolor="red")
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Mean of |weights| ± |diff of weights| and positive')


def momentum(num, filename='test', title='Results of Momentum optimizer',
                          usetex=False):
    with open(filename + '.log', 'r') as file:
        data = json.load(file)
        if "Momentum" not in data:
            raise ValueError(f"The {filename} doesn't have results of Momentum optimizer")
        training = data['training']

    iteration = np.zeros(len(training))
    sum_abs_w = np.zeros_like(iteration)
    mean_abs_w = np.zeros_like(iteration)
    mean_abs_dw = np.zeros_like(iteration)

    for i in range(len(iteration) ):
        iteration[i] = training[i]['iteration']
        sum_abs_w[i] = training[i]["optimizer results"]['sum[|w|]']
        mean_abs_w[i] = training[i]["optimizer results"]["mean[|w|]"]
        mean_abs_dw[i] = training[i]["optimizer results"]["mean[|dw|]"]

    if usetex:
        rc('text', usetex=True)

    fig = plt.figure(num=num)
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(iteration, sum_abs_w, color="black")
    ax1.set_ylabel('Sum of |weights|')
    ax1.set_xlabel('Iterations')
    ax1.set_title(title)

    ax2 = fig.add_axes([0.3, 0.2, 0.5, 0.45])
    yerr = np.array(
        [np.heaviside(mean_abs_w - mean_abs_dw, 0) * mean_abs_dw,
         mean_abs_dw]
    ).reshape((2, int(iteration[-1]) ) )
    ax2.errorbar(iteration, mean_abs_w, yerr=yerr, color="blue", ecolor="red")
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Mean of |weights| ± |diff of weights| and positive')


def adam(num, filename, title, usetex=False):
    with open(filename + '.log', 'r') as file:
        data = json.load(file)
        if "Adam" not in data:
            raise ValueError(f"The {filename} doesn't have results of Adam optimizer")
        training = data['training']

    iteration = np.zeros(len(training))
    sum_abs_w = np.zeros_like(iteration)
    mean_abs_w = np.zeros_like(iteration)
    mean_abs_dw = np.zeros_like(iteration)

    for i in range(len(iteration) ):
        iteration[i] = training[i]['iteration']
        sum_abs_w[i] = training[i]["optimizer results"]['sum[|w|]']
        mean_abs_w[i] = training[i]["optimizer results"]["mean[|w|]"]
        mean_abs_dw[i] = training[i]["optimizer results"]["mean[|dw|]"]

    if usetex:
        rc('text', usetex=True)

    fig = plt.figure(num=num)
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(iteration, sum_abs_w, color="black")
    ax1.set_ylabel('Sum of |weights|')
    ax1.set_xlabel('Iterations')
    ax1.set_title(title)

    ax2 = fig.add_axes([0.3, 0.2, 0.5, 0.45])
    yerr = np.array(
        [np.heaviside(mean_abs_w - mean_abs_dw, 0) * mean_abs_dw,
         mean_abs_dw]
    ).reshape((2, int(iteration[-1]) ) )
    ax2.errorbar(iteration, mean_abs_w, yerr=yerr, color="blue", ecolor="red")
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Mean of |weights| ± |diff of weights| and positive')
