import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Optimizes the parameters of a restricted Boltzmann machine (RBM) with respect to
    a Hamiltonian by Adam method.

    :param sampler: used sampler
    :param l_r: initial learning rate
    :param mm1_factor: momentum factor number 1
    :param mm2_factor: momentum factor number 2
    :param SR: if True, optimizer uses stochastic-reconfiguration method.
    :param L2: if True, optimizer uses L2 regularization
    """

    def __init__(self, nqs, learning_rate=1e-2, first_momentum_factor=0.9,
                 second_momentum_factor=0.999, SR=True, L2=True):
        super().__init__()

        self.learning_rate = learning_rate

        # determines changing of the learning step
        self.schedule_multiplier = "fixed"

        # parameter that decays the learning step each iteration
        self.decay_factor = 3 / 4

        # if True, optimizer uses stochastic-reconfiguration method.
        self.SR = SR

        # regularization parameters for SR method
        self.lambd0 = 100.
        self.b = 0.9
        self.lambdMin = 1e-4

        # if True, optimizer uses L2 regularization method
        self.L2 = L2

        self.L2_factor = 1e-4

        self.first_momentum_factor = first_momentum_factor
        self.second_momentum_factor = second_momentum_factor

        # to pass divide on zero
        self.eps = 1e-8

        # first moment vector
        self.m = np.zeros(
            shape=[nqs.get_nspins() + nqs.get_nhidden() +
                   nqs.get_nspins() * nqs.get_nhidden()]
        )

        # second moment vector
        self.v = np.zeros(
            shape=[nqs.get_nspins() + nqs.get_nhidden() +
                   nqs.get_nspins() * nqs.get_nhidden()]
        )

        # the name of the optimization method
        self.name = "Adam"

        # results of optimizing
        self.sum_abs_w = {"sum[|w|]": 0.}
        self.mean_abs_w = {"mean[|w|]": 0.}
        self.mean_abs_dw = {"mean[|dw|]": 0.}

    def compute_update(self, nqs, sampler, iteration):
        """
        Computes update to weights.
        """
        self.sum_abs_w["sum[|w|]"] = float(
            np.sum(np.abs(nqs.get_a() ) ) +
            np.sum(np.abs(nqs.get_b() ) ) +
            np.sum(np.abs(nqs.get_W() ) )
        )
        self.mean_abs_w["mean[|w|]"] = self.sum_abs_w["sum[|w|]"] / \
            (len(nqs.get_a() ) + len(nqs.get_b() ) + len(np.abs(nqs.get_W() ) ) )

        if self.SR:
            grad = self.compute_sr_grad(nqs, sampler, iteration)
        else:
            grad = self.compute_grad(nqs, sampler)

        dm = (1-self.first_momentum_factor) * grad
        self.m = self.first_momentum_factor * self.m + dm

        dv = (1-self.second_momentum_factor) * np.square(grad)
        self.v = self.second_momentum_factor + dv

        M = self.m / (1 - self.first_momentum_factor**(iteration))
        V = self.v / (1 - self.second_momentum_factor**(iteration))

        dw = self.get_schedule_multiplier(iteration) * (self.learning_rate * M / (np.sqrt(V) + self.eps))
        self.mean_abs_dw["mean[|dw|]"] = float(np.mean(np.abs(dw)))

        update = -dw

        return update

    def get_params(self):
        params = dict()
        params[self.name] = dict()
        params[self.name]['learning_rate'] = self.learning_rate

        if self.schedule_multiplier == "fixed":
            params[self.name]["schedule_multiplier"] = self.schedule_multiplier
        elif self.schedule_multiplier == "decayed":
            params[self.name]["schedule_multiplier"] = dict()
            params[self.name]["schedule_multiplier"]["type"] = self.schedule_multiplier
            params[self.name]["schedule_multiplier"]["decay_factor"] = self.decay_factor

        params[self.name]['first_momentum_factor'] = self.first_momentum_factor
        params[self.name]['second_momentum_factor'] = self.second_momentum_factor

        if self.SR:
            params[self.name]['SR'] = dict()
            params[self.name]['SR']["lambd0"] = self.lambd0
            params[self.name]['SR']["b"] = self.b
            params[self.name]['SR']["lambdMin"] = self.lambdMin
        else:
            params[self.name]['SR'] = False

        if self.L2:
            params[self.name]["L2"] = dict()
            params[self.name]["L2"]['L2_factor'] = self.L2_factor
        else:
            params[self.name]["L2"] = False

        return params

    def get_results(self):
        results = dict()
        results["optimizer results"] = dict()
        results["optimizer results"].update(self.sum_abs_w)
        results["optimizer results"].update(self.mean_abs_w)
        results["optimizer results"].update(self.mean_abs_dw)

        return results

