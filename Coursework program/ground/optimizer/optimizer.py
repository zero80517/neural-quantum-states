import tensorflow as tf
import numpy as np


class Optimizer(object):
    """
    Will eventually allow for more general implementations of optimizers
    """

    def __init__(self):
        self.learning_rate = 1e-2

        # determines changing of the learning step
        self.schedule_multiplier = "fixed"

        if self.schedule_multiplier != "fixed" and \
                self.schedule_multiplier != "decayed":
            raise ValueError(f'Please choose a schedule multiplier from: fixed or decayed')

        # parameter that decays the learning step each iteration
        self.decay_factor = 3/4

        # if True, optimizer uses stochastic-reconfiguration method.
        self.SR = True

        # regularization parameters for SR method
        self.lambd0 = 100.
        self.b = 0.9
        self.lambdMin = 1e-4

        # if True, optimizer uses L2 regularization method
        self.L2 = True

        self.L2_factor = 1e-4

    def compute_update(self, nqs, sampler, iteration):
        pass

    def compute_grad(self, nqs, sampler):
        """
        Computes stochastic gradients based on the parameters derivatives,
        using L2 regularization method, optional.
        """
        states = tf.constant(sampler.get_samples(), dtype=tf.complex128)
        E_locs = tf.reshape(
            tf.constant(sampler.get_local_energies(), dtype=tf.complex128),
            shape=[len(sampler.get_local_energies() ), 1]
        )
        a = np.array(nqs.get_a() )
        b = tf.constant(nqs.get_b(), dtype=tf.complex128)
        W = tf.constant(nqs.get_W(), dtype=tf.complex128)
        n_spins = nqs.get_nspins()
        n_h = nqs.get_nhidden()
        n_samples = sampler.get_nsweeps()

        theta = tf.tensordot(tf.transpose(W), tf.transpose(states), axes=1) + tf.reshape(b, shape=[n_h, 1])  # n_h x n_samples
        D_a = tf.transpose(states)  # n_spins x n_samples
        D_b = tf.math.tanh(theta)  # n_h x n_samples
        D_W = tf.reshape(tf.transpose(states), shape=[n_spins, 1, n_samples]) * \
              tf.math.tanh(tf.reshape(theta, shape=[1, n_h, n_samples]))
        derivs = tf.concat(values=[D_a, D_b, tf.reshape(D_W, [n_spins * n_h, n_samples])], axis=0)

        F_p = tf.math.reduce_sum(tf.transpose(E_locs) * tf.math.conj(derivs), axis=1) / n_samples
        F_p -= tf.math.reduce_sum(tf.transpose(E_locs), axis=1) * \
               tf.math.reduce_sum(tf.math.conj(derivs), axis=1) / (n_samples ** 2)

        if self.L2:
            weights = tf.constant(np.concatenate(
                (a.reshape((n_spins,)), np.array(b).reshape((n_h,)),
                 np.array(W).reshape((n_spins * n_h,))),
                axis=0
            ), dtype=tf.complex128)
            grad = np.array(
                tf.cast(2 * tf.math.real(F_p), dtype=tf.complex128) + self.L2_factor * weights
            )
        else:
            grad = np.array(tf.cast(2 * tf.math.real(F_p), dtype=tf.complex128) )

        return grad

    def compute_sr_grad(self, nqs, sampler, iteration):
        """
        Computes gradients based on the parameters derivatives and stochastic
        reconfiguration method using L2 regularization method, optional.
        """
        states = tf.constant(sampler.get_samples(), dtype=tf.complex128)
        E_locs = tf.reshape(
            tf.constant(sampler.get_local_energies(), dtype=tf.complex128),
            shape=[len(sampler.get_local_energies()), 1]
        )
        a = np.array(nqs.get_a())
        b = tf.constant(nqs.get_b(), dtype=tf.complex128)
        W = tf.constant(nqs.get_W(), dtype=tf.complex128)
        n_spins = nqs.get_nspins()
        n_h = nqs.get_nhidden()
        n_samples = sampler.get_nsweeps()

        theta = tf.tensordot(tf.transpose(W), tf.transpose(states), axes=1) + tf.reshape(b, shape=[n_h, 1])  # n_h x n_samples
        D_a = tf.transpose(states)  # n_spins x n_samples
        D_b = tf.math.tanh(theta)  # n_h x n_samples
        D_W = tf.reshape(tf.transpose(states), shape=[n_spins, 1, n_samples]) * \
              tf.math.tanh(tf.reshape(theta, shape=[1, n_h, n_samples]))
        derivs = tf.concat(values=[D_a, D_b, tf.reshape(D_W, [n_spins * n_h, n_samples])], axis=0)

        avg_derivs = tf.math.reduce_sum(derivs, axis=1, keepdims=True) / n_samples
        avg_derivs_mat = tf.math.conj(tf.reshape(avg_derivs, shape=[derivs.shape[0], 1]))
        avg_derivs_mat = avg_derivs_mat * tf.reshape(avg_derivs, shape=[1, derivs.shape[0]])

        moment2 = tf.einsum('ik,jk->ij', tf.math.conj(derivs), derivs) / n_samples
        S_kk = tf.math.subtract(moment2, avg_derivs_mat)
        S_kk2 = tf.linalg.tensor_diag(self.lambd(iteration) * tf.linalg.tensor_diag_part(S_kk))
        S_reg = S_kk + S_kk2

        F_p = tf.math.reduce_sum(tf.transpose(E_locs) * tf.math.conj(derivs), axis=1) / n_samples
        F_p -= tf.math.reduce_sum(tf.transpose(E_locs), axis=1) * \
               tf.math.reduce_sum(tf.math.conj(derivs), axis=1) / (n_samples ** 2)

        if self.L2:
            weights = tf.constant(np.concatenate(
                (a.reshape((n_spins,)), np.array(b).reshape((n_h,)),
                 np.array(W).reshape((n_spins * n_h,))),
                axis=0
            ), dtype=tf.complex128)
            try:
                grad = np.dot(
                    np.linalg.inv(S_reg), tf.cast(2 * tf.math.real(F_p), dtype=tf.complex128) +
                                          self.L2_factor * weights
                )
            except np.linalg.LinAlgError:
                raise ValueError('matrix S_reg is not invertable. Please choose another regularization parameters or method')
        else:
            try:
                grad = np.dot(np.linalg.inv(S_reg), tf.cast(2 * tf.math.real(F_p), dtype=tf.complex128) )
            except np.linalg.LinAlgError:
                raise ValueError('matrix S_reg is not invertable. Please choose another regularization parameters or method')

        return grad

    def lambd(self, iteration):
        """
        Lambda regularization parameter for S_kk matrix,
        see supplementary materials.
        """
        return max(self.lambd0 * (self.b ** iteration), self.lambdMin)

    def set_sr_params(self, SR=None, lambd0=None, b=None, lambdMin=None):
        if SR is not None:
            self.SR = SR
        if lambd0 is not None:
            self.lambd0 = lambd0
        if b is not None:
            self.b = b
        if lambdMin is not None:
            self.lambdMin = lambdMin

    def set_L2_params(self, L2=None, L2_factor=None):
        if L2 is not None:
            self.L2 = L2
        if L2_factor is not None:
            self.L2_factor = L2_factor

    def set_schedule_multiplier(self, schedule_multiplier=None,
                                decay_factor=None):
        if schedule_multiplier is not None:
            self.schedule_multiplier = schedule_multiplier

            if self.schedule_multiplier != "fixed" and \
                    self.schedule_multiplier != "decayed":
                raise ValueError(f'Please choose a schedule multiplier from: fixed or decayed')

        if decay_factor is not None:
            self.decay_factor = decay_factor

    def get_schedule_multiplier(self, iteration):
        if self.schedule_multiplier == 'fixed':
            return 1.
        elif self.schedule_multiplier == 'decayed':
            return np.power(self.decay_factor, iteration)

    def get_params(self):
        pass

    def get_results(self):
        pass
