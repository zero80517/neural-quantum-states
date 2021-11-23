import time

from .. import logging


class Vmc(object):
    """
    Implementation of the variational Monte-Carlo method.

    :param nqs: used neural-network
    :param operator: used operator
    :param sampler: used sampler
    :param optimizer: used optimizer
    :param filename: filename of the NQS and training
    """

    def __init__(self, nqs, operator, sampler, optimizer, filename=None):
        self.nqs = nqs
        self.operator = operator
        self.sampler = sampler
        self.optimizer = optimizer

        if filename is None:
            self.filename = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
        else:
            self.filename = filename

        # the results of training
        self.sampler_results = []
        self.optimizer_results = []

    """
    Run VMC method.
    """
    def run(self, num_iter):
        print(f"Considering the operator: {self.operator.get_params()}")
        print(f"Using the sampler: {self.sampler.get_params()}")
        print(f"Using the optimizer: {self.optimizer.get_params()}", end="\n\n\n")
        print('Start training...')

        for iteration in range(1, num_iter+1):
            show_disp = {"Iteration": iteration}
            tic = time.time()

            self.sampler.run()
            update = self.optimizer.compute_update(self.nqs, self.sampler, iteration)
            self.nqs.update_weights(update)
            self.sampler_results.append(self.sampler.get_results() )
            self.optimizer_results.append(self.optimizer.get_results() )
            logging.save_data(self)

            toc = time.time()
            rest_time = int((toc-tic)*num_iter - (toc-tic)*iteration)
            show_disp.update(self.sampler_results[-1])
            show_disp.update({"Time left": f"{rest_time // 60} min {rest_time % 60} s"})
            print(show_disp)
