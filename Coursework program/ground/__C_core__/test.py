import pybind11module
import time

pybind11module._inst.seed(3141592654)

nqs = pybind11module.Nqs(n_visible=20)
ising1dsymmetric = pybind11module.Ising1dSymmetric(nqs=nqs)
metropolis_hastings = pybind11module.MetropolisHastings(nqs=nqs, operator=ising1dsymmetric)
tic = time.time()
metropolis_hastings.run()
print("time =", time.time() - tic)
print(metropolis_hastings.get_results())

pybind11module._inst.seed(3141592654)

nqs = pybind11module.Nqs(n_visible=15)
v15_isotropic = pybind11module.V15Isotropic(nqs=nqs)
metropolis_hastings = pybind11module.MetropolisHastings(nqs=nqs, operator=v15_isotropic)
tic = time.time()
metropolis_hastings.run()
print("time =", time.time() - tic)
print(metropolis_hastings.get_results())

pybind11module._inst.seed(3141592654)

nqs = pybind11module.Nqs(n_visible=20)
heisenberg_1d_symmetric = pybind11module.Heisenberg1dSymmetric(nqs=nqs)
metropolis_hastings = pybind11module.MetropolisHastings(nqs=nqs, operator=heisenberg_1d_symmetric)
tic = time.time()
metropolis_hastings.run()
print("time =", time.time() - tic)
print(metropolis_hastings.get_results())
