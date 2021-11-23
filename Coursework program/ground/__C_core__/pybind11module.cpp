#include <cmath>
#include <pybind11/pybind11.h>
#include "src/random.h"
#include "src/nqs.h"
#include "src/ising_1d_symmetric.h"
#include "src/v15_isotropic.h"
#include "src/heisenberg_1d_symmetric.h"
#include "src/metropolis_hastings.h"

namespace py = pybind11;

extern Random _inst;

PYBIND11_MODULE(pybind11module, m) {
 	py::class_<Random>(m, "Random")
		.def(py::init())
		.def("seed", &Random::seed, py::arg("seed"))
		.def("normal", &Random::normal, py::arg("loc") = 0., py::arg("scale") = 1.)
		.def("randrange", &Random::randrange, py::arg("start"), 
			 py::arg("stop") = 0, py::arg("step") = 1)
		.def("randint", &Random::randint, py::arg("a"), py::arg("b"))
        .def("random", &Random::random);

    py::class_<Nqs>(m, "Nqs")
		.def(py::init<int , int , double ,
		 	 list<complex128> ,
		 	 list<complex128> ,
		 	 list<list<complex128> > >(), 
		 	 py::arg("n_visible") = 20, 
		 	 py::arg("alpha") = 2, 
		 	 py::arg("sigma") = 0.001,
		     py::arg("a")=list<complex128>{std::nan("0")},
		     py::arg("b")=list<complex128>{std::nan("0")},
		     py::arg("W")=list<list<complex128> >{{std::nan("0")}})
		.def("get_nspins", &Nqs::getNspins)
		.def("get_nhidden", &Nqs::getNhidden)
		.def("get_a", &Nqs::getA)
		.def("get_b", &Nqs::getB)
		.def("get_W", &Nqs::getW)
		.def("get_params", &Nqs::getParams)
		.def("update_weights", &Nqs::UpdateWeights);

	py::class_<Ising1dSymmetric>(m, "Ising1dSymmetric")
		.def(py::init<Nqs & , double , double , bool>(),
		 	 py::arg("nqs"), 
		 	 py::arg("hfield") = 1.0, 
		 	 py::arg("Jz") = 1.0,
		     py::arg("pbc") = true)
		.def("get_params", &Ising1dSymmetric::getParams);

	py::class_<V15Isotropic>(m, "V15Isotropic")
		.def(py::init<Nqs & , double , double , double>(),
		 	 py::arg("nqs"), 
		 	 py::arg("J") = 800., 
		 	 py::arg("J1") = 225.,
		     py::arg("J2") = 350.)
		.def("get_params", &V15Isotropic::getParams);
		
	py::class_<Heisenberg1dSymmetric>(m, "Heisenberg1dSymmetric")
		.def(py::init<Nqs & , double , bool >(),
		 	 py::arg("nqs"),
		 	 py::arg("Jz") = 1.0,
		     py::arg("pbc") = true)
		.def("get_params", &Heisenberg1dSymmetric::getParams);

	py::class_<MetropolisHastings>(m, "MetropolisHastings")
		.def(py::init<Nqs & , Ising1dSymmetric & , list<int> , bool ,
			 int , double , double , int , int>(),
		 	 py::arg("nqs"), 
		 	 py::arg("operator"),
		 	 py::arg("initial_state") = list<int>{0}, 
		 	 py::arg("zero_magnetization") = true, 
		 	 py::arg("n_sweeps") = 10000,
		     py::arg("therm_factor") = 0.1,
		     py::arg("sweep_factor") = 1.,
		     py::arg("n_flips") = 1,
		     py::arg("n_blocks") = 50)
		.def(py::init<Nqs & , V15Isotropic & , list<int> , bool ,
			 int , double , double , int , int>(),
		 	 py::arg("nqs"), 
		 	 py::arg("operator"),
		 	 py::arg("initial_state") = list<int>{0},
		 	 py::arg("zero_magnetization") = false, 
		 	 py::arg("n_sweeps") = 10000,
		     py::arg("therm_factor") = 0.1,
		     py::arg("sweep_factor") = 1.,
		     py::arg("n_flips") = 1,
		     py::arg("n_blocks") = 50)
		.def(py::init<Nqs & , Heisenberg1dSymmetric & , list<int> , bool ,
			 int , double , double , int , int>(),
		 	 py::arg("nqs"), 
		 	 py::arg("operator"),
		 	 py::arg("initial_state") = list<int>{0}, 
		 	 py::arg("zero_magnetization") = true, 
		 	 py::arg("n_sweeps") = 10000,
		     py::arg("therm_factor") = 0.1,
		     py::arg("sweep_factor") = 1.,
		     py::arg("n_flips") = 2,
		     py::arg("n_blocks") = 50)
		.def("run", &MetropolisHastings::Run)
		.def("get_nqs", &MetropolisHastings::getNqs)
		.def("get_nsweeps", &MetropolisHastings::getNsweeps)
		.def("get_samples", &MetropolisHastings::getSamples)
		.def("get_local_energies", &MetropolisHastings::getEnLocal)
		.def("get_params", &MetropolisHastings::getParams)
		.def("get_results", &MetropolisHastings::getResults);

    m.attr("_inst") = py::cast(_inst, py::return_value_policy::reference);
}