#ifndef HEISENBERG_1D_SYMMETRIC_H
#define HEISENBERG_1D_SYMMETRIC_H

#include <pybind11/stl.h>
#include "nqs.h"

//Anti-ferromagnetic Heisenberg model in 1d with lattice symmetry
class Heisenberg1dSymmetric {
private:
    //number of spins
    const int m_nspins;

    //value of the exchange
    const double m_Jz;

    //option to use periodic boundary conditions
    const bool m_pbc;

    //Name of the hamiltonian
    str m_name = "Heisenberg1dSymmetric";

    //pre-computed quantities
    list<complex128> m_mel;
    list<list<int> > m_flipsh;

public:
    Heisenberg1dSymmetric(Nqs & nqs, double Jz=1.0, bool pbc=true);
    void Init();
    void FindConn(const list<int> & state, list<list<int> > & flipsh, list<complex128> & mel);
    dict getParams();
};

#endif