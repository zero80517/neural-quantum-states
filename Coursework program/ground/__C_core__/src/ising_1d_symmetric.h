#ifndef ISING_1D_SYMMETRIC_H
#define ISING_1D_SYMMETRIC_H

#include <pybind11/stl.h>
#include "nqs.h"

//Transverse-field Ising model in 1d with lattice symmetry
class Ising1dSymmetric {
private:
    //number of spins
    const int m_nspins;

    //value of the transverse field
    const double m_hfield;

    //value of the exchange
    const double m_Jz;

    //option to use periodic boundary conditions
    const bool m_pbc;

    //Name of the hamiltonian
    str m_name = "Ising1dSymmetric";

    //pre-computed quantities
    list<complex128> m_mel;
    list<list<int> > m_flipsh;

public:
    Ising1dSymmetric(Nqs & nqs, double hfield=1.0, double Jz=1.0, bool pbc=true);
    void Init();
    void FindConn(const list<int> & state, list<list<int> > & flipsh, list<complex128> & mel);
    dict getParams();
};

#endif