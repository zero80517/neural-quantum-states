#ifndef V15_ISOTROPIC_H
#define V15_ISOTROPIC_H

#include <pybind11/stl.h>
#include "nqs.h"

//isotropic part of the Hamiltonian
//of the magnetic claster V15
class V15Isotropic {
private:
    //number of spins
    const int m_nspins;

   	//parameters from the paper: "Konstantinidis. Magnetic Anisotropy in the Molecular Complex V15"
    const double m_J;
    const double m_J1;
    const double m_J2;

    //name of the hamiltonian
    str m_name = "V15Isotropic";

    //indexes of J, J1 and J2 parts of the Hamiltonian 
    list<list<int> > m_Jpart  = list<list<int> >{{0, 1}, {2, 3}, {4, 5}, {9, 10}, {11, 12}, {13, 14}};
    list<list<int> > m_J1part = list<list<int> >{{1, 2}, {3, 4}, {0, 5}, {10, 11}, {12, 13}, {9, 14},
                                                 {1, 6}, {6, 10}, {3, 7}, {7, 14}, {5, 8}, {8, 12}};
 	list<list<int> > m_J2part = list<list<int> >{{0, 2}, {2, 4}, {0, 4}, {9, 11}, {11, 13}, {9, 13},
                                                 {0, 6}, {6, 9}, {2, 7}, {7, 13}, {4, 8}, {8, 11}};

    //pre-computed quantities
    list<complex128> m_mel;
    list<list<int> > m_flipsh;

public:
    V15Isotropic(Nqs & nqs, double J=800., double J1=225., double J2=350.);
    void Init();
    void FindConn(const list<int> & state, list<list<int> > & flipsh, list<complex128> & mel);
    dict getParams();
};

#endif