#include "nqs.h"
#include "ising_1d_symmetric.h"

Ising1dSymmetric::Ising1dSymmetric(Nqs & nqs, double hfield, double Jz, bool pbc):
	   m_nspins(nqs.getNspins()), m_hfield(hfield), m_Jz(Jz), m_pbc(pbc) 
{
    Init();
}

void Ising1dSymmetric::Init() {
    m_mel.resize(m_nspins+1, 0.);
    m_flipsh.resize(m_nspins+1);

    for (int i=0; i < m_nspins; ++i) {
        m_mel[i+1] = -m_hfield;
        m_flipsh[i+1] = list<int>(1, i);
    }
}

//Finds the non-zero matrix elements of the hamiltonian
//on the given state
//i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
//state' is encoded as the sequence of spin flips to be performed on state
void Ising1dSymmetric::FindConn(const list<int> & state, list<list<int> > & flipsh, list<complex128> & mel) {
    mel.resize(m_nspins+1);
    flipsh.resize(m_nspins+1);

    //assigning pre-computed matrix elements and spin flips
    mel = m_mel;
    flipsh = m_flipsh;

    //computing interaction part Sz*Sz
    mel[0] = 0.;

    for (int i=0; i < (m_nspins-1); ++i) {
        mel[0] -= double(state[i]*state[i+1]);
    }

    if (m_pbc) {
        mel[0] -= double(state[m_nspins-1]*state[0]);
    }

    mel[0] *= m_Jz;
}

dict Ising1dSymmetric::getParams() {
	dict params;
	params[m_name]["hfield"] = m_hfield;
	params[m_name]["Jz"] = m_Jz;
	params[m_name]["pbc"] = m_pbc;

	return params;
}