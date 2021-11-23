#include "nqs.h"
#include "heisenberg_1d_symmetric.h"

Heisenberg1dSymmetric::Heisenberg1dSymmetric(Nqs & nqs, double Jz, bool pbc):
	   m_nspins(nqs.getNspins()), m_Jz(Jz), m_pbc(pbc) 
{
    Init();
}

void Heisenberg1dSymmetric::Init() {
    m_mel.resize(1);
    m_flipsh.resize(1);
}

//Finds the non-zero matrix elements of the hamiltonian
//on the given state
//i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
//state' is encoded as the sequence of spin flips to be performed on state
void Heisenberg1dSymmetric::FindConn(const list<int> & state, list<list<int> > & flipsh, list<complex128> & mel) {
    //assigning pre-computed matrix elements and spin flips
    mel = m_mel;
    flipsh = m_flipsh;

    //computing interaction part Sz*Sz
    mel[0] = 0.;

    for (int i=0; i < (m_nspins-1); ++i) {
        mel[0] += double(state[i]*state[i+1]);
    }

    if (m_pbc) {
        mel[0] += double(state[m_nspins-1]*state[0]);
    }

    mel[0] *= m_Jz;

    //Looks for possible spin flips
    for (int i=0; i < (m_nspins-1); ++i) {
        if(state[i]!=state[i+1]){
            mel.push_back(-2.);
            flipsh.push_back(list<int>({i, i+1}) );
        }
    }

    if (m_pbc) {
        if (state[m_nspins-1] != state[0]) {
            mel.push_back(-2.);
            flipsh.push_back(list<int>({m_nspins-1, 0}) );
        }
    }
}

dict Heisenberg1dSymmetric::getParams() {
	dict params;
	params[m_name]["Jz"] = m_Jz;
	params[m_name]["pbc"] = m_pbc;

	return params;
}