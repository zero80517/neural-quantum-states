#include <cstdlib>
#include <pybind11/pybind11.h>
#include "v15_isotropic.h"

V15Isotropic::V15Isotropic(Nqs & nqs, double J, double J1, double J2):
	m_nspins(nqs.getNspins()), m_J(J), m_J1(J1), m_J2(J2) {

	Init();
}
void V15Isotropic::Init(){
	if (m_nspins != 15) {
		throw pybind11::value_error("must be 15 spins");
		std::exit(EXIT_FAILURE);
	}
	
	m_mel.resize(1);
    m_flipsh.resize(1);
}

//Finds the non-zero matrix elements of the hamiltonian
//on the given state
//i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
//state' is encoded as the sequence of spin flips to be performed on state
void V15Isotropic::FindConn(const list<int> & state, list<list<int> > & flipsh, list<complex128> & mel) {
	mel = m_mel;
    flipsh = m_flipsh;
	
	mel[0] = 0.;
	
	//computing interaction part sigmaZ*sigmaZ
	for (const auto & Jpart : m_Jpart) {
		mel[0] += 1./4. * (m_J * state[Jpart[0]] * state[Jpart[1]]);
	}
	
	for (const auto & J1part : m_J1part) {
		mel[0] += 1./4. * (m_J1 * state[J1part[0]] * state[J1part[1]]);
	}
	
	for (const auto & J2part : m_J2part) {
		mel[0] += 1./4. * (m_J2 * state[J2part[0]] * state[J2part[1]]);
	}
	
	//computing other part sigmaX*sigmaX + sigmaY*sigmaY
	for (const auto & Jpart : m_Jpart) {
		if (state[Jpart[0]] != state[Jpart[1]]) {
			mel.push_back(m_J * (-1./2.));
			flipsh.push_back(Jpart);
		}
	}
	
	for (const auto & J1part : m_J1part) {
		if (state[J1part[0]] != state[J1part[1]]) {
			mel.push_back(m_J1 * (-1./2.));
			flipsh.push_back(J1part);
		}
	}
	
	for (const auto & J2part : m_J2part) {
		if (state[J2part[0]] != state[J2part[1]]) {
			mel.push_back(m_J2 * (-1./2.));
			flipsh.push_back(J2part);
		}
	}
}

dict V15Isotropic::getParams() {
	dict params;
	params[m_name]["J"] = m_J;
	params[m_name]["J1"] = m_J1;
	params[m_name]["J2"] = m_J2;

	return params;
}