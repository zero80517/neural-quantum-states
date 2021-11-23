#include <iostream>
#include <cstdlib>
#include <cmath>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "random.h"
#include "nqs.h"

extern Random _inst;

Nqs::Nqs(int nvisible, int alpha, double sigma,
	     list<complex128> a, list<complex128> b, list<list<complex128> > W
	    ): m_nvisible(nvisible), m_nhidden(alpha * nvisible), m_a(a), m_b(b), m_W(W) 
{ 
	if ( std::isnan(m_a[0].real()) ) {
		m_a.clear();
		m_a.resize(m_nvisible);
		for (int i=0; i < m_nvisible; ++i) {
			m_a[i] = _inst.normal(0., sigma);
		}
	} else {
		m_nvisible = a.size();
	}

	if ( std::isnan(m_b[0].real()) ) {
		m_b.clear();
		m_b.resize(m_nhidden);
		for (int j=0; j < m_nhidden; ++j) {
			m_b[j] = _inst.normal(0., sigma);
		}
	} else {
		m_nhidden = b.size();
	}

	if ( std::isnan(m_W[0][0].real()) ) {
		m_W.clear();
		m_W.resize(m_nvisible, list<complex128> (m_nhidden, 0.));
		for (int i=0; i < m_nvisible; ++i) {
				for (int j=0; j < m_nhidden; ++j) {
				m_W[i][j] = _inst.normal(0., sigma);
			}
		}
	}
}

//initialization of the look-up tables
void Nqs::InitLt(const list<int> & state) {
	m_Lt.clear();
	m_Lt.resize(m_nhidden);

	for(int h=0; h < m_nhidden; ++h) {
		m_Lt[h]=m_b[h];
		for(int v=0; v < m_nvisible; ++v) {
			m_Lt[h] += double(state[v])*(m_W[v][h]);
		}
	}
}

/*
Computes the ratio of wave function amplitudes Psi(state')/Psi(state)
where state' is obtained by flipping the spins of state

flips contains a list of which spins to flip to get from
state to state
*/
complex128 Nqs::PoP(const list<int> & state, const list<int> & flips) const {
	return std::exp(LogPoP(state, flips));
}

//computes the logarithm of Psi(state')/Psi(state)
//where state' is a state with a certain number of flipped spins
//the list "flips" contains the sites to be flipped
//look-up tables are used to speed-up the calculation
inline complex128 Nqs::LogPoP(const list<int> & state, const list<int> & flips) const {
	if (flips.size() == 0) {
		return 0.;
	}

	complex128 logpop(0., 0.);

	//Change due to the visible bias
	for (const auto & flip : flips) {
		logpop -= m_a[flip] * 2.*double(state[flip]);
	}

	//Change due to the interaction weights
	for (int h=0; h < m_nhidden; ++h) {
		const complex128 thetah=m_Lt[h];
		complex128 thetahp=thetah;

		for (const auto & flip : flips) {
			thetahp -= 2.*double(state[flip]) * (m_W[flip][h]);
		}

		logpop += ( Nqs::lncosh(thetahp) - Nqs::lncosh(thetah) );
	}

	return logpop;
}

//ln(cos(x)) for real argument
//for large values of x we use the asymptotic expansion
inline double Nqs::lncosh(double x) const {
	const double xp=std::abs(x);

	if (xp <= 12.) {
		return std::log(std::cosh(xp));
	} else {
		return xp - m_log2;
	}
}

//ln(cos(x)) for complex argument
//the modulus is computed by means of the previously defined function
//for real argument
inline complex128 Nqs::lncosh(complex128 x) const {
	const double xr=x.real();
	const double xi=x.imag();

	complex128 res=Nqs::lncosh(xr);
	res += std::log( complex128(std::cos(xi), std::tanh(xr)*std::sin(xi)) );

	return res;
}	

//updates the look-up tables after spin flips
//the list "flips" contains the indices of sites to be flipped
void Nqs::UpdateLt(const list<int> & state, const list<int> & flips) {
    if (flips.size() == 0) {
      return;
    }

    for (int h=0; h < m_nhidden; ++h) {
		for (const auto & flip : flips) {
			m_Lt[h] -= 2.*double(state[flip]) * m_W[flip][h];
		}
    }
}	

//updates the biases and weights of the network.
void Nqs::UpdateWeights(const list<complex128> upvals) {
	long unsigned int need=m_nvisible + m_nhidden + m_nvisible*m_nhidden;
	if (upvals.size() != need) {
		throw pybind11::value_error("wrong size for updates");
		std::exit(EXIT_FAILURE);
	} else {
		for (int v=0; v < m_nvisible; ++v) {
			m_a[v] += upvals[v];
		}

		for (int h=0; h < m_nhidden; ++h) {
			m_b[h] += upvals[m_nvisible+h];
		}

		for (int v=0; v < m_nvisible; ++v) {
			for (int h=0; h < m_nhidden; ++h) {
				m_W[v][h] += upvals[ m_nvisible+m_nhidden*(1+v)+h ];
			}
		}
	}
}

dict Nqs::getParams() {
	dict params;
	params[m_name]["n_visible"] = m_nvisible;
	params[m_name]["n_hidden"] = m_nhidden;

	list<complex128> weights;
	weights.resize(m_nvisible + m_nhidden + m_nvisible*m_nhidden);

	for (int v=0; v < m_nvisible; ++v) {
		weights[v] = m_a[v];
	}

	for (int h=0; h < m_nhidden; ++h) {
		weights[ m_nvisible+h ] = m_b[h];
	}

	for (int v=0; v < m_nvisible; ++v) {
		for (int h=0; h < m_nhidden; ++h) {
			weights[ m_nvisible+m_nhidden*(1+v)+h ] = m_W[v][h];
		}
	}

	params[m_name]["weights"] = weights;

	return params;
}