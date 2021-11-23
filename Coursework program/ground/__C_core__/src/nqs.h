#ifndef NQS_H
#define NQS_H

#include <cmath>
#include <map>
#include <variant>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "random.h"

using complex128 = std::complex<double>;
using str = std::string;
template<typename T> using list = std::vector<T>;
using dict = std::map<str, 
	std::map<str, std::variant<str, int, double, complex128, 
		list<str>, list<int>, list<double>, list<complex128> > > >;

class Nqs {
private:
	//Number of visible units
	int m_nvisible;

	//Number of hidden units
	int m_nhidden;

	//Neural-network visible bias
	list<complex128> m_a;

	//Neural-network hidden bias
	list<complex128> m_b;

	//Neural-network weights
    list<list<complex128> > m_W;

	//look-up tables or theta
	list<complex128> m_Lt;

	//Name of the machine
	str m_name = "NQS";

	//Useful quantities for safe computation of ln(cosh(x))
	const double m_log2 = std::log(2);

public:
	Nqs(int nvisible=20, int alpha=2, double sigma=0.001,
		list<complex128> a={std::nan("0")},
		list<complex128> b={std::nan("0")},
		list<list<complex128> > W={{std::nan("0")}});
	void InitLt(const list<int> & state);
	complex128 PoP(const list<int> & state, const list<int> & flips) const;
	inline complex128 LogPoP(const list<int> & state,const list<int> & flips) const;
	inline double lncosh(double x) const;
	inline complex128 lncosh(complex128 x) const;
	void UpdateLt(const list<int> & state, const list<int> & flips);
	void UpdateWeights(const list<complex128> upvals);
	int getNspins() { return m_nvisible; }
	int getNhidden() { return m_nhidden; }
	list<complex128> & getA() { return m_a; }
	list<complex128> & getB() { return m_b; }
	list<list<complex128> > & getW() { return m_W; }
	dict getParams();
};

#endif