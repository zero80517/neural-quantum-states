#include <random>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include "random.h"

Random _inst;

Random::Random() { 
	std::random_device rd;
	m_prng.seed(rd());
}

void Random::seed(unsigned int seed) {
	m_prng.seed(seed);
}

double Random::normal(double loc, double scale) {
	std::normal_distribution<double> n_dist{loc, scale};

	return n_dist(m_prng);
}

int Random::randrange(int start, int stop, int step) {
	/*Choose a random item from range(start, stop[, step]).

	This fixes the problem with randint() which includes the
	endpoint; in Python this is usually not what you want.

	*/
	
	// This code is a bit messy to make it fast for the
	// common case while still doing adequate error checking.
	if (stop == 0) {
		if (start > 0) {
			return randint(0, start-1);
		}
		throw pybind11::value_error("empty range for randrange()");
		std::exit(EXIT_FAILURE);
	}

	// stop argument supplied.
	int width = stop - start;
	if (step == 1 && width > 0) {
		return start + randint(0, width-1);
	}
	if (step == 1) {
		throw pybind11::value_error("empty range for randrange()");
		std::exit(EXIT_FAILURE);
	}

	// step argument supplied.
	int n;
	if (step > 0) {
		n = (width + step - 1) / step;
	} else if (step < 0) {
		n = (width + step + 1) / step;
	} else {
		throw pybind11::value_error("zero step for randrange()");
		std::exit(EXIT_FAILURE);
	}

	if (n <= 0) {
		throw pybind11::value_error("empty range for randrange()");
		std::exit(EXIT_FAILURE);
	}

	return start + step*randint(0, n-1);
}

int Random::randint(int a, int b) {
	/*Return random integer in range [a, b], including both end points.
    */
	
	std::uniform_int_distribution<int> uint_dist{a, b};
	
	return uint_dist(m_prng);
}

double Random::random() {
	/*Get the next random number in the range [0.0, 1.0).*/
	std::uniform_real_distribution<double> ureal_dist{0, 1};

	return ureal_dist(m_prng);
}