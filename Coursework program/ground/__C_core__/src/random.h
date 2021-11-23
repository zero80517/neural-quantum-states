#ifndef RANDOM_H
#define RANDOM_H

#include <random>

class Random {
private:
	std::mt19937 m_prng;
public:
	Random ();
	void seed(unsigned int seed);
	double normal(double loc=0., double scale=1.);
	int randrange(int start, int stop=0, int step=1);
	int randint(int a, int b);
	double random();
};

#endif