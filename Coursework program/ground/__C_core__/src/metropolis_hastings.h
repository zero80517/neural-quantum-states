#ifndef METROPOLIS_HASTINGS_H
#define METROPOLIS_HASTINGS_H

#include <variant>
#include "random.h"
#include "ising_1d_symmetric.h"
#include "v15_isotropic.h"
#include "heisenberg_1d_symmetric.h"
#include "nqs.h"

//Simple Monte Carlo sampler, that sampling a spin Wave-Function
//with Metropolis-Hastings algorithm
class MetropolisHastings {
private:
    //the Neural-network Quantum States
    Nqs & m_nqs;

    //the Hamiltonian
    std::variant<Ising1dSymmetric, V15Isotropic, Heisenberg1dSymmetric> m_hamiltonian;

    //number of spins
    const int m_nspins;

    //current state in the sampling
    list<int> m_cstate;

    //initial state of sampling
    list<int> m_istate;

    //if true, the initial state is prepared with zero total magnetization along z
    const bool m_zeromag;

    //is the total number of sweeps to be done
    const int m_nsweeps;

    //is the fraction of nsweeps to be discarded during the initial equilibration
    const double m_thermfactor;

    //set the number of random spin flips per sweeps to nspins*sweepfactor
    const double m_sweepfactor;

    //is the number of random spin flips to be done
    const int m_nflips;

    //error estimating with binning analysis consisting of nblocks bins
    const int m_nblocks;

    //the reuslts of sampling
    list<list<int> > m_samples;
    list<double> m_enlocal;
    double m_acceptance;
    double m_enmean;
    double m_enerror;
    double m_taucorr;

    //sampling statistics
    int m_nacceptance;
    int m_nmoves;

    //the name of the sampler
    str m_name = "MetropolisHastings";

    //container for indices of randomly chosen spins to be flipped
    list<int> m_flips;

    //quantities needed by the hamiltonian
    //non-zero matrix elements
    list<complex128> m_mel;

    //flip connectors for the hamiltonian(see body for details)
    list<list<int> > m_flipsh;

public:
    MetropolisHastings(
        Nqs & nqs, Ising1dSymmetric & hamiltonian, list<int> initstate=list<int>{0}, 
        bool zeromag=true, int nsweeps=10000, double thermfactor=0.1, 
        double sweepfactor=1., int nflips=1, int nblocks=50
    );
    MetropolisHastings(
        Nqs & nqs, V15Isotropic & hamiltonian, list<int> initstate=list<int>{0}, 
        bool zeromag=false, int nsweeps=10000, double thermfactor=0.1, 
        double sweepfactor=1., int nflips=1, int nblocks=50
    );
	MetropolisHastings(
        Nqs & nqs, Heisenberg1dSymmetric & hamiltonian, list<int> initstate=list<int>{0}, 
        bool zeromag=true, int nsweeps=10000, double thermfactor=0.1, 
        double sweepfactor=1., int nflips=2, int nblocks=50
    );
    inline void InitSampler();
    inline void InitRandomState(bool zeromag);
    inline int PickSite();
    void Run();
    inline void Move(int nflips, bool zeromag);
    inline bool RandSpin(list<int> & flips, int nflips, bool zeromag);
    inline double EnergyLocal();
    inline void EstimateEnergy();
    Nqs & getNqs() { return m_nqs; }
	int getNspins() const { return m_nspins; }
    int getNsweeps() const { return m_nsweeps; }
    list<list<int> > & getSamples() { return m_samples; }
    list<double> & getEnLocal() { return m_enlocal; }
	dict getParams();
    dict getResults();
};

#endif