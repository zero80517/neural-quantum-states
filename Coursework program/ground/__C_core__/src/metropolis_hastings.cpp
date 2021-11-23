#include <cstdlib>
#include <pybind11/pybind11.h>
#include "metropolis_hastings.h"

extern Random _inst;

MetropolisHastings::MetropolisHastings(
	Nqs & nqs, Ising1dSymmetric & hamiltonian, list<int> initstate, 
	bool zeromag, int nsweeps, double thermfactor, 
	double sweepfactor, int nflips, int nblocks):
    	m_nqs(nqs), m_hamiltonian(hamiltonian), m_nspins(nqs.getNspins()), m_istate(initstate), 
    	m_zeromag(zeromag), m_nsweeps(nsweeps), m_thermfactor(thermfactor), m_sweepfactor(sweepfactor), 
    	m_nflips(nflips), m_nblocks(nblocks) 
{
    InitSampler();
}

MetropolisHastings::MetropolisHastings(
	Nqs & nqs, V15Isotropic & hamiltonian, list<int> initstate, 
	bool zeromag, int nsweeps, double thermfactor, 
	double sweepfactor, int nflips, int nblocks):
    	m_nqs(nqs), m_hamiltonian(hamiltonian), m_nspins(nqs.getNspins()), m_istate(initstate), 
    	m_zeromag(zeromag), m_nsweeps(nsweeps), m_thermfactor(thermfactor), m_sweepfactor(sweepfactor), 
    	m_nflips(nflips), m_nblocks(nblocks) 
{
    InitSampler();
}

MetropolisHastings::MetropolisHastings(
	Nqs & nqs, Heisenberg1dSymmetric & hamiltonian, list<int> initstate, 
	bool zeromag, int nsweeps, double thermfactor, 
	double sweepfactor, int nflips, int nblocks):
    	m_nqs(nqs), m_hamiltonian(hamiltonian), m_nspins(nqs.getNspins()), m_istate(initstate), 
    	m_zeromag(zeromag), m_nsweeps(nsweeps), m_thermfactor(thermfactor), m_sweepfactor(sweepfactor), 
    	m_nflips(nflips), m_nblocks(nblocks) 
{
    InitSampler();
}

inline void MetropolisHastings::InitSampler() {
	if (m_istate[0] == 0) {
    	InitRandomState(m_zeromag);
    	m_istate = m_cstate;
    } else {
    	m_cstate = m_istate;
    }

    if (m_nflips != 1 && m_nflips != 2) {
    	throw pybind11::value_error("invalid number of spin flips");
		std::exit(EXIT_FAILURE);				
    }

    if (m_thermfactor > 1 || m_thermfactor < 0) {
    	throw pybind11::value_error("the thermalization factor should be "
    		                        "a real number between 0 and 1");
		std::exit(EXIT_FAILURE);				
    }

    if (m_nblocks > m_nsweeps) {
    	throw pybind11::value_error("Please enter a number of sweeps sufficiently large "
    		                        "the number of blocks");
		std::exit(EXIT_FAILURE);	
    }

    m_samples.resize(m_nsweeps, list<int> (m_nspins, 0));
    m_enlocal.resize(m_nsweeps);

    m_nmoves = m_nsweeps * int(m_sweepfactor*m_nspins);
}

//Initializes a random spin state
//if zeromag=true, the initial state is prepared with zero total magnetization
inline void MetropolisHastings::InitRandomState(bool zeromag) {
    m_cstate.resize(m_nspins);
    for (int i=0; i < m_nspins; ++i){
      	m_cstate[i] = _inst.randrange(-1, 2, 2);
    }

    if (zeromag) {
      	int tmag=1;
      	if (m_nspins % 2) {
      		throw pybind11::value_error("cannot initializate a random state with zero magnetization "
      			                        "for odd number of spins");
			std::exit(EXIT_FAILURE);
      	}

      	while (tmag != 0) {
       		tmag=0;
	        for (int i=0; i < m_nspins; ++i) {
	          	tmag += m_cstate[i];
	        }

	        if (tmag > 0) {
				int rs=PickSite();
				while (m_cstate[rs] < 0) {
					rs=PickSite();
				}
				m_cstate[rs] = -1;
				tmag -= 1;
	        }

	        else if (tmag < 0) {
	          	int rs=PickSite();
	          	while (m_cstate[rs] > 0) {
	            	rs=PickSite();
	          	}
	          	m_cstate[rs] = 1;
	          	tmag += 1;
	        }
        }
    }
}

inline int MetropolisHastings::PickSite() {
	int site = _inst.randint(0, m_nspins - 1);
	return site;
}

//run the Monte Carlo sampling
void MetropolisHastings::Run() {
    m_cstate = m_istate;
    m_nacceptance = 0;

    m_flips.resize(m_nflips);

    //initializing look-up tables in the NQS
    m_nqs.InitLt(m_cstate);

    //thermalization
    for (int n=0; n < int(m_nsweeps*m_thermfactor); ++n) { 
      	for (int i=0; i < int(m_nspins*m_sweepfactor); ++i) {
        	Move(m_nflips, m_zeromag);
      	}
    }

    m_nacceptance = 0;

    //sequence of sweeps
    for (int n=0; n < m_nsweeps; ++n) {
        for (int i=0; i < int(m_nspins*m_sweepfactor); ++i) {
        	Move(m_nflips, m_zeromag);
        }
        m_enlocal[n] = EnergyLocal();
        m_samples[n] = m_cstate;
    }
    
    m_acceptance = double(m_nacceptance) / m_nmoves; 

    EstimateEnergy();
}

inline void MetropolisHastings::Move(int nflips, bool zeromag) {
    //Picking "nflips" random spins to be flipped
    if (RandSpin(m_flips, nflips, zeromag)) {

        //Computing acceptance probability
        double acceptance = std::norm(m_nqs.PoP(m_cstate, m_flips));

        //Metropolis-Hastings test
        if (acceptance > _inst.random()) {

	        //Updating look-up tables in the machine
	        m_nqs.UpdateLt(m_cstate, m_flips);

	        //Moving to the new configuration
	        for (const auto& flip : m_flips) {
	            m_cstate[flip] *= -1;
	        }

        	m_nacceptance += 1;
        }
    }
}

//Random spin flips (max 2 spin flips in this implementation)
//if zeromag=true, when doing 2 spin flips the total magnetization is kept equal to 0
inline bool MetropolisHastings::RandSpin(list<int> & flips, int nflips, bool zeromag) {
    flips.resize(nflips);

    flips[0] = PickSite();
    if (nflips == 2) {
      	flips[1] = PickSite();
      	if (!zeromag) {
        	return flips[1] != flips[0];
        } else {
        	return m_cstate[flips[1]] != m_cstate[flips[0]];
        }
    }

    return true;
}

//Measuring the value of the local energy
//on the current state
inline double MetropolisHastings::EnergyLocal(){
    double enlocal=0.;

    //Finds the non-zero matrix elements of the hamiltonian
    //on the given state
    //i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
    //state' is encoded as the sequence of spin flips to be performed on state
    if (std::get_if<Ising1dSymmetric> (&m_hamiltonian)) 
        (std::get<Ising1dSymmetric> (m_hamiltonian)).FindConn(m_cstate, m_flipsh, m_mel);
        
    else if (std::get_if<V15Isotropic> (&m_hamiltonian)) 
        (std::get<V15Isotropic> (m_hamiltonian)).FindConn(m_cstate, m_flipsh, m_mel);
	
	else if (std::get_if<Heisenberg1dSymmetric> (&m_hamiltonian)) 
        (std::get<Heisenberg1dSymmetric> (m_hamiltonian)).FindConn(m_cstate, m_flipsh, m_mel);

    for (int i=0; i < int(m_flipsh.size()); ++i) {
        enlocal += (m_nqs.PoP(m_cstate, m_flipsh[i]) * m_mel[i]).real();
    }

    return enlocal;
}

inline void MetropolisHastings::EstimateEnergy() {
    int nblocks=m_nblocks;

    int blocksize=std::floor(double(m_enlocal.size())/double(nblocks));

    double enmean=0;
    double enmeansq=0;

    double enmean_unblocked=0;
    double enmeansq_unblocked=0;

    for (int i=0; i < nblocks; ++i) {
        double eblock=0;
        for(int j=i*blocksize; j < (i+1)*blocksize; ++j) {
	        eblock+=m_enlocal[j];
	        double delta=m_enlocal[j]-enmean_unblocked;
	        enmean_unblocked+=delta/double(j+1);
	        double delta2=m_enlocal[j]-enmean_unblocked;
	        enmeansq_unblocked+=delta*delta2;
    	}
		eblock/=double(blocksize);
		double delta=eblock-enmean;
		enmean+=delta/double(i+1);
		double delta2=eblock-enmean;
		enmeansq+=delta*delta2;
    }

    enmeansq/=(double(nblocks-1));
    enmeansq_unblocked/=(double((nblocks*blocksize-1)));

    m_enmean=enmean;
    m_enerror=std::sqrt(enmeansq/double(nblocks));
    m_taucorr=0.5*double(blocksize)*enmeansq/enmeansq_unblocked;
}

dict MetropolisHastings::getParams() {
    dict params;
    params[m_name]["nsweeps"] = m_nsweeps;
    params[m_name]["thermfactor"] = m_thermfactor;
    params[m_name]["sweepfactor"] = m_sweepfactor;
    params[m_name]["nflips"] = m_nflips;
    params[m_name]["nmoves"] = m_nmoves;
    
    return params;
}

dict MetropolisHastings::getResults() {
    dict results;
    results["sampler results"]["acceptance"] = m_acceptance;
    results["sampler results"]["nacceptance"] = m_nacceptance;
    results["sampler results"]["energy mean"] = m_enmean;
    results["sampler results"]["energy error"] = m_enerror;
    results["sampler results"]["taucorr"] = m_taucorr;

    return results;
}