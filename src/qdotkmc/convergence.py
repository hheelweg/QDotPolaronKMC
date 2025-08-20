import numpy as np
from .config import GeometryConfig, DisorderConfig, BathConfig, RunConfig
from numpy.random import SeedSequence, default_rng
from . import lattice, hamiltonian_box, const
from .hamiltonian_box import SpecDens
from .montecarlo import KMCRunner


class ConvergenceAnalysis(KMCRunner):

    def __init__(self, geom : GeometryConfig, dis : DisorderConfig, bath_cfg : BathConfig, run : RunConfig):

        super().__init__(geom, dis, bath_cfg, run)
        # TODO : load some more class attributes here?
    

    def _build_rate_convergenc_env(self, no_samples):

        # (0) build bath
        bath = SpecDens(self.bath_cfg.spectrum, const.kB * self.bath_cfg.temp)

        # (1) draw a lattice realization
        qd_lattice = self._build_grid_realization(bath, rid=0)

        # (2) produce no_samples starting indices from where to compute rate vectors
        ss_conv = self._ss_root.spawn(1)[0]
        rng_conv = default_rng(ss_conv)
        start_sites = rng_conv.integers(0, qd_lattice.geom.n_sites, size=no_samples)

        # (3) get Boltzmann weights for each start polaron in start_sites
        E = qd_lattice.full_ham.evals
        beta = qd_lattice.full_ham.beta
        w = np.exp(- beta * E[start_sites])
        Z = np.sum(np.exp(- beta * E))
        weights = w / Z

        return qd_lattice, start_sites, weights
    

    # TODO : write input parameters so that we can also use this for r_hop/r_ove
    def _rate_score(self, theta_pol, theta_sites, 
                    qd_lattice, start_sites, weights,
                    criterion=None, score_info=False,
                    ):

        # get rates starting from each polaron starting index and analyze by criterion
        rates_criterion = None
        nsites_sel, npols_sel = 0, 0
        for start_idx in start_sites:

            # NEW way to obtain rates from theta_pol/theta_sites
            rates, final_sites, _, sel_info = self._make_kmatrix_boxNEW(qd_lattice, start_idx,
                                                                        theta_sites=theta_sites,
                                                                        theta_pol=theta_pol,
                                                                        selection_info = True
                                                                        )
            
            # how many polarons/sites were selected 
            nsites_sel += sel_info['nsites_sel']
            npols_sel += sel_info['npols_sel']
            
            # evaluate convergence criterion on rates vector
            if criterion == "rate-displacement":
                start_loc = qd_lattice.qd_locations[start_idx]                                                      # r(0)
                sq_displacments = ((qd_lattice.qd_locations[final_sites] - start_loc)**2).sum(axis = 1)             # ||Δr||^2 per destination
                lamda = (rates * sq_displacments).sum() / (2 * qd_lattice.geom.dims)
                rates_criterion = (weights * lamda).sum()
            else:
                raise ValueError("please specify valid convergence criterion for rates!")
            

        # optional : store additional information
        info = {}
        if score_info:
            info['ave_sites'] = nsites_sel / len(start_sites)
            info['ave_pols'] = npols_sel / len(start_sites)

        return rates_criterion, info
