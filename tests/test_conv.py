from multiprocessing import Pool, cpu_count
import os
import numpy as np
# track performance bottlenecks
from pyinstrument import Profiler

import qdotkmc



def main():


    # NOTE : a lot of the input parameters (especially the ones that are not used regularly)
    # have been moved as defaults to .config dataclasses. 

    # ---- QDLattice gometry ------
    ndim = 2                                    # number of dimensions
    N = 20                                      # number of QDs in each dimension

    # ---- system parameters ------
    inhomog_sd = 0.002                          # inhomogenous broadening (units?)
    nrg_center = 2.0                            # mean site energy (units ?)
    J_c = 10                                    # J_c (units?)

    # ----- bath parameters -------
    w_c = 0.1                                   # cutoff frequency (units?)
    temp = 200                                  # temperature (K)
    reorg_nrg = 0.01                            # reorganization energy (units?)

    # ---- KMC parameters ---------
    ntrajs = 10                                 # number of trajectories to compute MSDs over
    nrealizations = 8                           # number of disorder realizations (i.e. number of time we initialize a new QD lattice)

    rates_by = "weight"                         # select mode/strategy for rates comutation
    
    ntrajs = 20                                 # number of trajectories to compute MSDs over
    nrealizations = 8                           # number of disorder realizations (i.e. number of time we initialize a new QD lattice)
    #-------------------------------------------------------------------------
    # define dataclasses
    geom = qdotkmc.config.GeometryConfig(dims = ndim, N = N)
    dis  = qdotkmc.config.DisorderConfig(nrg_center = nrg_center, inhomog_sd = inhomog_sd, J_c = J_c)
    bath_cfg = qdotkmc.config.BathConfig(temp = temp, w_c = w_c, reorg_nrg = reorg_nrg)
    run  = qdotkmc.config.RunConfig(ntrajs = ntrajs, nrealizations = nrealizations)

    # check .config to see defaults here
    exec_plan = qdotkmc.config.ExecutionPlan(prefer_gpu = True,
                                             gpu_use_c64 = True,
                                             do_parallel = True)
    

    # input parameter configuration for convergence
    # NOTE : this convergence is currenly only implemented for weight-based rate modus
    tune_cfg = qdotkmc.config.ConvergenceTuneConfig(no_samples      = 20,
                                                    criterion       = "rate-displacement",
                                                    theta_site_lo   = 0.30,
                                                    theta_site_hi   = 0.001,
                                                    theta_pol_start = 0.30,
                                                    theta_pol_min   = 0.001,
                                                    rho             = 0.8,
                                                    delta           = 0.05               
                                                    )

    # set up convergence
    convergence_setup = qdotkmc.convergence.ConvergenceAnalysis(geom, dis, bath_cfg, run, exec_plan, tune_cfg)
    

    #  ---------    automatically optimize thetas    ----------
    # set verbose = False/True to avoid/enable printing updates on convergence

    thetas_result = convergence_setup.auto_tune_thetas(verbose=True)
    convergence_setup._clean()

    theta_site_opt = thetas_result['theta_site']
    theta_pol_opt = thetas_result['theta_pol']
    
    print(f"theta_site (opt): {theta_site_opt:.4f}")
    print(f"theta_pol (opt): {theta_pol_opt:.4f}")

    # ----------   use thetas for KMC simulation     ----------
    # setup KMC simulation
    kmc = qdotkmc.montecarlo.KMCRunner(geom, dis, bath_cfg, run, exec_plan)
    
    kmc.run.rates_by = rates_by
    kmc.run.theta_site = theta_site_opt                     # feed in optimized parameter for theta_site
    kmc.run.theta_pol = theta_pol_opt                       # feed in optimized paramter for theta_pol
    
    # perform KMC simulation (automatically switches parallel/serial based on max_workers)
    times, msds = kmc._simulate_kmc()

    # export msds as .csv file for inspection
    qdotkmc.utils.export_msds(times, msds)

    # get noise-averaged (pooled) trajectory MSD
    msds_mean = np.mean(msds, axis = 0)

    # obtain diffusivities in two distinct ways
    diff1, sigma_D1 = qdotkmc.utils.get_diffusivity(msds_mean, times, ndim)

    diff2, sigma_D2 = qdotkmc.utils.summarize_diffusivity(msds, times, ndim)

    # -------------------------------------------------------------------------
    print('diffusivity ', diff1, diff2)
    print('diffusivity error', sigma_D1, sigma_D2)
    

if __name__ == '__main__':

    profiler = Profiler(use_timing_thread=True)
    profiler.start()

    try:
        main()
    finally:
        profiler.stop()
        submit_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()) 
        output_path = os.path.join(submit_dir, "pyinstrument_output.txt")
        with open(output_path, "w") as f:
            f.write(profiler.output_text(unicode=True, color=False))