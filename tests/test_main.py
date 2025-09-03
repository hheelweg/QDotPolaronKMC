import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import pandas as pd
import os
import numpy as np
# track performance bottlenecks
from pyinstrument import Profiler
import qdotkmc



def main():

    # NOTE : a lot of the input parameters (especially the ones that are not used regularly)
    # have been moved as defaults to .config dataclasses. 

    # ---- QDLattice gometry ------
    ndim = 1                                    # number of dimensions
    N = 400                                     # number of QDs in each dimension

    # ---- system parameters ------
    inhomog_sd = 0.03                          # inhomogenous broadening (units?) (legacy: 0.002)
    nrg_center = 2.0                            # mean site energy (units ?) (legacy: 2.0)
    J_c = 30                                    # J_c (units?) (legacy: 10)

    # ----- bath parameters -------
    w_c = 0.03                                  # cutoff frequency (units?) (legacy: 0.1)
    temp = 300                                  # temperature (K) (legacy: 200)
    reorg_nrg = 0.1                            # reorganization energy (units?)

    # ---- KMC parameters ---------
    ntrajs = 5000                                 # number of trajectories to compute MSDs over
    nrealizations = 50                          # number of disorder realizations (i.e. number of time we initialize a new QD lattice)
    t_final = 10

    rates_by = "weight"                         # select mode/strategy for rates comutation
    # NOTE : as soon as we pick "radius" or "weight" we confine ourselves ro r_hop/r_ove or theta_site/theta_pol
    # here we leave both 

    # (2) for "weight"
    theta_site = 0.01                           # legacy 0.05
    theta_pol = 0.01                            # legacy 0.05

    #-------------------------------------------------------------------------

    # define dataclasses
    geom = qdotkmc.config.GeometryConfig(dims = ndim, N = N)
    dis  = qdotkmc.config.DisorderConfig(nrg_center = nrg_center, inhomog_sd = inhomog_sd, J_c = J_c)
    bath_cfg = qdotkmc.config.BathConfig(temp = temp, w_c=w_c, reorg_nrg=reorg_nrg)
    run  = qdotkmc.config.RunConfig(ntrajs = ntrajs, nrealizations = nrealizations,
                                    rates_by = rates_by, 
                                    theta_site = theta_site, theta_pol = theta_pol, 
                                    t_final = t_final)
    
    # check .config to see defaults here
    exec_plan = qdotkmc.config.ExecutionPlan(prefer_gpu = True,
                                             gpu_use_c64 = True,
                                             do_parallel = True)
    
    # set up KMC simulation
    kmc = qdotkmc.montecarlo.KMCRunner(geom, dis, bath_cfg, run, exec_plan, backend_verbose=True)

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
