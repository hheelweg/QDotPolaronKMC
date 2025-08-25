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
    ndim = 2                                    # number of dimensions
    N = 50                                      # number of QDs in each dimension

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
    nrealizations = 2                           # number of disorder realizations (i.e. number of time we initialize a new QD lattice)

    rates_by = "weight"                         # select mode/strategy for rates comutation
    # NOTE : as soon as we pick "radius" or "weight" we confine ourselves ro r_hop/r_ove or theta_site/theta_pol
    # here we leave both 

    # (2) for "weight"
    theta_site = 0.01
    theta_pol = 0.01
    
    #-------------------------------------------------------------------------
    # obtain max_workers from SLURM environment for parallelization of work
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
    # enforce serial for debugging
    max_workers = 1


    # define dataclasses
    geom = qdotkmc.config.GeometryConfig(dims = ndim, N = N)
    dis  = qdotkmc.config.DisorderConfig(nrg_center = nrg_center, inhomog_sd = inhomog_sd, J_c = J_c)
    bath_cfg = qdotkmc.config.BathConfig(temp = temp, w_c=w_c, reorg_nrg=reorg_nrg)
    run  = qdotkmc.config.RunConfig(ntrajs = ntrajs, nrealizations = nrealizations,
                                    rates_by = rates_by, 
                                    theta_site = theta_site, theta_pol = theta_pol, 
                                    max_workers = max_workers
                                    )
    
    # set up KMC simulation
    kmc = qdotkmc.montecarlo.KMCRunner(geom, dis, bath_cfg, run)

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

    profiler = Profiler()
    profiler.start()

    try:
        main()
    finally:
        profiler.stop()
        submit_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()) 
        output_path = os.path.join(submit_dir, "pyinstrument_output.txt")
        with open(output_path, "w") as f:
            f.write(profiler.output_text(unicode=True, color=False))
