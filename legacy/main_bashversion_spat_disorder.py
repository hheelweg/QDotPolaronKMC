import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import qdotkmc
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os
import sys

def main():
    N = int(sys.argv[1])
    ndims = np.array([1]).astype(int)
    J_cs = np.array([0.3, 3, 30])
    inhomog_sds = np.array([0.001, 0.003, 0.01, 0.03])
    reorg_nrgs = np.array([0.03, 0.1, 0.3])
    w_cs = np.array([0.01, 0.03, 0.1])
    # rel_spatial_disorders = np.array([0, 0.1, 0.5])
    # r_hops = np.array([4,5]).astype(int)
    # r_oves = np.array([5.5, 6.5])
    
    
    
    
    # number of cores you have allocated for your SLURM task:
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    
    #number of parallel runs
    numPara=number_of_cores
    #-------------------------------------------------------------------------
    args=[]
    for h in range(len(ndims)):
        for i in range(len(J_cs)):
            for j in range(len(inhomog_sds)):
                for k in range(len(reorg_nrgs)):
                    for l in range(len(w_cs)):
                        params = [ndims[h], J_cs[i], inhomog_sds[j], reorg_nrgs[k], w_cs[l]]
                        args.append((params, N))
    
    with Pool(numPara) as pool:
        results = pool.starmap(run_sim, args)
    # print(results)


def run_sim(args, N):
    print("run started", flush = True)
    temps = np.linspace(1, 1000, 50)
    # rel_spatial_disorder = args[5]
    
    # NOTE : a lot of the input parameters (especially the ones that are not used regularly)
    # have been moved as defaults to .config dataclasses. 

    # ---- QDLattice gometry ------
    ndim = args[0]                                    # number of dimensions
    # N = 50                                      # number of QDs in each dimension

    # ---- system parameters ------
    inhomog_sd = args[2]                          # inhomogenous broadening (units?)
    nrg_center = 2.0                            # mean site energy (units ?)
    J_c = args[1]                                    # J_c (units?)

    # ----- bath parameters -------
    w_c = args[4]                                   # cutoff frequency (units?)
    # temp = 200                                  # temperature (K)
    reorg_nrg = args[3]                            # reorganization energy (units?)

    # ---- KMC parameters ---------
    ntrajs = 5000                                 # number of trajectories to compute MSDs over
    nrealizations = 50                           # number of disorder realizations (i.e. number of time we initialize a new QD lattice)
    t_final = 10

    rates_by = "weight"                         # select mode/strategy for rates comutation
    # NOTE : as soon as we pick "radius" or "weight" we confine ourselves ro r_hop/r_ove or theta_site/theta_pol
    # here we leave both 

    # (2) for "weight"
    theta_site = 0.01
    theta_pol = 0.01


    # define dataclasses
    geom = qdotkmc.config.GeometryConfig(dims = ndim, N = N)
    dis  = qdotkmc.config.DisorderConfig(nrg_center = nrg_center, inhomog_sd = inhomog_sd, J_c = J_c)
    run  = qdotkmc.config.RunConfig(ntrajs = ntrajs, nrealizations = nrealizations,
                                    rates_by = rates_by, 
                                    theta_site = theta_site, theta_pol = theta_pol, 
                                    t_final = t_final
                                    )
    
    # -------------------------------------------------------------------------
    # print('diffusivity ', diff1, diff2)
    # print('diffusivity error', sigma_D1, sigma_D2)
    
    for temp in temps:
        bath_cfg = qdotkmc.config.BathConfig(temp = temp, w_c=w_c, reorg_nrg=reorg_nrg)
        
        # set up KMC simulation
        kmc = qdotkmc.montecarlo.KMCRunner(geom, dis, bath_cfg, run)
        # perform KMC simulation (automatically switches parallel/serial based on max_workers)
        times, msds = kmc._simulate_kmc()

        # export msds as .csv file for inspection
        qdotkmc.utils.export_msds(times, msds)
        
        # get noise-averaged (pooled) trajectory MSD
        msds_mean = np.mean(msds, axis = 0)
        
        diff1, sigma_D1 = qdotkmc.utils.get_diffusivity(msds_mean, times, ndim)
        
        filenamebase = ("J_c_%f,sigma_i_%f,reorg_nrg_%f,w_c_%f")%(J_c, inhomog_sd, reorg_nrg, w_c)
        results_txt = open("results_" + filenamebase + ".txt", "a")
        results_txt.write(str(diff1) + "," + str(sigma_D1) + "\n")
        results_txt.close()


if __name__ == '__main__':
    main()
