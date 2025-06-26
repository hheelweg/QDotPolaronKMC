import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import src.montecarlo as mc
import src.utils as utils
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os

def main():
    ndims = np.array([1]).astype(int)
    J_cs = np.array([0.3, 3, 30])
    inhomog_sds = np.array([0, 0.001, 0.01])
    reorg_nrgs = np.array([0.03, 0.1, 0.3])
    w_cs = np.array([0.01, 0.03, 0.1])
    temps = np.linspace(1, 1000, 20)
    rel_spatial_disorders = np.array([0, 0.1, 0.5])
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
                        for m in range(len(rel_spatial_disorders)):
                            params = [ndims[h], J_cs[i], inhomog_sds[j], reorg_nrgs[k], w_cs[l], rel_spatial_disorders[m]]
                            args.append((params, 1))
    
    with Pool(numPara) as pool:
        results = pool.starmap(run_sim, args)
    # print(results)


def run_sim(args, dummy = 1):
    print("run started", flush = True)
    ndim = args[0]
    J_c = args[1]
    inhomog_sd = args[2]
    reorg_nrg = args[3]
    w_c = args[4]
    temps = np.linspace(1, 1000, 20)
    rel_spatial_disorder = args[5]
    
    
    # ndim = 1                                    # number of dimensions
    N = 200                                      # number of QDs in each dimension
    nc_edgelength = 8                           # length of each QD (units?)
    ligand_length = 1                           # length of ligands on QD (units?)

    # seed for randomness of Hamiltonian (if None, then Hamiltonian is randomly drawn for every instance of the class)
    seed = None

    # Hamiltonian and bath related parameters
    # reorg_nrg = 0.01                            # reorganization energy (units?)
    # w_c = 0.1                                   # cutoff frequency (units?) 
    # J_c = 30                                    # J_c (units?)
    # inhomog_sd = 0.002                           # inhomogenous broadening (units?)
    nrg_center = 2.0                            # mean site energy (units ?)
    # rel_spatial_disorder = 0.0                  # relative spatial disorder
    dipolegen = 'random'                        # dipole generation procedure
    # temp = 200                                  # temperature (K)
    spec_density = 'cubic-exp'                  # bath spectral density

    # PTRE and KMC related parameters
    numtrials = 1                               # number of trials to average over (here: 3)
    method = 'first-order'                      # method for computing bath integrals 
    r_hop = 5                                   # hopping radius (see Kassal) (in units of lattice spacing)
    r_ove = 6.5
    r_box = math.ceil(min(r_hop, r_ove))        
                                # overlap radius (see Kassal) (in units of lattice spacing)

    ntrajs = 1000                                 # number of trajectories to compute MSDs over
    t_final = 3                                   # final time for each trajectory (units?)
    #-------------------------------------------------------------------------

    # lattice spacing
    spacing = nc_edgelength + 2 * ligand_length

    # specify the bath
    spectrum = [spec_density, reorg_nrg, w_c, method]

    for temp in temps:
        # greate instance of MC class to run KMC simulation
        kmc_setup = mc.KMCRunner(ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                                    J_c, spectrum, temp, ntrajs, r_hop, r_ove, r_box)
        
        # perform a KMC simulation
        times, msds = kmc_setup.NEW_simulate_kmc(t_final)
        
        diff, diff_err = kmc_setup.get_diffusivity_hh(msds, times, ndim)
        
        filenamebase = ("J_c_%f,sigma_i_%f,reorg_nrg_%f,w_c_%f,spat_disorder_%f")%(J_c, inhomog_sd, reorg_nrg, w_c, rel_spatial_disorder)
        results_txt = open("results_" + filenamebase + ".txt", "a")
        results_txt.write(str(diff) + "," + str(diff_err) + "\n")
        results_txt.close()
        
        ipr = kmc_setup.get_ipr()
        ipr_txt = open("ipr_" + filenamebase + ".txt", "a")
        ipr_txt.write(str(ipr) + ",")
        ipr_txt.close()

    
    # without taking into account units:
    # print('diffusivity ', diff)
    # print('diffusivity error', diff_err)
    
    # test plot of the msds
    # plt.xlabel(r'$t$')
    # plt.ylabel('MSD')
    # plt.plot(times, msds)
    # plt.show()


if __name__ == '__main__':
    main()
