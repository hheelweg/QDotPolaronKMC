import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  
import time
import scipy.stats as st
import scipy.integrate as integrate
import sys
import scipy
import redfield_box, hamiltonian_box
import utils
import itertools
import montecarlo as mc
from multiprocessing import Pool, cpu_count
import os
import const

def main():
    ndims = np.array([1, 2]).astype(int)
    J_cs = np.array([3, 30])
    inhomog_sds = np.array([0.001, 0.01])
    temps = np.linspace(1, 300, 10)
    r_hops = np.array([2,3,4,5]).astype(int)
    r_oves = np.array([2.5, 3.5, 4.5, 5.5, 6.5])
    
    
    
    # reorg_nrgs = np.array([0.01, 0.03, 0.1])
    # w_cs = np.array([0.05, 0.1])
    # J_cs = np.array([10, 100, 1000])
    # inhomog_sds = np.array([0.001, 0.01, 0.1])
    
    # number of cores you have allocated for your SLURM task:
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    
    #number of parallel runs
    numPara=number_of_cores
    #-------------------------------------------------------------------------
    args=[]
    for h in range(len(ndims)):
        for i in range(len(J_cs)):
            for j in range(len(inhomog_sds)):
                for k in range(len(temps)):
                    for l in range(len(r_hops)):
                        for m in range(len(r_oves)):
                            params = [ndims[h], J_cs[i], inhomog_sds[j], temps[k], r_hops[l], r_oves[m]]
                            args.append((params, 1))
    
    with Pool(numPara) as pool:
        results = pool.starmap(run_sim, args)
    # print(results)


def run_sim(args, dummy = 1):
    ndim = args[0]
    J_c = args[1]
    inhomog_sd = args[2]
    temp = args[3]
    r_hop = args[4]
    r_ove = args[5]
    
    
    # ndim = 1                                    # number of dimensions
    N = 40                                      # number of QDs in each dimension
    nc_edgelength = 8                           # length of each QD (units?)
    ligand_length = 1                           # length of ligands on QD (units?)

    # seed for randomness of Hamiltonian (if None, then Hamiltonian is randomly drawn for every instance of the class)
    seed = None

    # Hamiltonian and bath related parameters
    reorg_nrg = 0.01                            # reorganization energy (units?)
    w_c = 0.1                                   # cutoff frequency (units?)
    # J_c = 30                                    # J_c (units?)
    # inhomog_sd = 0.002                           # inhomogenous broadening (units?)
    nrg_center = 2.0                            # mean site energy (units ?)
    rel_spatial_disorder = 0.0                  # relative spatial disorder
    dipolegen = 'random'                        # dipole generation procedure
    # temp = 200                                  # temperature (K)
    spec_density = 'cubic-exp'                  # bath spectral density

    # PTRE and KMC related parameters
    numtrials = 1                               # number of trials to average over (here: 3)
    method = 'first-order'                      # method for computing bath integrals 
    # r_hop = 1                                   # hopping radius (see Kassal) (in units of lattice spacing)
    # r_ove = 1                                   # overlap radius (see Kassal) (in units of lattice spacing)

    ntrajs = 1000                                 # number of trajectories to compute MSDs over
    t_final = 3                                   # final time for each trajectory (units?)
    #-------------------------------------------------------------------------

    # lattice spacing
    spacing = nc_edgelength + 2 * ligand_length

    # specify the bath
    spectrum = [spec_density, reorg_nrg, w_c, method]


    # greate instance of MC class to run KMC simulation
    kmc_setup = mc.kmc_runner(ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                                J_c, spectrum, temp, ntrajs, r_hop, r_ove)
    
    # perform a KMC simulation
    times, msds = kmc_setup.NEW_simulate_kmc(t_final)
    
    diff, diff_err = kmc_setup.get_diffusivity_hh(msds, times, ndim)
    
    
    filenamebase = ("dim_%d,J_c_%f,sigma_i_%f,temp_%f,r_hop_%f,r_ove_%f")%(ndim, J_c, inhomog_sd, temp, r_hop, r_ove)
    results_txt = open("results_" + filenamebase + ".txt", "w")
    results_txt.write(str(diff) + "," + str(diff_err))
    results_txt.close()

    
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
