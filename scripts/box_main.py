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
import math


def main():

    ndim = 1                                    # number of dimensions
    N = 100                                      # number of QDs in each dimension
    nc_edgelength = 8                           # length of each QD (units?)
    ligand_length = 1                           # length of ligands on QD (units?)

    # seed for randomness of Hamiltonian (if None, then Hamiltonian is randomly drawn for every instance of the class)
    seed = None

    # Hamiltonian and bath related parameters
    reorg_nrg = 0.01                            # reorganization energy (units?)
    w_c = 0.1                                   # cutoff frequency (units?)
    J_c = 10                                    # J_c (units?)
    inhomog_sd = 0.002                           # inhomogenous broadening (units?)
    nrg_center = 2.0                            # mean site energy (units ?)
    rel_spatial_disorder = 0.0                  # relative spatial disorder
    dipolegen = 'random'                        # dipole generation procedure
    temp = 200                                  # temperature (K)
    spec_density = 'cubic-exp'                  # bath spectral density

    # PTRE and KMC related parameters
    numtrials = 1                               # number of trials to average over (here: 1)
    method = 'first-order'                      # method for computing bath integrals 
    r_hop = 3                                   # hopping radius (see Kassal) (in units of lattice spacing)
    r_ove = 3.5                                 # overlap radius (see Kassal) (in units of lattice spacing)
    r_box = math.ceil(min(r_hop, r_ove))
    
    ntrajs = 1000                                 # number of trajectories to compute MSDs over
    t_final = 1                               # final time for each trajectory (units?)
    #-------------------------------------------------------------------------

    # lattice spacing
    spacing = nc_edgelength + 2 * ligand_length

    # specify the bath
    spectrum = [spec_density, reorg_nrg, w_c, method]


    # greate instance of MC class to run KMC simulation
    kmc_setup = mc.kmc_runner(ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                                J_c, spectrum, temp, ntrajs, r_hop, r_ove, r_box)
    
    # perform a KMC simulation
    times, msds = kmc_setup.simulate_kmc(t_final)
    
    diff, diff_err = kmc_setup.get_diffusivity_hh(msds, times, ndim)
    
    # without taking into account units:
    print('diffusivity ', diff)
    print('diffusivity error', diff_err)
    
    # test plot of the msds
    plt.xlabel(r'$t$')
    plt.ylabel('MSD')
    plt.plot(times, msds)
    plt.show()


if __name__ == '__main__':
    main()
