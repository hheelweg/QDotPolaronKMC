import matplotlib.pyplot as plt
from src import montecarlo as mc
from multiprocessing import Pool, cpu_count
import math
import time
import os
# track performance bottlenecks
from pyinstrument import Profiler


def main():


    ndim = 2                                    # number of dimensions
    N = 30                                      # number of QDs in each dimension
    nc_edgelength = 8                           # length of each QD (units?)
    ligand_length = 1                           # length of ligands on QD (units?)

    # seed for randomness of Hamiltonian (if None, then Hamiltonian is randomly drawn for every instance of the class)
    seed = 12345

    # Hamiltonian and bath related parameters
    reorg_nrg = 0.01                            # reorganization energy (units?)
    w_c = 0.1                                   # cutoff frequency (units?)
    J_c = 10                                    # J_c (units?)
    inhomog_sd = 0.002                          # inhomogenous broadening (units?)
    nrg_center = 2.0                            # mean site energy (units ?)
    rel_spatial_disorder = 0.0                  # relative spatial disorder
    dipolegen = 'random'                        # dipole generation procedure
    temp = 200                                  # temperature (K)
    spec_density = 'cubic-exp'                  # bath spectral density

    # PTRE and KMC related parameters
    numtrials = 1                               # number of trials to average over (here: 1)
    #method = 'first-order'                      # method for computing bath integrals 
    method = 'exact'
    r_hop = 9                                   # hopping radius (see Kassal) (in units of lattice spacing)
    r_ove = 9                                   # overlap radius (see Kassal) (in units of lattice spacing)
    r_box = math.ceil(min(r_hop, r_ove))
    
    ntrajs = 10                                 # number of trajectories to compute MSDs over
    t_final = 5                                 # final time for each trajectory (units?)
    #-------------------------------------------------------------------------

    # lattice spacing
    spacing = nc_edgelength + 2 * ligand_length

    # specify the bath
    spectrum = [spec_density, reorg_nrg, w_c, method]


    # greate instance of MC class to run KMC simulation
    print('parameter check:', ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                                J_c, spectrum, temp, ntrajs, r_hop, r_ove, r_box)
    kmc_setup = mc.KMCRunner(ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                                J_c, spectrum, temp, ntrajs, r_hop, r_ove, r_box)
    
    # perform a KMC simulation
    times, msds = kmc_setup.NEW_simulate_kmc(t_final)

    
    diff, diff_err = kmc_setup.get_diffusivity_hh(msds, times, ndim)
    
    # -------------------------------------------------------------------------
    # without taking into account units:
    print('diffusivity ', diff)
    print('diffusivity error', diff_err)
    

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
