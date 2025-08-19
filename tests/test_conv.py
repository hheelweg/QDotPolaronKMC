import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import pandas as pd
import os
import numpy as np
# track performance bottlenecks
from pyinstrument import Profiler
import qdotkmc



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
    r_hop = 8                                   # hopping radius (see Kassal) (in units of lattice spacing)
    r_ove = 8                                   # overlap radius (see Kassal) (in units of lattice spacing)
    
    ntrajs = 20                                 # number of trajectories to compute MSDs over
    nrealizations = 8                           # number of disorder realizations (i.e. number of time we initialize a new QD lattice)

    t_final = 5                                # final time for each trajectory (units?)
    #-------------------------------------------------------------------------

    # lattice spacing
    spacing = nc_edgelength + 2 * ligand_length

    # specify the bath
    spectrum = [spec_density, reorg_nrg, w_c]


    # create instance of MC class to run KMC simulation
    print('parameter summary:', ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                                J_c, spectrum, temp, ntrajs, nrealizations, r_hop, r_ove)
    
    # define dataclasses
    geom = qdotkmc.config.GeometryConfig(dims = ndim, N = N, qd_spacing = spacing, r_hop = r_hop, r_ove = r_ove)
    dis  = qdotkmc.config.DisorderConfig(nrg_center = nrg_center, inhomog_sd = inhomog_sd, relative_spatial_disorder = rel_spatial_disorder,
                          dipolegen=dipolegen, J_c = J_c, seed_base = seed)
    bath_cfg = qdotkmc.config.BathConfig(temp = temp, spectrum = spectrum)
    run  = qdotkmc.config.RunConfig(ntrajs = ntrajs, nrealizations = nrealizations, t_final = t_final, time_grid_density=200)

    
    kmc_setup = qdotkmc.montecarlo.KMCRunner(geom, dis, bath_cfg, run)
    
    # test rate convergence
    criterion_coll = kmc_setup._rate_score(no_samples = 100, criterion = "rate-displacement")
    print('collective criterion (score)', criterion_coll)

    


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