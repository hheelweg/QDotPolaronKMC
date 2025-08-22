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
    N = 20                                      # number of QDs in each dimension
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
    
    ntrajs = 10                                 # number of trajectories to compute MSDs over
    nrealizations = 8                           # number of disorder realizations (i.e. number of time we initialize a new QD lattice)

    t_final = 5                                # final time for each trajectory (units?)
    #-------------------------------------------------------------------------

    # lattice spacing
    spacing = nc_edgelength + 2 * ligand_length

    # specify the bath
    spectrum = [spec_density, reorg_nrg, w_c]

    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
    print('max_workers', max_workers)


    # create instance of MC class to run KMC simulation
    print('parameter summary:', ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                                J_c, spectrum, temp, ntrajs, nrealizations, r_hop, r_ove, max_workers)
    
    # define dataclasses
    geom = qdotkmc.config.GeometryConfig(dims = ndim, N = N, qd_spacing = spacing)
    dis  = qdotkmc.config.DisorderConfig(nrg_center = nrg_center, inhomog_sd = inhomog_sd, relative_spatial_disorder = rel_spatial_disorder,
                          dipolegen=dipolegen, J_c = J_c, seed_base = seed)
    bath_cfg = qdotkmc.config.BathConfig(temp = temp, spectrum = spectrum)
    run  = qdotkmc.config.RunConfig(ntrajs = ntrajs, nrealizations = nrealizations, t_final = t_final, time_grid_density=200,
                                    r_hop = r_hop, r_ove = r_ove, 
                                    max_workers = max_workers)
    
    
    kmc = qdotkmc.montecarlo.KMCRunner(geom, dis, bath_cfg, run)
    print('inital', kmc.run.r_hop, kmc.run.r_ove)

    kmc.run.r_hop = 7
    kmc.run.r_ove = 7
    
    # perform KMC simulation (parallel)
    #times, msds = kmc_setup.simulate_kmc_parallel()
    times, msds = kmc._simulate_kmc()

    print('inital', kmc.run.r_hop, kmc.run.r_ove)

    # perform KMC simulation (serial)
    #times, msds = kmc_setup.simulate_kmc_serial()

    # export msds as .csv file for inspection
    qdotkmc.utils.export_msds(times, msds)


    # get noise-averaged (pooled) trajectory MSD
    msds_mean = np.mean(msds, axis = 0)


    diff1, sigma_D1 = qdotkmc.utils.get_diffusivity(msds_mean, times, ndim)

    diff2, sigma_D2 = qdotkmc.utils.summarize_diffusivity(msds, times, ndim)
    
    # -------------------------------------------------------------------------
    # without taking into account units:
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
