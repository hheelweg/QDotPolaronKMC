# %%
import numpy as np
import src.montecarlo as mc
import src.utils as utils
import math
import matplotlib.pyplot as plt
import time
# %% [markdown]
# ### Test Script for getting the PTRE KMC calculations going
# this script just contains some tests. we use the following sample parameters
# %%
ndim = 1                                   # number of dimensions
N = 8                                      # number of QDs in each dimension
nc_edgelength = 8                           # length of each QD (units?)
ligand_length = 1                           # length of ligands on QD (units?)

# seed for randomness of Hamiltonian (if None, then Hamiltonian is randomly drawn for every instance of the class)
seed = None

# Hamiltonian and bath related parameters
reorg_nrg = 0.01                            # reorganization energy (units?)
w_c = 0.1                                   # cutoff frequency (units?)
J_c = 1                                    # J_c (units?)
inhomog_sd = 0.2                          # inhomogenous broadening (units?)
nrg_center = 2.0                            # mean site energy (units ?)
rel_spatial_disorder = 0.0                  # relative spatial disorder
dipolegen = 'random'                        # dipole generation procedure
temp = 200                                  # temperature (K)
spec_density = 'cubic-exp'                  # bath spectral density

# PTRE and KMC related parameters
numtrials = 1                               # number of trials to average over (here: 1)
method = 'cheby-fit'                      # method for computing bath integrals 
r_hop = 3                                  # hopping radius (see Kassal) (in units of lattice spacing)
r_ove = 3                                   # overlap radius (see Kassal) (in units of lattice spacing)
r_box = math.ceil(min(r_hop, r_ove))        

ntrajs = 10                                 # number of trajectories to compute MSDs over
t_final = 1                                 # final time for each trajectory (units?)

# lattice spacing
spacing = nc_edgelength + 2 * ligand_length

# specify the bath
spectrum = [spec_density, reorg_nrg, w_c, method]

# %% [markdown]
# This is me playin around with the interplay between $J_c$ and `inhomog_sd` and looking at the effect it has on the location of the
# polaron eigenstates. It looks like the bigger $J_c$ gets as compared to `inhomog_sd`, the more clamped the polaron states become in
# the middle of the lattice.
# %%

J_c = 10
inhomog_sd = 0.02
ndim = 1
N = 100

# greate instance of MC class to run KMC simulation
kmc_setup = mc.KMCRunner(ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                            J_c, spectrum, temp, ntrajs, r_hop, r_ove, r_box)


# get polaron states
polaron_states = kmc_setup.polaron_locs

# plot polaron states as compared to grid points
utils.plot_lattice(polaron_states ,kmc_setup.qd_locations, label = 'polaron states', periodic = True)
plt.legend()
plt.show()
# %% [markdown]
# Test function that makes box around specific center point on the lattice. This just serves as a sanity check to see
# whether we have implemented the correct function to make a box
# %%
J_c = 2
inhomog_sd = 0.02
ndim = 1
N = 10

# greate instance of MC class to run KMC simulation
kmc_setup = mc.KMCRunner(ndim, N, spacing, nrg_center, inhomog_sd, dipolegen, seed, rel_spatial_disorder,
                            J_c, spectrum, temp, ntrajs, r_hop, r_ove, r_box)

ntrajs = 10                                 # number of trajectories to compute MSDs over
t_final = 10                               # final time for each trajectory (units?)
run_time= time.time()
times, msds = kmc_setup.NEW_simulate_kmc(t_final)
run_time = time.time()-run_time
print(run_time)

diff, diff_err = kmc_setup.get_diffusivity_hh(msds, times, ndim)

# without taking into account units:
print('diffusivity ', diff)
print('diffusivity error', diff_err)

# test plot of the msds
plt.xlabel(r'$t$')
plt.ylabel('MSD')
plt.plot(times, msds)
plt.show()

# get polaron states
polaron_states = kmc_setup.polaron_locs

# plot polaron states as compared to grid points
utils.plot_lattice(polaron_states ,kmc_setup.qd_locations, label = 'polaron states', periodic = True)
plt.legend()
plt.show()

center_test = [5, 5]

# get polaron locations (abs. and rel)
kmc_setup.NEW_get_box(center_test)

selection_abs = kmc_setup.eigstates_locs_abs
selection_rel = kmc_setup.eigstates_locs

# get polaron states locations
polaron_states = kmc_setup.polaron_locs

# plot polaron states as compared to grid points
utils.plot_lattice(polaron_states ,kmc_setup.qd_locations, label = 'polaron states', periodic = True)
plt.scatter(center_test[0], color ='red', s =20, label = 'center')
# plot absolute and relative positions to center of polarons in box 
plt.scatter(selection_abs.T[0], selection_abs.T[1], color = 'blue', s = 20, alpha = 0.2, label = 'box (abs.)')
plt.scatter(selection_rel.T[0] + center_test[0], selection_rel.T[1] + center_test[1], color = 'green', s = 20, alpha = 0.2, label = 'box (rel.)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=True)
plt.show()