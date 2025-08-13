import numpy as np
from scipy.linalg import eigh
from .config import GeometryConfig, DisorderConfig, BathConfig, RunConfig
from numpy.random import SeedSequence, default_rng
from . import lattice
import time


class KMCRunner():
    
    
    def __init__(self, geom : GeometryConfig, dis : DisorderConfig, bath : BathConfig, run : RunConfig):

        self.geom = geom
        self.dis = dis
        self.bath = bath
        self.run = run

        # root seed sequence controls the entire experiment for reproducibility
        self._ss_root = SeedSequence(self.dis.seed_base)
    
    # make join time-grid
    def _make_time_grid(self):
        npts = int(self.run.t_final * self.run.time_grid_density)
        npts = max(npts, 2)     # ensure at least 2 points
        time_grid = np.linspace(0.0, self.run.t_final, npts)
        return time_grid
    
    # per-realization seed for each (per-QDLattice)
    def _spawn_realization_seed(self, rid : int):
        # spawn exactly nrealizations once; pick the rid-th child
        ss_real = self._ss_root.spawn(self.run.nrealizations)[rid]
        return int(ss_real.generate_state(1, dtype=np.uint64)[0])
    
    # child SeedSequences for all trajectories of a given realization (QDLattice)
    def _spawn_trajectory_seedseq(self, rid : int):
        ss_real = SeedSequence(self._spawn_realization_seed(rid))
        return ss_real.spawn(self.run.ntrajs)



    def _make_kmatrix_box(self, qd_lattice, center_global):

        # (1) use the global indices of polaron and site inside box
        pol_box  = qd_lattice.pol_idxs_last
        site_box = qd_lattice.site_idxs_last

        # (2) refine the polaron and site indices by additional constraints on r_hop and r_ove
        # NOTE : refine_by_radius function can maybe be moved into this module ? 
        pol_g, site_g = qd_lattice.redfield.refine_by_radius(
                    pol_idxs_global = pol_box,
                    site_idxs_global = site_box,
                    center_global = center_global,                      # global index of the center polaron
                    periodic=True,                                      # or False to match array setup
                    grid_dims=qd_lattice.geom.lattice_dimension
                    )

        # (2) compute rates on those exact indices (no re-derivation)
        rates, final_states, tot_time = qd_lattice.redfield.make_redfield_box(
            pol_idxs_global=pol_g, site_idxs_global=site_g, center_global=center_global
        )

        # (3) cache by global center index
        overall_idx_start = center_global
        qd_lattice.stored_npolarons_box[overall_idx_start] = len(pol_g)
        qd_lattice.stored_polaron_sites[overall_idx_start] = np.copy(final_states)   # global indices
        qd_lattice.stored_rate_vectors[overall_idx_start]  = np.copy(rates)

        return rates, final_states, tot_time

    # make box around center position where we are currently at
    # TODO : incorporate periodic boundary conditions explicty (boolean)
    def _get_box(self, qd_lattice, center, periodic=True):

        # (1) box size (unchanged)
        qd_lattice.box_size = qd_lattice.box_length * qd_lattice.geom.qd_spacing

        # (2) helpers (unchanged logic)
        # NOTE : put this somewhere as a helper function ?
        def find_indices_within_box(points, center, grid_dims, box_size, periodic=True):
            half_box = box_size / 2
            dim = len(center)
            assert len(points[0]) == len(center) == len(grid_dims)

            if dim == 1:
                dx = np.abs(points - center[0])
                if periodic:
                    dx = np.minimum(dx, grid_dims[0] - dx)
                return np.where(dx <= half_box)[0]

            elif dim == 2:
                dx = np.abs(points[:, 0] - center[0])
                dy = np.abs(points[:, 1] - center[1])
                if periodic:
                    dx = np.minimum(dx, grid_dims[0] - dx)
                    dy = np.minimum(dy, grid_dims[1] - dy)
                return np.where((dx <= half_box) & (dy <= half_box))[0]

            else:
                raise NotImplementedError("find_indices_within_box: only 1D/2D supported.")

        # (3) global index sets inside the axis-aligned periodic box
        pol_idxs = find_indices_within_box(qd_lattice.polaron_locs, center, qd_lattice.geom.lattice_dimension, qd_lattice.box_size, periodic)
        site_idxs = find_indices_within_box(qd_lattice.qd_locations,  center, qd_lattice.geom.lattice_dimension, qd_lattice.box_size, periodic)

        # keep order, store contiguously
        qd_lattice.pol_idxs_last  = np.ascontiguousarray(pol_idxs.astype(np.intp))
        qd_lattice.site_idxs_last = np.ascontiguousarray(site_idxs.astype(np.intp))

        # (4) define the GLOBAL center index once
        qd_lattice.center_global = int(self.get_closest_idx(qd_lattice, center, qd_lattice.polaron_locs))

        # (5) optional: local position of the center inside the box (rarely needed now)
        where = np.nonzero(qd_lattice.pol_idxs_last == qd_lattice.center_global)[0]
        # If the box is tight or discrete, it should be present; if not, refine_by_radius will handle it.
        qd_lattice.center_local = int(where[0]) if where.size == 1 else None


    def _make_kmc_step(self, qd_lattice, polaron_start_site, rnd_generator = None):

        # (0) check whether we have a valid instance of QDLattice class
        assert isinstance(qd_lattice, lattice.QDLattice), "need to feed valid QDLattice instance!"

        # (1) build box (just indices + center_global)
        self._get_box(qd_lattice, polaron_start_site)

        center_global = qd_lattice.center_global
        start_pol = qd_lattice.polaron_locs[center_global]

        # (2) compute (or reuse) rates
        if qd_lattice.stored_npolarons_box[center_global] == 0:
            rates, final_states, tot_time = self._make_kmatrix_box(qd_lattice, center_global)
        else:
            tot_time = 0.0
            final_states = qd_lattice.stored_polaron_sites[center_global]  # global indices
            rates        = qd_lattice.stored_rate_vectors[center_global]

        # (3) rejection-free KMC step
        cum_rates = np.cumsum(rates)
        S = cum_rates[-1]

        # two random numbers for rejection-free KMC
        u1 = np.random.uniform() if rnd_generator is None else rnd_generator.uniform()
        u2 = np.random.uniform() if rnd_generator is None else rnd_generator.uniform()

        final_idx = int(np.searchsorted(cum_rates, u1 * S))
        self.time += -np.log(u2) / S

        # (4) final polaron coordinate in GLOBAL frame
        end_pol = qd_lattice.polaron_locs[final_states[final_idx]]

        return start_pol, end_pol, tot_time
    

    # build realization of QD lattice
    def _build_grid_realization(self, rid : int):

        # get random seef from realization id (rid)
        rnd_seed = self._spawn_realization_seed(rid)
        print('seed realization', rnd_seed)
        
        # initialize instance of QDLattice class
        # NOTE : change to rnd_seed = self.dis.seed_base for default seed
        #qd = lattice.QDLattice(geom=self.geom, dis=self.dis, bath=self.bath, seed_realization=self.dis.seed_base)
        qd = lattice.QDLattice(geom=self.geom, dis=self.dis, bath=self.bath, seed_realization=rnd_seed)

        # setup QDLattice with (polaron-transformed) Hamiltonian, bath information, Redfield
        # NOTE : we currenly feed the bath information here as well
        qd._setup(self.bath.temp, self.bath.spectrum)

        return qd
    
    
    def simulate_kmc(self, t_final):

        times_msds = self._make_time_grid()                                 # time ranges to use for computation of msds                                                                 
        msds = np.zeros((self.run.nrealizations, len(times_msds)))          # initialize MSD output
        self.simulated_time = 0
        
        # loop over realization
        for r in range(self.run.nrealizations):

            # build QD lattice realization
            qd_lattice = self._build_grid_realization(rid=r)

            # get trajectory seed sequence
            traj_ss = self._spawn_trajectory_seedseq(rid=r)

            # loop over number of trajectories per realization
            for t in range(self.run.ntrajs):
                    
                self.time = 0                                           # reset clock for each trajectory
                self.step_counter = 0                                   # keeping track of the # of KMC steps
                self.trajectory_start = np.zeros(self.geom.dims)        # initialize trajectory start point
                self.trajectory_current = np.zeros(self.geom.dims)      # initialize current trajectopry state
                self.sds = np.zeros_like(times_msds)                    # store sq displacements on times_msd time grid
                
                comp_time = time.time()
                time_idx = 0
                sq_displacement = 0

                # random generator for trajectory
                rng_traj = default_rng(traj_ss[t])
                
                while self.time < t_final:

                    # (1) determine what polaron site we are at currently
                    if self.step_counter == 0:
                        # draw initial center of the box (here: 'uniform') in the exciton site basis
                        # TODO : might want to add other initializations
                        start_site = qd_lattice.qd_locations[rng_traj.integers(low=0, high=self.geom.n_sites-1)]
                        start_pol = qd_lattice.polaron_locs[self.get_closest_idx(qd_lattice, start_site, qd_lattice.polaron_locs)]

                    else:
                        # start_site is final_site from previous step
                        start_pol = end_pol
                
                    # (2) perform KMC step and obtain coordinates of polaron at beginning (start_pol) and end (end_pol) of the step
                    start_pol, end_pol, tot_time = self._make_kmc_step(qd_lattice, start_pol, rnd_generator = rng_traj)
                    self.simulated_time += tot_time
                    
                    # (3) update trajectory and compute squared displacements 
                    if self.step_counter == 0:
                        self.trajectory_start = start_pol
                        self.trajectory_current = start_pol
                    if self.time < t_final:
                        # get current location in trajectory and compute squared displacement
                        self.trajectory_current = self.trajectory_current + end_pol - start_pol
                        sq_displacement = np.linalg.norm(self.trajectory_current-self.trajectory_start)**2 
                    
                        # add squared displacement at correct position in times_msd grid
                        time_idx += np.searchsorted(times_msds[time_idx:], self.time)
                        self.sds[time_idx:] = sq_displacement
                    
                    self.step_counter += 1 # update step counter
                    
                # compute mean squared displacement as a running average instead of storing all displacement vectors
                msds[r] = t/(t+1)*msds[r] + 1/(t+1)*self.sds
            
            print('----------------------------------')
            print('---- SIMULATED TIME SUMMARY -----')
            print(f'total simulated time {self.simulated_time:.3f}')
            print('----------------------------------')
        return times_msds, msds[0]

    
    
    # (08/09/2025) more efficient version
    def get_closest_idx(self, qd_lattice, pos, array):
        """
        Find the index in `array` closest to `pos` under periodic boundary conditions.
        """
        # Vectorized periodic displacement
        delta = array - pos  # shape (N, dims)

        # Apply periodic boundary condition (minimum image convention)
        delta -= np.round(delta / qd_lattice.geom.boundary) * qd_lattice.geom.boundary

        # Compute squared distances
        dists_squared = np.sum(delta**2, axis=1)

        return np.argmin(dists_squared)
    



    def get_diffusivity_hh(self, msds, times, dims):
        # note : I here assume that the whole time arrange is approx. linear (might break down)
        fit_params, cov = np.polyfit(times, msds, 1, cov=True)
        diff = fit_params[0]/(2*dims)
        # obtain error on diffusvity as from error on slope parameter 
        diff_err = np.sqrt(np.diag(cov))[0]/(2*dims)
        return diff, diff_err
    



    # ---------------------------------------------------------------------------------------------------------
    # HH : for some arrays of r_hop and r_ove, comopute the rates and check
    # for convergence (you are more than invited to play around with this!)
    def get_rate_convergence(self, r_hops, r_oves):
        # determine convergence of rates
        rateConvergence = np.zeros((len(r_hops), len(r_oves)))
        boxLengths = np.zeros((len(r_hops), len(r_oves)))
        # more microscopoic rate convergence information
        rateMeanFinal = np.zeros((len(r_hops), len(r_oves), self.dims))
        rateThreeBiggest = np.zeros((len(r_hops), len(r_oves), 3))
        rateCumThree = np.zeros((len(r_hops), len(r_oves)))

        # consider outgoing rates from the most central site:
        start_site = np.array([int(self.sidelength // 2) * self.qd_spacing, int(self.sidelength // 2) * self.qd_spacing])

        # loop over all combinations of r_ove and r_hop 
        for i, r_hop in enumerate(r_hops):
            for j, r_ove in enumerate(r_oves):

                # (0) get box properties
                self.make_box_dimensions(r_hop, r_ove)
                boxLengths[i, j] = self.box_length

                # (1) get rates
                # (1.1) create box
                self.get_box(start_site)
                # (1.2) get initial polaron state (polaron basis)
                idx_start = self.get_closest_idx(start_site, self.eigstates_locs)
                self.start_pol = self.eigstates_locs[idx_start]
                # (1.3) obtain rates
                self.make_kmatrix_box(idx_start)

                # (2) analyze those rates based on desired criteria
                # (2.1) cumulative rates
                cum_rates = np.sum(self.rates)
                # (2.2) more microscopic details on the rates
                mean_final, three_biggest, cum_three_biggest_norm  = self.analyze_rates()

                # (3) store
                rateConvergence[i, j] = cum_rates
                rateMeanFinal[i, j, :] = mean_final
                #rateThreeBiggest[i, j, :] = three_biggest
                #rateCumThree[i, j] = cum_three_biggest_norm

        return rateConvergence, boxLengths, rateMeanFinal, rateThreeBiggest, rateCumThree
    
    # HH : this gives some more microscopic insights into self.rates
    def analyze_rates(self):
        # normalize rates array
        rates_norm = self.rates / np.sum(self.rates)
        # find three biggest entries and also the percentage of rates
        # (1) get mean final state difference from original state
        locs = self.eigstates_locs[self.final_states]
        mean_final = np.matmul(locs.T, rates_norm) - self.start_pol
        # (2) get three biggest (normalized) rates
        rates_norm.sort()
        three_biggest = rates_norm[-3:][::-1]
        cum_three_biggest_norm = np.sum(three_biggest)
        return mean_final, three_biggest, cum_three_biggest_norm   
    
    # NOTE : move to utils.py ?
    def get_ipr(self):
        # returns ipr of one column vector, or mean ipr of multiple column vectors
        return np.mean(1/np.sum(self.eigstates ** 4, axis = 0))
    




