import numpy as np
from .config import GeometryConfig, DisorderConfig, BathConfig, RunConfig
from numpy.random import SeedSequence, default_rng
from . import lattice, hamiltonian_box, const
from .hamiltonian_box import SpecDens

# global variable to allow parallel workers to use the same bath setup
_BATH_GLOBAL = None

# --- top-level worker so it's picklable by ProcessPool ---
def _one_lattice_worker(args):
    (geom, dis, bath_cfg, run, t_final, times_msds, rid, sim_time) = args
    runner = KMCRunner(geom, dis, bath_cfg, run)
    return rid, *runner._run_single_lattice(ntrajs = run.ntrajs,
                                            bath = _BATH_GLOBAL,
                                            t_final = run.t_final,
                                            times = times_msds,
                                            realization_id=rid,
                                            simulated_time=sim_time)


class KMCRunner():
    
    
    def __init__(self, geom : GeometryConfig, dis : DisorderConfig, bath_cfg : BathConfig, run : RunConfig):

        self.geom = geom                        # geometry for QDLattice
        self.dis = dis                          # energetic parameters for QDLattice
        self.bath_cfg = bath_cfg                # bath setup, temperature ...
        self.run = run                          # KMC related parameters

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
    

    def _build_rate_convergenc_env(self, no_samples):

        # (0) build bath
        bath = SpecDens(self.bath_cfg.spectrum, const.kB * self.bath_cfg.temp)

        # (1) draw a lattice realization
        qd_lattice = self._build_grid_realization(bath, rid=0)

        # (2) produce no_samples starting indices from where to compute rate vectors
        ss_conv = self._ss_root.spawn(1)[0]
        rng_conv = default_rng(ss_conv)
        start_sites = rng_conv.integers(0, qd_lattice.geom.n_sites, size=no_samples)

        # (3) get Boltzmann weights for each start polaron in start_sites
        E = qd_lattice.full_ham.evals
        beta = qd_lattice.full_ham.beta
        w = np.exp(- beta * E[start_sites])
        Z = np.sum(np.exp(- beta * E))
        weights = w / Z

        return qd_lattice, start_sites, weights


    # TODO : write input parameters so that we can also use this for r_hop/r_ove
    def _rate_score(self, theta_pol, theta_sites, 
                    qd_lattice, start_sites, weights,
                    criterion=None, score_info=False,
                    ):

        # get rates starting from each polaron starting index and analyze by criterion
        rates_criterion = None
        nsites_sel, npols_sel = 0, 0
        for start_idx in start_sites:

            # NEW way to obtain rates from theta_pol/theta_sites
            rates, final_sites, _, sel_info = self._make_kmatrix_boxNEW(qd_lattice, start_idx,
                                                                        theta_sites=theta_sites,
                                                                        theta_pol=theta_pol,
                                                                        selection_info = True
                                                                        )
            
            # how many polarons/sites were selected 
            nsites_sel += sel_info['nsites_sel']
            npols_sel += sel_info['npols_sel']
            
            # evaluate convergence criterion on rates vector
            if criterion == "rate-displacement":
                start_loc = qd_lattice.qd_locations[start_idx]                                                      # r(0)
                sq_displacments = ((qd_lattice.qd_locations[final_sites] - start_loc)**2).sum(axis = 1)             # ||Δr||^2 per destination
                lamda = (rates * sq_displacments).sum() / (2 * qd_lattice.geom.dims)
                rates_criterion = (weights * lamda).sum()
            else:
                raise ValueError("please specify valid convergence criterion for rates!")
            

        # optional : store additional information
        info = {}
        if score_info:
            info['ave_sites'] = nsites_sel / len(start_sites)
            info['ave_pols'] = npols_sel / len(start_sites)

        return rates_criterion, info

    # rate computation based on r_hop/r_ove
    def _make_kmatrix_box(self, qd_lattice, center_global, r_hop, r_ove):

        # (0) set up r_hop and r_ove, intialize box in qd_lattice as well
        qd_lattice.redfield.r_hop, qd_lattice.redfield.r_ove = r_hop * qd_lattice.geom.qd_spacing, r_ove * qd_lattice.geom.qd_spacing

        # (1) use legacy self._get_box() and initialize box 
        # NOTE : this can likely be deleted
        qd_lattice._init_box_dims(r_hop, r_ove)
        polaron_start_site = qd_lattice.polaron_locs[center_global]
        self._get_box(qd_lattice, polaron_start_site)

        # (2) use the global indices of polaron and site inside box
        pol_box  = qd_lattice.pol_idxs_last
        site_box = qd_lattice.site_idxs_last

        # (3) refine the polaron and site indices by additional constraints on r_hop and r_ove
        # NOTE : refine_by_radius function can maybe be moved into this module ? 
        pol_g, site_g = qd_lattice.redfield.refine_by_radius(
                    pol_idxs_global = pol_box,
                    site_idxs_global = site_box,
                    center_global = center_global,                      # global index of the center polaron
                    periodic=True,                                      # or False to match array setup
                    grid_dims=qd_lattice.geom.lattice_dimension
                    )

        # (4) compute rates on those exact indices (no re-derivation)
        rates, final_states, tot_time = qd_lattice.redfield.make_redfield_box(
            pol_idxs_global=pol_g, site_idxs_global=site_g, center_global=center_global
        )

        # (5) cache by global center index
        overall_idx_start = center_global
        qd_lattice.stored_npolarons_box[overall_idx_start] = len(pol_g)
        qd_lattice.stored_polaron_sites[overall_idx_start] = np.copy(final_states)   # global indices
        qd_lattice.stored_rate_vectors[overall_idx_start]  = np.copy(rates)

        return rates, final_states, tot_time

    # make box around center position where we are currently at
    # TODO : incorporate periodic boundary conditions explicty (boolean)
    # NOTE : this can likely be deleted
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
        # NOTE : it seems like eventuall this is all we need, so we might get rid of this function alltogether ??
        qd_lattice.center_global = int(self.get_closest_idx(qd_lattice, center, qd_lattice.polaron_locs))

        # (5) optional: local position of the center inside the box (rarely needed now)
        where = np.nonzero(qd_lattice.pol_idxs_last == qd_lattice.center_global)[0]
        # If the box is tight or discrete, it should be present; if not, refine_by_radius will handle it.
        qd_lattice.center_local = int(where[0]) if where.size == 1 else None


    # rate computation based on theta_pol/theta_tau
    def _make_kmatrix_boxNEW(self, qd_lattice, center_global, theta_pol, theta_sites, selection_info = False):

        # (0) set up θ_pol and θ_sites 
        qd_lattice.redfield.theta_pol, qd_lattice.redfield.theta_sites = theta_pol, theta_sites

        # (1) select sites and polarons that ''matter'' for computing the rates
        site_g, pol_g = qd_lattice.redfield.select_sites_and_polarons(
                                                                      center_global=center_global, 
                                                                      theta_sites= qd_lattice.redfield.theta_sites, 
                                                                      theta_pol = qd_lattice.redfield.theta_pol, 
                                                                      verbose=False
                                                                      )

        # (2) compute rates on those exact indices (no re-derivation)
        rates, final_states, tot_time = qd_lattice.redfield.make_redfield_box(
            pol_idxs_global=pol_g, site_idxs_global=site_g, center_global=center_global,
            verbosity = False
        )

        # (3) cache by global center index
        qd_lattice.stored_npolarons_box[center_global] = len(pol_g)
        qd_lattice.stored_polaron_sites[center_global] = np.copy(final_states)   
        qd_lattice.stored_rate_vectors[center_global]  = np.copy(rates)

        # (4) optional: export information about selected polarons/sites for computation of rates
        sel_info= {}
        if selection_info:
            sel_info['npols_sel'] = len(pol_g)
            sel_info['nsites_sel'] = len(site_g)

        return rates, final_states, tot_time, sel_info


    def _make_kmc_step(self, qd_lattice, clock, polaron_start_site, rnd_generator = None):

        # (0) check whether we have a valid instance of QDLattice class
        assert isinstance(qd_lattice, lattice.QDLattice), "need to feed valid QDLattice instance!"

        # (1) start polarong (global) idx and location 
        center_global = self.get_closest_idx(qd_lattice, polaron_start_site, qd_lattice.polaron_locs)
        start_pol = qd_lattice.polaron_locs[center_global]

        # (2) compute (or reuse) rates
        if qd_lattice.stored_npolarons_box[center_global] == 0:
            # (a) compute rates from r_hop/r_ove (OLD) 
            rates, final_states, tot_time = self._make_kmatrix_box(qd_lattice, 
                                                                   center_global, 
                                                                   r_hop=self.geom.r_hop, 
                                                                   r_ove=self.geom.r_ove)
            # (b) compute rates from theta_pol/theta_sites (NEW)
            # NOTE : need to update arguments
            #rates, final_states, tot_time, _ = self._make_kmatrix_boxNEW(qd_lattice, center_global)
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
        # update clock
        clock += -np.log(u2) / S

        # (4) final polaron position
        end_pol = qd_lattice.polaron_locs[final_states[final_idx]]

        return start_pol, end_pol, clock, tot_time
    

    # build realization of QD lattice
    def _build_grid_realization(self, bath : SpecDens, rid : int):

        assert isinstance(bath, SpecDens), "Need to make sure we have a proper \
                                            bath set up to build QDLattice instance"

        # get random seef from realization id (rid)
        rnd_seed = self._spawn_realization_seed(rid)
        
        # initialize instance of QDLattice class
        # NOTE : change to rnd_seed = self.dis.seed_base for default seed
        qd = lattice.QDLattice(geom=self.geom, dis=self.dis, seed_realization=rnd_seed)

        # setup QDLattice with (polaron-transformed) Hamiltonian, bath information, Redfield
        qd._setup(bath)

        return qd
    
    
    # NOTE : if periodic = True, this incorporates periodic boundary conditions
    def _update_displacement_minimage(self, trajectory_curr, trajectory_start,
                                      start_pol, end_pol,
                                      box_lengths,
                                      periodic=None
                                      ):
        """
        Pper-step update for MSD under periodic boundary conditions.

        Inputs
        ------
        trajectory_curr : (D,) array
            The current (unwrapped) position accumulator R(t) in D dimensions.
            This is NOT a wrapped coordinate; it's the sum of all previous
            displacements.
        trajectory_start : (D,) array
            The reference position R(0) at the start of the trajectory.
        start_pol, end_pol : (D,) arrays
            Polaron coordinates before and after the current KMC hop. These should
            be in the SAME coordinate system as the box (e.g., in [0, L_d) per dim).
        box_lengths : float or (D,) array
            Box side lengths [Lx, Ly, ...]. If scalar, it is broadcast to all dims.
        periodic : None or (D,) bool array
            Mask indicating which dimensions are periodic. If None → all periodic.

        Returns
        -------
        new_current : (D,) array
            Updated unwrapped accumulator R(t + Δt) = R(t) + Δr.
        sq_displacement : float
            || R(t + Δt) - R(0) ||^2, i.e., squared net displacement
            from the start, in unwrapped space.
        """
        # ensure 1D vectors
        start_pol = np.atleast_1d(np.asarray(start_pol, dtype=float))
        end_pol   = np.atleast_1d(np.asarray(end_pol, dtype=float))
        curr      = np.atleast_1d(np.asarray(trajectory_curr, dtype=float))
        start0    = np.atleast_1d(np.asarray(trajectory_start, dtype=float))

        # raw hop vector in box coordinates
        delta = end_pol - start_pol

        # box lengths per dimension (broadcast a scalar if needed)
        L = np.atleast_1d(np.asarray(box_lengths, dtype=float))

        # set periodic axes
        if periodic is None:
            periodic = np.ones(L.shape, dtype=bool)
        else:
            periodic = np.asarray(periodic, dtype=bool)

        # for each periodic dimension d: Δ_d ← Δ_d − L_d * round(Δ_d / L_d)
        # this maps any hop across a boundary to the nearest periodic image
        if np.any(periodic):
            delta_p = delta[periodic]
            L_p = L[periodic]
            delta[periodic] = delta_p - L_p * np.round(delta_p / L_p)

        # accumulate the unwrapped displacement R(t) ← R(t) + Δr
        new_curr = curr + delta
        # squared net displacement from the start (unwrapped)
        diff = new_curr - start0
        return new_curr, float(np.dot(diff, diff))


    # run a single trajectory for a specified QDLattice
    def _run_single_kmc_trajectory(self, qd_lattice, t_final, rng = None):

        # (0) time grid and per-trajectory buffers for squared displacements
        times_msds = self._make_time_grid()
        sds = np.zeros_like(times_msds, dtype=float)

        # (1) local per-trajectory state
        clock = 0.0                             # set clock to 0
        step_counter = 0                        # counter of KMC steps
        time_idx = 0
        last_r2 = 0.0                           # last know squared displacement
        tot_comp_time = 0.0                     # NOTE : this is only for debugging and tracking computational time


        # (2) draw initial center uniformly in site basis and map to nearest polaron
        idx0 = (np.random.randint(0, qd_lattice.geom.n_sites) if rng is None
                else rng.integers(0, qd_lattice.geom.n_sites))
        start_site = qd_lattice.qd_locations[idx0]
        start_pol  = qd_lattice.polaron_locs[self.get_closest_idx(qd_lattice, start_site, qd_lattice.polaron_locs)]

        # (3) running positions (unwrapped accumulator + reference)
        trajectory_start = np.asarray(start_pol, dtype=float)           # stores R(0)
        trajectory_curr  = trajectory_start.copy()                      # stores R(t)
        
        # (4) main KMC loop
        while clock < t_final:
            # (4.1) perform a KMC step from start_pol to end_pol
            _, end_pol, clock, step_comp_time = self._make_kmc_step(qd_lattice, clock, start_pol, rnd_generator=rng)
            # update computational time
            tot_comp_time += step_comp_time

            # (4.2) accumulate current position as long as we have not exceeded t_final yet
            if clock < t_final:

                # accumulate current position by raw difference
                trajectory_curr, last_r2 = self._update_displacement_minimage(
                            trajectory_curr, 
                            trajectory_start, 
                            start_pol, end_pol, 
                            box_lengths=qd_lattice.geom.lattice_dimension, periodic=True
                            )

                # add squared displacement at correct position in the times_msds grid
                inc = np.searchsorted(times_msds[time_idx:], clock)
                time_idx += inc
                if time_idx < times_msds.size:
                    sds[time_idx:] = last_r2

            # prepare next step
            start_pol = end_pol
            step_counter += 1

            # OPTIONAL : avoid doing extra KMC steps when you’ve already filled all requested MSD time points.
            if time_idx >= times_msds.size:
                break

        # NOTE : this was missing before
        # this ensurs tail is filled if loop ended before the last grid point
        # this is needed if the trajectory ended before all time grid points were reached (e.g., t_final was reached mid-interval)
        # without this, the later entries in sds would stay at zero, which would artificially drop the average MSD in those bins
        if time_idx < times_msds.size:
            sds[time_idx:] = last_r2

        return sds, tot_comp_time


    # create specific realization (instance) of QDLattice and run many trajectories
    def _run_single_lattice(self, ntrajs, bath, t_final, times, realization_id, simulated_time):

        # build QD lattice realization
        qd_lattice = self._build_grid_realization(bath, rid = realization_id)

        # get trajectory seed sequence
        traj_ss = self._spawn_trajectory_seedseq(rid = realization_id)

        # initialize mean squared displacement
        msd = np.zeros_like(times)

        for t in range(ntrajs):
            # random generator for trajectory
            rng_traj = default_rng(traj_ss[t])

            # run trajectory and resturn squared displacement in unwrapped coordinates
            sds, comp = self._run_single_kmc_trajectory(qd_lattice, t_final, rng_traj)
            simulated_time += comp

            # streaming mean over trajectories (same as before)
            w = 1.0 / (t + 1)
            msd = (1.0 - w) * msd + w * sds

        return msd, simulated_time


    # parallel KMC
    def simulate_kmc_parallel(self, max_workers = None):
        """Parallel over realizations on CPU (one process per realization)."""
        import os
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        R = self.run.nrealizations
        times_msds = self._make_time_grid()
        msds = np.zeros((R, len(times_msds)))
        sim_time = 0.0

        # Build bath ONCE in parent
        bath = SpecDens(self.bath_cfg.spectrum, const.kB * self.bath_cfg.temp)

        # Expose it to workers via module-global, then FORK the pool
        global _BATH_GLOBAL
        _BATH_GLOBAL = bath

        # Use fork context so children inherit memory instead of pickling args
        ctx = mp.get_context("fork")

        # dispatch configs (lightweight) + indices
        jobs = [(self.geom, self.dis, self.bath_cfg, self.run, self.run.t_final, times_msds, r, sim_time) for r in range(R)]

        #msds = None
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_one_lattice_worker, j) for j in jobs]
            for fut in as_completed(futs):
                rid, msd_r, sim_time = fut.result()
                msds[rid] = msd_r

        print('----------------------------------')
        print('---- SIMULATED TIME SUMMARY -----')
        print(f'total simulated time {sim_time:.3f}')
        print('----------------------------------')
        return times_msds, msds


    # serial KMC
    def simulate_kmc(self):

        times_msds = self._make_time_grid()                                 # time ranges to use for computation of msds                                                                 
        msds = np.zeros((self.run.nrealizations, len(times_msds)))          # initialize MSD output
        sim_time = 0                                                        # simulated time

        R = self.run.nrealizations                                          # number of QDLattice realizations
        T = self.run.ntrajs                                                 # number of trajetories per QDLattice realization

        # build bath spectral density (once for all QDLattice realizations!)
        bath = hamiltonian_box.SpecDens(self.bath_cfg.spectrum, const.kB * self.bath_cfg.temp)


        # loop over number of QDLattice realizations
        for r in range(R):

            # run ntrajs KMC trajectories for single QDLattice realization indexed with r
            msd, sim_time = self._run_single_lattice(ntrajs = T,
                                                     bath = bath, 
                                                     t_final = self.run.t_final, 
                                                     times = times_msds,
                                                     realization_id = r,
                                                     simulated_time = sim_time
                                                     )
                
            # store mean squared displacement for QDLattice realization r
            msds[r] = msd

            print('----------------------------------')
            print('---- SIMULATED TIME SUMMARY -----')
            print(f'total simulated time {sim_time:.3f}')
            print('----------------------------------')
        return times_msds, msds

    
    
    # (08/09/2025) more efficient version
    # NOTE : move to uitls.py?
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
    

    




