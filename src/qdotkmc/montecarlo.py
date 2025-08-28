import numpy as np
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from .config import GeometryConfig, DisorderConfig, BathConfig, RunConfig, ExecutionPlan
from numpy.random import SeedSequence, default_rng
from . import hamiltonian, lattice, const, utils, print_utils
from .hamiltonian import SpecDens
from qdotkmc.backend import get_backend

# global variable to allow parallel workers to use the same bath setup
_BATH_GLOBAL = None

# top-level worker for a single lattice realization
def _one_lattice_worker_old(args):

    # (a) CPU has 9 arguments
    if len(args) == 8:
        geom, dis, bath_cfg, run, exec_plan, times_msds, rid, sim_time, seed = args
        device_id = None
    # (b) GPU has 10 arguments (new: device_id)
    else:
        geom, dis, bath_cfg, run, exec_plan, times_msds, rid, sim_time, seed, device_id = args

    # GPU binding (only if a device is assigned)
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id) 
        os.environ["QDOT_USE_GPU"] = "1"                    

    # TODO : can we avoid setting this function
    runner = KMCRunner(geom, dis, bath_cfg, run, exec_plan)
    # (a) CPU path:
    if device_id is None:
        msd_r, sim_time_out = runner._run_single_lattice(ntrajs=run.ntrajs, 
                                                         bath=_BATH_GLOBAL, 
                                                         t_final=run.t_final, 
                                                         times=times_msds,
                                                         realization_id=rid, 
                                                         simulated_time=sim_time, 
                                                         seed=seed,
                                                         )
    # (b) GPU path:
    elif device_id is not None:
        # rebuild bath in child (spawn path)
        bath = SpecDens(bath_cfg.spectrum, const.kB * bath_cfg.temp)
        msd_r, sim_time_out = runner._run_single_lattice(ntrajs=run.ntrajs, 
                                                         bath=bath, 
                                                         t_final=run.t_final, 
                                                         times=times_msds,
                                                         realization_id=rid, 
                                                         simulated_time=sim_time, 
                                                         seed=seed,
                                                         )
    else:
        raise NotImplementedError('Need to attach each process to device ID')

    return rid, msd_r, sim_time_out

# top-level worker for a single lattice realization
def _one_lattice_worker(args):

    geom, dis, bath_cfg, run, exec_plan, times_msds, rid, sim_time, seed, device_id = args

    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ["QDOT_USE_GPU"] = "1"

    # TODO : can we avoid setting this function
    runner = KMCRunner(geom, dis, bath_cfg, run, exec_plan)

    # set up bath
    bath = SpecDens(bath_cfg.spectrum, const.kB * bath_cfg.temp)

    # run KMC on sinfle lattice realization
    msd_r, sim_time_out = runner._run_single_lattice(ntrajs=run.ntrajs, 
                                                     bath=bath, 
                                                     t_final=run.t_final, 
                                                     times=times_msds,
                                                     realization_id=rid, 
                                                     simulated_time=sim_time, 
                                                     seed=seed,
                                                     )
    return rid, msd_r, sim_time_out


class KMCRunner():
    
    
    def __init__(self, geom : GeometryConfig, dis : DisorderConfig, bath_cfg : BathConfig, 
                 run : RunConfig, exec_plan : ExecutionPlan):

        self.geom = geom                        # geometry for QDLattice
        self.dis = dis                          # energetic parameters for QDLattice
        self.bath_cfg = bath_cfg                # bath setup, temperature ...
        self.run = run                          # KMC related parameters
        self.exec_plan = exec_plan              # execution plan

        # root seed sequence controls the entire experiment for reproducibility
        self._ss_root = SeedSequence(self.dis.seed_base)

        # backend selection (GPU/CPU)
        self.backend = self.exec_plan.build_backend()

        print(self.backend.plan.n_workers, self.backend.plan.device_ids, self.backend.plan.use_gpu)

        # print which backend we end up using for KMC
        # TODO : maybe move this to main?
        mode = "GPU" if self.backend.use_gpu else "CPU"
        parallel_mode = "parallel" if self.exec_plan.do_parallel else "serial"
        print(f"[qdotkmc] backend: {mode} {parallel_mode} (use_c64={self.backend.gpu_use_c64})")                                

    
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
    def _spawn_trajectory_seedseq(self, rid : int, seed : Optional[int] = None):
        # create trajectory seed sequence specific to the seed of the realization (seed)
        # otherwise (seed = None) initialize new seeds. 
        ss_real = SeedSequence(self._spawn_realization_seed(rid)) if seed is None else SeedSequence(seed)
        return ss_real.spawn(self.run.ntrajs)
    
    
    # rate computation based on r_hop/r_ove (based on RADIUS)
    @staticmethod
    def _make_rates_radius(qd_lattice, center_global, r_hop, r_ove, selection_info = False):


        # (0) set up r_hop and r_ove, intialize box in qd_lattice as well
        # qd_lattice.redfield.r_hop, qd_lattice.redfield.r_ove = r_hop * qd_lattice.geom.qd_spacing, r_ove * qd_lattice.geom.qd_spacing

        # (1) use legacy self._get_box() and initialize box 
        # NOTE : this can likely be deleted
        r_hop, r_ove, box_length = lattice.QDLattice._init_box_dims(r_hop, r_ove, 
                                                                    spacing = qd_lattice.geom.qd_spacing,
                                                                    max_length = qd_lattice.geom.N)
        polaron_start_site = qd_lattice.polaron_locs[center_global]
        KMCRunner._get_box(qd_lattice, polaron_start_site, box_length=box_length)

        # (2) use the global indices of polaron and site inside box to further refine
        # selection by r_ove r_ove
        pol_box  = qd_lattice.pol_idxs_last
        site_box = qd_lattice.site_idxs_last

        # (3) refine the polaron and site indices by additional constraints on r_hop and r_ove
        # NOTE : refine_by_radius function can maybe be moved into this module ? 
        pol_g, site_g = qd_lattice.redfield.select_by_radius(
                    center_global = center_global,                      # reference index
                    r_hop = r_hop,
                    r_ove = r_ove,
                    pol_idxs_global = pol_box,
                    site_idxs_global = site_box,
                    periodic = True,                                     # or False to match array setup
                    grid_dims = qd_lattice.geom.lattice_dimension
                    )

        # (4) compute rates on those exact indices (no re-derivation)
        rates, final_states, tot_time = qd_lattice.redfield.make_redfield(
            pol_idxs_global=pol_g, site_idxs_global=site_g, center_global=center_global,
            verbosity = False
        )

        # (5) cache by global center index
        overall_idx_start = center_global
        qd_lattice.stored_npolarons_box[overall_idx_start] = len(pol_g)
        qd_lattice.stored_polaron_sites[overall_idx_start] = np.copy(final_states)   # global indices
        qd_lattice.stored_rate_vectors[overall_idx_start]  = np.copy(rates)

        # (4) optional: export information about selected polarons/sites for computation of rates
        sel_info= {}
        if selection_info:
            sel_info['npols_sel'] = len(pol_g)
            sel_info['nsites_sel'] = len(site_g)

        return rates, final_states, tot_time, sel_info


    # rate computation based on theta_pol/theta_tau (based on WEIGHT)
    @staticmethod
    def _make_rates_weight(qd_lattice, center_global, theta_pol, theta_site, selection_info = False):

        # (0) set up θ_pol and θ_sites 
        qd_lattice.redfield.theta_pol, qd_lattice.redfield.theta_site = theta_pol, theta_site

        # (1) select sites and polarons that ''matter'' for computing the rates
        site_g, pol_g = qd_lattice.redfield.select_by_weight(center_global = center_global, 
                                                             theta_site = qd_lattice.redfield.theta_site, 
                                                             theta_pol = qd_lattice.redfield.theta_pol, 
                                                             verbose = False
                                                             )

        # (2) compute rates on those exact indices (no re-derivation)
        rates, final_states, tot_time = qd_lattice.redfield.make_redfield(
            pol_idxs_global=pol_g, site_idxs_global=site_g, center_global=center_global,
            verbosity = False                                                               # NOTE : can change this to True to print comments in make_redfield
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

    

    def _make_rates(self, qd_lattice, center_global, *, selection_info=False, **kwargs):
        """
        dynamically switch between rate modii 
        """
        # allow one-off mode switch per call (optional)
        rates_by = kwargs.pop("rates_by", getattr(self.run, "rates_by", "radius"))

        if self.run.rates_by == "radius":
            r_hop = kwargs.pop("r_hop", self.run.r_hop)
            r_ove = kwargs.pop("r_ove", self.run.r_ove)
            if r_hop is None or r_ove is None:
                raise ValueError("Need r_hop and r_ove for radius mode (pass as kwargs or set in RunConfig).")
            # ignore irrelevant overrides quietly
            return self._make_rates_radius(qd_lattice, center_global, r_hop, r_ove, selection_info)

        elif self.run.rates_by == "weight":
            theta_pol  = kwargs.pop("theta_pol",  self.run.theta_pol)
            theta_site = kwargs.pop("theta_site", self.run.theta_site)
            if theta_pol is None or theta_site is None:
                raise ValueError("Need theta_pol and theta_site for weight mode (pass as kwargs or set in RunConfig).")
            return self._make_rates_weight(qd_lattice, center_global, theta_pol, theta_site, selection_info)

        else:
            raise ValueError(f"Unknown rates_by: {rates_by!r}")


    def _make_kmc_step(self, qd_lattice, clock, polaron_start_site, rnd_generator = None):

        # (0) check whether we have a valid instance of QDLattice class
        assert isinstance(qd_lattice, lattice.QDLattice), "need to feed valid QDLattice instance!"

        # (1) start polarong (global) idx and location 
        center_global = utils.get_closest_idx(qd_lattice, polaron_start_site, qd_lattice.polaron_locs)
        start_pol = qd_lattice.polaron_locs[center_global]

        # (2) compute (or reuse) rates
        if qd_lattice.stored_npolarons_box[center_global] == 0:
            # compute rates
            rates, final_states, tot_time, _ = self._make_rates(qd_lattice, 
                                                                center_global, 
                                                                r_hop = self.run.r_hop, 
                                                                r_ove = self.run.r_ove
                                                                )
        else:
            tot_time = 0.0
            final_states = qd_lattice.stored_polaron_sites[center_global]  
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
    def _build_grid_realization(self, bath : SpecDens, rid : int, seed : Optional[int] = None):

        assert isinstance(bath, SpecDens), "Need to make sure we have a proper \
                                            bath set up to build QDLattice instance"

        # get random seef from realization id (rid), if no seed already specified
        if seed is None:
            rnd_seed = self._spawn_realization_seed(rid)
        else:
            rnd_seed = seed 
        
        # initialize instance of QDLattice class
        qd = lattice.QDLattice(geom=self.geom, dis=self.dis, seed_realization=rnd_seed)

        # attach GPU/CPU backend
        qd.backend = self.backend

        # setup QDLattice with (polaron-transformed) Hamiltonian, bath information, Redfield
        qd._setup(bath)

        return qd, rnd_seed
    
    
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
        start_pol  = qd_lattice.polaron_locs[utils.get_closest_idx(qd_lattice, start_site, qd_lattice.polaron_locs)]

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
    def _run_single_lattice(self, ntrajs, bath, t_final, times, realization_id, simulated_time, seed = None):

        # build QD lattice realization
        qd_lattice, real_seed = self._build_grid_realization(bath, rid = realization_id, seed = seed)

        # get trajectory seed sequence
        traj_ss = self._spawn_trajectory_seedseq(rid = realization_id, seed = real_seed)

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

    # execute parallel if available based on max_worker (otherwise serial)
    def _simulate_kmc(self):
        #if self.exec_plan.max_workers is None or self.exec_plan.max_workers == 1:
        if not self.exec_plan.do_parallel:
            return self.simulate_kmc_serial()
        else:
            return self.simulate_kmc_parallel()

    # parallel KMC
    def simulate_kmc_parallel_old(self):
        """Parallel over realizations. Uses fork on CPU, spawn on GPU (one process per GPU)."""

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        R = self.run.nrealizations
        times_msds = self._make_time_grid()
        msds = np.zeros((R, len(times_msds)))
        sim_time = 0.0

        # decide backend mode
        use_gpu = getattr(self.backend, "use_gpu", False)

        # (a)  -------  CPU path ---------
        if not use_gpu:

            # build bath ONCE in parent and share via fork 
            bath = SpecDens(self.bath_cfg.spectrum, const.kB * self.bath_cfg.temp)
            global _BATH_GLOBAL
            _BATH_GLOBAL = bath

            # use fork context so children inherit memory instead of pickling args
            ctx = mp.get_context("fork")

            # create seeds for lattice realizations, its important to feed them here
            # in order to have parallel execution yield the same results as serial
            seeds = [self._spawn_realization_seed(rid) for rid in range(R)]
            jobs = [(self.geom, self.dis, self.bath_cfg, self.run, self.exec_plan, times_msds, rid, sim_time, seeds[rid])
                    for rid in range(R)]

            with ProcessPoolExecutor(max_workers=self.exec_plan.max_workers, mp_context=ctx) as ex:
                futs = [ex.submit(_one_lattice_worker, j) for j in jobs]
                for fut in as_completed(futs):
                    rid, msd_r, sim_time = fut.result()
                    msds[rid] = msd_r

        # (b) --------- GPU path ---------------
        elif use_gpu:

            # GPU: spawn + one process per GPU
            # avoid parent-global bath in GPU mode because spawn doesn’t inherit it, rebuild in child
            ctx = mp.get_context("spawn")

            # detect how many GPUs are visible (fallback to 1 if detection fails)
            try:
                #import cupy as cp
                n_gpus = int(self.backend.cp.cuda.runtime.getDeviceCount())
            except Exception:
                n_gpus = 1

            # cap workers to # of GPUs (one process per GPU recommended)
            # TODO : can we change this to make code even faster?
            max_workers = max(1, min(self.exec_plan.max_workers, n_gpus))

            # create seeds for lattice realizations, its important to feed them here
            # in order to have parallel execution yield the same results as serial
            seeds = [self._spawn_realization_seed(rid) for rid in range(R)]

            # assign each job a device id round-robin over [0..n_gpus-1]
            jobs = []
            for rid in range(R):
                dev = rid % n_gpus
                jobs.append((self.geom, self.dis, self.bath_cfg, self.run, self.exec_plan, times_msds, rid, sim_time, seeds[rid], dev))

            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futs = [ex.submit(_one_lattice_worker, j) for j in jobs]
                for fut in as_completed(futs):
                    rid, msd_r, sim_time = fut.result()
                    msds[rid] = msd_r

        # print total time spent on Redfield rates
        print(print_utils.simulated_time(sim_time))

        return times_msds, msds
    
    # parallel KMC
    def simulate_kmc_parallel(self):
        """Parallel over realizations. Uses fork on CPU, spawn on GPU (one process per GPU)."""
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        R = self.run.nrealizations
        times_msds = self._make_time_grid()
        msds = np.zeros((R, len(times_msds)))
        sim_time = 0.0

        # create seeds for lattice realizations, its important to feed them here
        # in order to have parallel execution yield the same results as serial
        seeds = [self._spawn_realization_seed(rid) for rid in range(R)]

        # use fork context so children inherit memory instead of pickling args (for CPU)
        # or use spawn (for GPU) 
        ctx = mp.get_context(self.backend.plan.context)
        jobs = []
        # (a) GPU bath
        if self.backend.plan.device_ids:  
            for rid in range(R):
                dev = self.backend.plan.device_ids[rid % len(self.backend.plan.device_ids)]
                jobs.append((self.geom, self.dis, self.bath_cfg, self.run, self.exec_plan,
                            times_msds, rid, sim_time, seeds[rid], dev))
        # (b) CPU path
        else:
            print('do CPU path')
            dev = self.backend.plan.device_ids                                              # should be None for CPU path
            jobs = [(self.geom, self.dis, self.bath_cfg, self.run, self.exec_plan,
                    times_msds, rid, sim_time, seeds[rid], dev) for rid in range(R)]
        
        with ProcessPoolExecutor(max_workers=self.exec_plan.max_workers, mp_context=ctx) as ex:
                futs = [ex.submit(_one_lattice_worker, j) for j in jobs]
                for fut in as_completed(futs):
                    rid, msd_r, sim_time = fut.result()
                    msds[rid] = msd_r
        
        # print total time spent on Redfield rates
        print(print_utils.simulated_time(sim_time))

        return times_msds, msds
        

    # serial KMC
    def simulate_kmc_serial(self):

        times_msds = self._make_time_grid()                                 # time ranges to use for computation of msds                                                                 
        msds = np.zeros((self.run.nrealizations, len(times_msds)))          # initialize MSD output
        sim_time = 0                                                        # simulated time

        R = self.run.nrealizations                                          # number of QDLattice realizations
        T = self.run.ntrajs                                                 # number of trajetories per QDLattice realization

        # build bath spectral density (once for all QDLattice realizations!)
        bath = hamiltonian.SpecDens(self.bath_cfg.spectrum, const.kB * self.bath_cfg.temp)


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

        # print total time spent on Redfield rates
        print(print_utils.simulated_time(sim_time))

        return times_msds, msds

    # make box around center position where we are currently at
    # TODO : incorporate periodic boundary conditions explicty (boolean)
    # NOTE : this can likely be deleted as it not doing much, which we couldn't implement directly
    @staticmethod
    def _get_box(qd_lattice, center, box_length, periodic=True):

        # (1) box size (unchanged)
        #qd_lattice.box_size = qd_lattice.box_length * qd_lattice.geom.qd_spacing

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
        pol_idxs = find_indices_within_box(qd_lattice.polaron_locs, center, qd_lattice.geom.lattice_dimension, box_length, periodic)
        site_idxs = find_indices_within_box(qd_lattice.qd_locations,  center, qd_lattice.geom.lattice_dimension, box_length, periodic)

        # keep order, store contiguously
        qd_lattice.pol_idxs_last  = np.ascontiguousarray(pol_idxs.astype(np.intp))
        qd_lattice.site_idxs_last = np.ascontiguousarray(site_idxs.astype(np.intp))

        # (4) define the GLOBAL center index once
        # NOTE : it seems like eventuall this is all we need, so we might get rid of this function alltogether ??
        qd_lattice.center_global = int(utils.get_closest_idx(qd_lattice, center, qd_lattice.polaron_locs))

        # (5) optional: local position of the center inside the box (rarely needed now)
        where = np.nonzero(qd_lattice.pol_idxs_last == qd_lattice.center_global)[0]
        # If the box is tight or discrete, it should be present; if not, refine_by_radius will handle it.
        qd_lattice.center_local = int(where[0]) if where.size == 1 else None

    


    

    




