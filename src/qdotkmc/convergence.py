import numpy as np
from numpy.random import default_rng
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .config import GeometryConfig, DisorderConfig, BathConfig, RunConfig, ConvergenceTuneConfig, ExecutionPlan
from . import const
from .hamiltonian import SpecDens
from .montecarlo import KMCRunner

# global variable to allow parallel workers to use the same QDLattice for convergence tests
_QDLAT_GLOBAL = None

# top-level worker for computing the rate scores from a single lattice site
def _rate_score_worker(args):
    (start_idx, theta_pol, theta_site, criterion, weight) = args
    # compute rates for this start index
    qd_lattice = _QDLAT_GLOBAL
    rates, final_sites, _, sel_info = KMCRunner._make_rates_weight(qd_lattice, start_idx,
                                                                   theta_pol=theta_pol, theta_site=theta_site,
                                                                   selection_info = True)
    
    # evaluate convergence criterion on rates vector
    if criterion == "rate-displacement":
        start_loc = qd_lattice.qd_locations[start_idx]                                                      # r(0)
        sq_displacments = ((qd_lattice.qd_locations[final_sites] - start_loc)**2).sum(axis = 1)             # ||Δr||^2 per destination
        lamda = (rates * sq_displacments).sum() / (2 * qd_lattice.geom.dims)
    else:
        raise ValueError("please specify valid convergence criterion for rates!")
    
    return lamda * weight, sel_info['nsites_sel'], sel_info['npols_sel']


# top-level worker for computing the rate scores from a single lattice site
def _rate_score_worker_gpuswitch(args):

    (geom, dis, bath_cfg, exec_plan,
     start_idx, theta_pol, theta_site, criterion, weight,
     rnd_seed, device_id) = args

    import time 
    start = time.time()
    qd_lattice = None
    # CPU path (fork): fork qd_lattice from global environment (inherited from parent)
    if device_id is None:
        qd_lattice = _QDLAT_GLOBAL
    # GPU path (spawn): need to rebuild qd_lattice locally
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ["QDOT_USE_GPU"] = "1"  

        # set up bath
        bath = SpecDens(bath_cfg.spectrum, const.kB * bath_cfg.temp)
        # build backend locally
        backend = exec_plan.build_backend()
        # create lattice 
        qd_lattice, _ = KMCRunner._build_grid_realization(geom = geom,
                                                          dis = dis,
                                                          bath = bath,
                                                          seed = rnd_seed,
                                                          backend = backend)

    end = time.time()
    print(f'build lattice: {end-start:.6f}', flush=True)
    # compute rates for this qd_lattice and specified start_index
    rates, final_sites, _, sel_info = KMCRunner._make_rates_weight(qd_lattice, start_idx,
                                                                   theta_pol=theta_pol, theta_site=theta_site,
                                                                   selection_info=True)

    # evaluate convergence criterion on rates vector
    if criterion == "rate-displacement":
        start_loc = qd_lattice.qd_locations[int(start_idx)]
        sq_disp = ((qd_lattice.qd_locations[final_sites] - start_loc)**2).sum(axis=1)
        lamda = (rates * sq_disp).sum() / (2 * qd_lattice.geom.dims)
    else:
        raise ValueError("please specify valid convergence criterion for rates!")

    return lamda * float(weight), int(sel_info['nsites_sel']), int(sel_info['npols_sel'])



# TODO : only implemented for rates_by = "weight" so far 
class ConvergenceAnalysis(KMCRunner):

    def __init__(self, geom : GeometryConfig, dis : DisorderConfig, bath_cfg : BathConfig, run : RunConfig, exec_plan : ExecutionPlan,
                 tune_cfg : ConvergenceTuneConfig):

        super().__init__(geom, dis, bath_cfg, run, exec_plan)
        self.tune_cfg = tune_cfg
        self.exec_plan = exec_plan

        assert geom.n_sites >= tune_cfg.no_samples, "cannot have no_sample >= number of sites in lattice"

        # backend selection
        self.backend = self.exec_plan.build_backend()

        # intialize environment to perform rate convergence analysis in
        self._build_rate_convergenc_env()

    
    # build setu-uop for obtaining convergence
    def _build_rate_convergenc_env(self):

        # (0) build bath
        bath = SpecDens(self.bath_cfg.spectrum, const.kB * self.bath_cfg.temp)

        # (1) draw a lattice realization
        self.rnd_seed = self._spawn_realization_seed(rid = 0)
        self.qd_lattice, _ = KMCRunner._build_grid_realization(geom = self.geom,
                                                               dis = self.dis,
                                                               bath = bath,
                                                               seed = self.rnd_seed,
                                                               backend = self.backend)

        # (2) produce no_samples starting indices from where to compute rate vectors
        ss_conv = self._ss_root.spawn(1)[0]
        rng_conv = default_rng(ss_conv)
        self.start_sites = rng_conv.integers(0, self.qd_lattice.geom.n_sites, size=self.tune_cfg.no_samples)

        # (3) get Boltzmann weights for each start polaron in start_sites
        E = self.qd_lattice.full_ham.evals
        beta = self.qd_lattice.full_ham.beta
        w = np.exp(- beta * E[self.start_sites])
        Z = np.sum(np.exp(- beta * E))
        self.weights = w / Z


    # compute rate score in parallel (if available) or in serial
    def _rate_score(self, *args, **kwargs):
        #if self.tune_cfg.max_workers is None or self.tune_cfg.max_workers == 1:
        if not self.exec_plan.do_parallel:
            return self._rate_score_serial(*args, **kwargs)
        else:
            return self._rate_score_parallel(*args, **kwargs)

    
    def _rate_score_serial(self, theta_pol, theta_site, score_info = True):

        # get rates starting from each polaron starting index and analyze by criterion
        rates_criterion = None
        lambdas = np.zeros_like(self.start_sites, dtype=np.float32)
        nsites_sel, npols_sel = 0, 0
        for i, start_idx in enumerate(self.start_sites):

            # NEW way to obtain rates from theta_pol/theta_site
            rates, final_sites, _, sel_info = KMCRunner._make_rates_weight(self.qd_lattice, start_idx,
                                                                           theta_site=theta_site,
                                                                           theta_pol=theta_pol,
                                                                           selection_info = True
                                                                           )
            
            # how many polarons/sites were selected 
            nsites_sel += sel_info['nsites_sel']
            npols_sel += sel_info['npols_sel']
            
            # evaluate convergence criterion on rates vector
            if self.tune_cfg.criterion == "rate-displacement":
                start_loc = self.qd_lattice.qd_locations[start_idx]                                                      # r(0)
                sq_displacments = ((self.qd_lattice.qd_locations[final_sites] - start_loc)**2).sum(axis = 1)             # ||Δr||^2 per destination
                lamda = (rates * sq_displacments).sum() / (2 * self.qd_lattice.geom.dims)
                lambdas[i] = lamda
            else:
                raise ValueError("please specify valid convergence criterion for rates!")
        
        rates_criterion = (self.weights * lambdas).sum()

        # optional : store additional information
        info = {}
        if score_info:
            info['ave_sites'] = nsites_sel / self.tune_cfg.no_samples
            info['ave_pols'] = npols_sel / self.tune_cfg.no_samples

        return rates_criterion, info


    def _rate_score_parallel_old(self, theta_pol, theta_site, score_info = True):
        """
        Parallel version of _rate_score over self.start_sites.
        Returns the same aggregate score and selection counts.
        """
        
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        # Expose QDLattice to workers via module-global, then FORK the pool
        global _QDLAT_GLOBAL
        _QDLAT_GLOBAL = self.qd_lattice

        # Use fork context so children inherit memory instead of pickling args
        ctx = mp.get_context("fork")   

        # Weighted sum uses the Boltzmann weights you precomputed per start index.
        weight_by_idx = {int(i): float(w) for i, w in zip(self.start_sites, self.weights)}

        # dispatch configs + indices to parallelize over
        jobs = [(int(start_idx), float(theta_pol), float(theta_site), self.tune_cfg.criterion, weight_by_idx[start_idx]) 
                for start_idx in self.start_sites]

        rates_criterion = 0
        nsites_sel, npols_sel = 0, 0

        with ProcessPoolExecutor(max_workers=self.tune_cfg.max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_rate_score_worker, job) for job in jobs]
            for fut in as_completed(futs):

                # let worker obtain weighted convergence criterion
                weighted_criterion, nsite_sel, npol_sel = fut.result()

                nsites_sel += nsite_sel
                npols_sel += npol_sel
                rates_criterion += weighted_criterion
                

        # optional : store additional information
        info = {}
        if score_info:
            info['ave_sites'] = nsites_sel / self.tune_cfg.no_samples
            info['ave_pols'] = npols_sel / self.tune_cfg.no_samples

        return rates_criterion, info


    def _rate_score_parallel_gpuswitch(self, theta_pol, theta_site, score_info = True):

        """Parallel over realizations. Uses fork on CPU, spawn on GPU (one process per GPU)."""

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        # expose QDLattice to workers via module-global, then fork it in CPU path
        if not self.backend.use_gpu:
            global _QDLAT_GLOBAL
            _QDLAT_GLOBAL = self.qd_lattice

        # weights lookup for start sites
        weight_by_idx = {int(i): float(w) for i, w in zip(self.start_sites, self.weights)}

        # build jobs for CPU/GPU path and define context (spawn/fork)
        ctx = mp.get_context(self.backend.plan.context)
        devs = self.backend.plan.device_ids if self.backend.use_gpu else [None]
        jobs = []
        for k, start_idx in enumerate(self.start_sites):
            device_id = devs[k % len(devs)]
            jobs.append((
                self.geom, self.dis, self.bath_cfg, self.exec_plan,
                start_idx, theta_pol, theta_site, self.tune_cfg.criterion, weight_by_idx[start_idx],
                self.rnd_seed,                                                  # deterministic lattice rebuild
                (None if not self.backend.use_gpu else device_id),
            ))

        rates_criterion = 0
        nsites_sel, npols_sel = 0, 0

        # allocate jobs to workers
        # TODO : 
        with ProcessPoolExecutor(max_workers=self.exec_plan.max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_rate_score_worker_new, job) for job in jobs]
            for fut in as_completed(futs):

                # let worker obtain weighted convergence criterion
                lam_w, nsite_sel, npol_sel = fut.result()

                rates_criterion += lam_w
                nsites_sel      += nsite_sel
                npols_sel       += npol_sel

        info = {}
        if score_info:
            info['ave_sites'] = nsites_sel / float(self.tune_cfg.no_samples)
            info['ave_pols']  = npols_sel  / float(self.tune_cfg.no_samples)

        return rates_criterion, info


    def _rate_score_parallel(self, theta_pol, theta_site, score_info=True):


        use_gpu = self.backend.plan.use_gpu and self.exec_plan.do_parallel

        # Expose QDLattice to workers via module-global, then FORK the pool
        global _QDLAT_GLOBAL
        _QDLAT_GLOBAL = self.qd_lattice

        weights = {int(i): float(w) for i, w in zip(self.start_sites, self.weights)}

        jobs = [(self.qd_lattice, int(idx), float(theta_pol), float(theta_site),
                self.tune_cfg.criterion, weights[int(idx)]) for idx in self.start_sites]

        rates_criterion = 0.0
        nsites_sel = 0
        npols_sel  = 0

        # CPU = processes (fork), GPU = threads (one process, shared CUDA context)
        if use_gpu:
            # keep moderate workers (4–8); cuBLAS/cuSolver are already parallel
            #n_workers = min(self.tune_cfg.max_workers or 4, 8)
            with ThreadPoolExecutor(max_workers=self.backend.plan.n_workers) as ex:
                for fut in as_completed(ex.submit(_rate_score_worker_thread, j) for j in jobs):
                    lam_w, ns, np_ = fut.result()
                    rates_criterion += lam_w; nsites_sel += ns; npols_sel += np_
        else:
            # your existing process-based path (fork)
            ctx = mp.get_context(self.backend.plan.context)
            with ProcessPoolExecutor(max_workers=self.backend.plan.n_workers, mp_context=ctx) as ex:
                for fut in as_completed(ex.submit(_rate_score_worker, (
                        int(idx), float(theta_pol), float(theta_site),
                        self.tune_cfg.criterion, weights[int(idx)]
                )) for idx in self.start_sites):
                    lam_w, ns, np_ = fut.result()
                    rates_criterion += lam_w; nsites_sel += ns; npols_sel += np_

        info = {}
        if score_info:
            info['ave_sites'] = nsites_sel / float(self.tune_cfg.no_samples)
            info['ave_pols']  = npols_sel  / float(self.tune_cfg.no_samples)
        return rates_criterion, info

    @staticmethod
    def _per_oct_gain(lam_from: float, lam_to: float, span_factor: float):
        '''
        Compute the normalized fractional improvement ("gain") per octave of shrinkage.
        Given two rate-scores Λ (before and after tightening) and the shrinkage factor
        span_factor = θ_new / θ_old, this function measures the relative improvement
        scaled by the number of octaves of reduction (log_2 of the shrinkage).

        Parameters
        ----------
        lam_from : float
            Initial rate-score Λ at the looser threshold.
        lam_to : float
            Rate-score Λ after shrinkage.
        span_factor : float
            Shrinkage ratio (new threshold / old threshold), typically < 1.

        Returns
        -------
        float
            Normalized fractional improvement per octave of shrinkage.
            A value ≈ δ (1 - 2 %) indicates a plateau where further tightening 
            brings little additional benefit.
        '''
        # (1) relative gain
        rel = (lam_to - lam_from) / (abs(lam_from) + 1e-300)
        # (2) normalize per-octave 
        octaves = np.log(1.0 / max(span_factor, 1e-12)) / np.log(2.0)

        return rel / max(octaves, 1e-12)

    # inner progressive shrinkage algorithm for finding θ_pol^* for fixed θ_sites
    def _tune_theta_pol(self, theta_site: float, verbose = True):
        
        '''
        Inner progressive shrinkage algorithm to find the optimal θ_pol^* for a fixed θ_sites.
        Starting from an initial θ_pol, the algorithm repeatedly shrinks θ_pol by a factor ρ
        and evaluates the corresponding rate-score Λ. The process continues until further
        tightening yields negligible improvement (per-octave gain ≤ δ) or θ_pol_min is reached.

        Parameters
        ----------
        theta_site : float
            Fixed θ_sites value at which θ_pol is tuned.
        theta_pol_start : float
            Initial (looser) θ_pol value.
        theta_pol_min : float
            Minimum (tightest) allowed θ_pol value.
        rho : float
            Shrinkage factor applied each step (typically < 1).
        delta : float
            Plateau threshold, target fractional gain per octave (approx. 1 - 2%).
        max_steps : int
            Maximum number of shrinkage steps to attempt.
        criterion : str
            Convergence criterion for the rate-score.
        verbose : bool
            If True, prints progress of each shrinkage step.

        Returns
        -------
        tuple
            (θ_pol^*, Λ^*), where θ_pol^* is the selected polarization threshold 
            and Λ^* the corresponding rate-score at fixed θ_sites.
    
        '''
        # (0) initialize θ_pol
        theta_p = float(self.tune_cfg.theta_pol_start)
        # evaluate rate-score Λ at current (initial) θ_pol
        lam_from, info = self._rate_score(theta_p, theta_site, score_info=True)

        for _ in range(int(self.tune_cfg.max_steps_pol)):

            # (1) perform shrinkage
            theta_p_next = max(float(self.tune_cfg.theta_pol_min), self.tune_cfg.rho * theta_p)
            if theta_p_next >= theta_p - 1e-15:
                break

            # (2) evaluate new rate score for 
            lam_to, info = self._rate_score(theta_p_next, theta_site, score_info=True)

            # (3) per-octave gain G_p over a fixed span θ_pol -> ρ * θ_pol
            gain = self._per_oct_gain(lam_from, lam_to, self.tune_cfg.rho)

            # (4) accept move by default
            theta_p, lam_from = theta_p_next, lam_to

            if verbose:
                nsites, npols = int(info['ave_sites']), int(info['ave_pols'])
                print(f"[pol] tp→{theta_p:.4f}  nsites={nsites} npols={npols} per-oct gain={gain*100:.2f}%/oct")

            # (5) stop when additional tightening barely helps, i.e. G_p <= 𝛿
            if gain < float(self.tune_cfg.delta) or theta_p <= float(self.tune_cfg.theta_pol_min) + 1e-12:
                break
        
        # return θ_pol^* and corresponding Λ(θ_sites, θ_pol^*) for fixed θ_sites as well as dictionary score_info
        return theta_p, float(lam_from), info

    # main auto-tune loop to obtain θ_pol/θ_sites
    def auto_tune_thetas(self, verbose = True):
        '''
        Automatically tune the convergence thresholds (θ_sites, θ_pol) by nested optimization.

        The algorithm balances efficiency (looser cutoffs) against accuracy (tighter cutoffs):
        1. For a given θ_sites, θ_pol is optimized internally via _tune_theta_pol.
        2. Using this inner optimization, we evaluate the per-octave gain G_s(θ_sites) by
            tightening θ_sites → ρ * θ_sites and comparing improvements.
        3. A bisection search in log(θ_sites) space finds the largest θ_sites with 
            G_s(θ_sites) ≤ δ (plateau criterion).
        4. Edge cases are handled:
            - If even the tightest θ_sites is too steep, return θ_sites = hi.
            - If the loosest θ_sites is already flat, return θ_sites = lo.

        Parameters
        ----------
        theta_site_lo, theta_site_hi : float
            Bracketing range for θ_sites (looser > tighter).
        theta_pol_start, theta_pol_min : float
            Initial and minimum values for θ_pol in the inner loop.
        rho : float
            Multiplicative span factor for testing improvements (typically < 1).
        delta : float
            Plateau threshold, target fractional gain per octave (approx. 1 - 2%).
        max_outer : int
            Maximum number of bisection iterations.
        criterion : str
            Convergence criterion for the rate-score.
        verbose : bool
            If True, prints progress of the bisection search.

        Returns
        -------
        dict
            Dictionary with final values of optimization.

        '''
        
        # ensure valid ordering of θ_sites bracket θ_sites ∈ [lo, hi] with lo > hi
        lo = max(self.tune_cfg.theta_site_lo, self.tune_cfg.theta_site_hi)
        hi = min(self.tune_cfg.theta_site_lo, self.tune_cfg.theta_site_hi)
        if not (lo > hi):
            raise ValueError("Require theta_site_lo > theta_site_hi (looser > tighter).")

        # define a function : compute per-octave gain for sites by comparing ts and ρ * θ_sites
        # we call this funcion g = G_s(θ_sites), note that every call of this function triggers the inner loop
        def sites_gain(theta_s: float, verbose=verbose):
            theta_s_tight = max(hi, self.tune_cfg.rho * theta_s)
            _, lam_lo, info_lo = self._tune_theta_pol(theta_s, verbose=verbose)
            _, lam_hi, info_hi = self._tune_theta_pol(theta_s_tight, verbose=verbose)
            g = self._per_oct_gain(lam_lo, lam_hi, max(theta_s_tight / theta_s, 1e-12))
            return g, lam_lo, lam_hi, info_hi

        #  -------------------------    (1) edge-case handling     ----------------------------------------
        # (1.1) evaluate g_lo = G_s(lo) and g_hi = G_s(hi)
        g_lo, lam_lo, _, info_lo = sites_gain(lo, verbose=verbose)                                # loose point gain toward tighter
        g_hi, _, lam_hi, info_hi = sites_gain(hi, verbose=verbose)                                # tight point gain toward even tighter (may be zero-span)

        # (1.2) if even the tight end is still “steep”, return the tightest (best we can do)
        if g_hi > float(self.tune_cfg.delta):
            tp_star, lam_star, info_star = self._tune_theta_pol(hi, verbose=verbose)
            print('[range-warning] algorithm cannot yield a reasonable result at hi (tight end of theta_site is not flat enough).')
            return dict(theta_site=hi, theta_pol=tp_star, lambda_final=float(lam_star), 
                        nsites=info_star['ave_sites'], npols=info_star['ave_pols'])

        # (1.3) if the loose end is already “flat”, keep the largest (cheapest) feasible theta_site
        if g_lo <= float(self.tune_cfg.delta):
            tp_star, lam_star, info_star = self._tune_theta_pol(lo, verbose=verbose)
            print('[range-warning] algorithm yields trivial result at lo (loose end of  is already flat).')
            return dict(theta_site=lo, theta_pol=tp_star, lambda_final=float(lam_star), 
                        nsites=info_star['ave_sites'], npols=info_star['ave_pols'])

        #  -------------------------    (2) bisection search for θ_sites     ----------------------------------
        # otherwise, bisection on log(θ_sites) to find largest θ_sites with gain <= 𝛿  
        for _ in range(int(self.tune_cfg.max_outer)):

            # (1) get bisection midpoint 
            mid = float(np.sqrt(lo * hi))                               # geometric midpoint (bisection in log-space)
            g_mid, _, _, info_mid = sites_gain(mid, verbose=verbose)    # evaluate G_s(mid)

            # (2) make decision based on G_s(mid)
            if g_mid > float(self.tune_cfg.delta):
                lo = mid                                                # still steep at mid, i.e. move tighter
            else:
                hi = mid                                                # flat enough at mid, i.e. keep it as new “hi” (feasible)
            
            if verbose:
                nsites_mid, npols_mid = int(info_mid['ave_sites']), int(info_mid['ave_pols'])
                print(f"[sites] lo={lo:.4f} hi={hi:.4f} mid={mid:.4f} nsites(mid)={nsites_mid} npols(mid)={npols_mid} gain(mid)={g_mid*100:.2f}%/oct")

            if lo / hi <= 1.10:                                         # bracket [lo, hi] within ~ 10 % is enough
                break

        #  -------------------------    (3) obtain θ_sites^*, θ_pol^*, Λ^*     ----------------------------------
        # finalize at hi (largest θ_sites in bracket with gain <= 𝛿)
        tp_star, lam_star, info_star = self._tune_theta_pol(hi, verbose=verbose)
        
        return dict(theta_site=hi, theta_pol=tp_star, lambda_final=float(lam_star), 
                    nsites=int(info_star['ave_sites']), npols=int(info_star['ave_pols']))

    