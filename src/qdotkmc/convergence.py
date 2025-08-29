import numpy as np
from numpy.random import default_rng
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from typing import Optional, Dict

from .config import GeometryConfig, DisorderConfig, BathConfig, RunConfig, ConvergenceTuneConfig, ExecutionPlan
from . import const
from .hamiltonian import SpecDens
from .montecarlo import KMCRunner
from qdotkmc.backend import Backend, get_backend


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
def _rate_score_worker_new(args):

    (geom, dis, bath_cfg, exec_plan,
     start_idx, theta_pol, theta_site, criterion, weight,
     rnd_seed, device_id) = args

    import time 
    qd_lattice = None
    # CPU path (fork): fork qd_lattice from global environment (inherited from parent)
    if device_id is None:
        qd_lattice = _QDLAT_GLOBAL
    # GPU path (spawn): need to rebuild qd_lattice locally
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ["QDOT_USE_GPU"] = "1"  

        # set up bath
        start = time.time()
        bath = SpecDens(bath_cfg.spectrum, const.kB * bath_cfg.temp)
        end = time.time()
        print(f'build SpecDens: {end-start:.6f}', flush=True)
        # build backend locally
        start = time.time()
        backend = exec_plan.build_backend()
        end = time.time()
        print(f'build backend: {end-start:.6f}', flush=True)
        # create lattice 
        qd_lattice, _ = KMCRunner._build_grid_realization(geom = geom,
                                                          dis = dis,
                                                          bath = bath,
                                                          seed = rnd_seed,
                                                          backend = backend)

    
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


def _rate_score_worker_gpu(in_q: mp.queues.Queue, out_q: mp.queues.Queue):

    qd_lattice = None

    while True:

        msg = in_q.get()

        # decide if we stop loop
        if msg[0] == "stop":
            break

        # if we are in init mode, we create qd_lattice once
        if msg[0] == "init":

            # (0) load arguments
            (geom_cfg, dis_cfg, bath_cfg, seed, prefer_gpu, use_c64, device_id) = msg[1]

            # (1) intialize cuda/cupy
            import cupy as cp
            cp.cuda.Device(int(device_id)).use()
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            try:
                cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
            except Exception:
                pass
            
            # (2) build backend on selected device with device_id
            backend = get_backend(prefer_gpu=prefer_gpu, use_c64=use_c64)

            # (3) build SpecDens
            bath = SpecDens(bath_cfg.spectrum, const.kB * bath_cfg.temp)

            # (4) build qd_lattice
            qd_lattice, _ = KMCRunner._build_grid_realization(geom=geom_cfg, 
                                                              dis=dis_cfg, 
                                                              bath=bath, 
                                                              seed=seed, 
                                                              backend=backend)

            #out_q.put(("ok", None))

        # if we are in batch mode, we use created qd_lattice
        elif msg[0] == "batch":
            (batch_indices, theta_pol, theta_site, criterion, weights) = msg[1]
            lam_sum = 0.0
            nsites_sum = 0
            npols_sum = 0
            
            for start_idx in batch_indices:
                rates, final_sites, _, sel_info = KMCRunner._make_rates_weight(
                            qd_lattice, start_idx,
                            theta_pol = theta_pol, theta_site = theta_site,
                            selection_info=True
                            )

                if criterion != "rate-displacement":
                    raise ValueError("invalid criterion")

                s0 = qd_lattice.qd_locations[start_idx]
                dr2 = ((qd_lattice.qd_locations[final_sites] - s0) ** 2).sum(axis=1)
                lam = (rates * dr2).sum() / (2 * qd_lattice.geom.dims)

                w = weights.get(int(start_idx), 1.0)
                lam_sum += lam * w
                nsites_sum += sel_info['nsites_sel']
                npols_sum += sel_info['npols_sel']

            out_q.put(("batch_done", (lam_sum, nsites_sum, npols_sum)))



class GpuRatePool:
    def __init__(self, backend : Backend):

        # load from backend 
        self.use_gpu = backend.use_gpu 
        self.use_c64 = backend.gpu_use_c64
        self.max_procs = 8#backend.plan.n_workers
        self.ctx = mp.get_context(backend.plan.context)
        self.device_ids = backend.plan.device_ids

        # initialize GpuPool attributes
        self.procs = []
        self.inqs = []
        self.outqs = []

    @staticmethod
    def _chunks(seq, k):
        n = len(seq)
        if n == 0: return []
        m = math.ceil(n / k)
        return [seq[i:i+m] for i in range(0, n, m)]


    def start(self, geom_cfg, dis_cfg, bath_cfg, seed):
        
        # spawn workers
        for i in range(self.max_procs):

            in_q = self.ctx.Queue()
            out_q = self.ctx.Queue()
            p = self.ctx.Process(target=_rate_score_worker_gpu, args=(in_q, out_q))
            p.start()
            self.procs.append(p); self.inqs.append(in_q); self.outqs.append(out_q)

            # init each worker with the same config/seed, pinned to a device
            dev = self.device_ids[i % len(self.device_ids)]
            in_q.put(("init", (geom_cfg, dis_cfg, bath_cfg, seed,
                               self.use_gpu, self.use_c64, dev)))
            #tag, payload = out_q.get()
            # if tag != "ok":
            #     raise RuntimeError(f"GPU worker init failed: {payload}")


    def run_batches(self, start_indices, theta_pol, theta_site, criterion, weights: Dict[int, float]):
        batches = GpuRatePool._chunks(list(map(int, start_indices)), max(1, len(self.inqs)))

        # send work to workers
        for i, batch in enumerate(batches):
            self.inqs[i].put(("batch", (batch, float(theta_pol), float(theta_site), criterion, weights)))

        # collect
        lam_total = 0.0; nsites_total = 0; npols_total = 0
        for i in range(len(batches)):
            tag, payload = self.outqs[i].get()
            # if tag != "batch_done":
            #     raise RuntimeError(f"worker error: {payload}")
            lam, ns, np_ = payload
            lam_total += lam; nsites_total += ns; npols_total += np_

        return lam_total, nsites_total, npols_total


    # close GPU pool
    def close(self):
        for q in self.inqs:
            q.put(("stop", None))
        for p in self.procs:
            p.join()
        self.procs.clear(); self.inqs.clear(); self.outqs.clear()


# TODO : only implemented for rates_by = "weight" so far 
class ConvergenceAnalysis(KMCRunner):

    def __init__(self, geom : GeometryConfig, dis : DisorderConfig, bath_cfg : BathConfig, run : RunConfig, exec_plan : ExecutionPlan,
                 tune_cfg : ConvergenceTuneConfig):

        super().__init__(geom, dis, bath_cfg, run, exec_plan)
        self.tune_cfg = tune_cfg
        self.exec_plan = exec_plan

        self._gpu_pool = None

        assert geom.n_sites >= tune_cfg.no_samples, "cannot have no_sample >= number of sites in lattice"

        # backend selection
        self.backend = self.exec_plan.build_backend()
        print(self.backend.plan.device_ids)

        # intialize environment to perform rate convergence analysis in
        self._build_rate_convergenc_env()

    
    def close_pool(self):
        """Call at end of script or when geometry/disorder/bath/seed changes."""
        if self._gpu_pool is not None:
            self._gpu_pool.close()
            self._gpu_pool = None

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
        

        # # freeze QDLattice; need to attach bath configuration manually for this procedure
        # self.qd_lattice_frozen = self.qd_lattice.to_frozen(self.bath_cfg)
        # print('succesfully froze QDLattice')

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

        # (4) Start GPU pool if requested in backend
        if self.backend.use_gpu and self._gpu_pool is None:
            self._gpu_pool = GpuRatePool(backend=self.backend)
            self._gpu_pool.start(self.geom, self.dis, self.bath_cfg, self.rnd_seed)


    # compute rate score in parallel (if available) or in serial
    def _rate_score(self, *args, **kwargs):
        #if self.tune_cfg.max_workers is None or self.tune_cfg.max_workers == 1:
        if not self.exec_plan.do_parallel:
            return self._rate_score_serial(*args, **kwargs)
        else:
            return self._rate_score_parallel(*args, **kwargs)

    
    # compute rate score serial (suited for both CPU/GPU)
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

    

    def _rate_score_parallel(self, theta_pol: float, theta_site: float, score_info: bool = True):


        weights_map = {int(i): float(w) for i, w in zip(self.start_sites, self.weights)}

        # Preferred: persistent GPU pool (reuses lattice on device; very low overhead)
        if self._gpu_pool is not None:
            lam_total, ns_total, np_total = self._gpu_pool.run_batches(
                self.start_sites, theta_pol, theta_site, self.tune_cfg.criterion, weights_map
            )
            info = {}
            if score_info and self.tune_cfg.no_samples > 0:
                info["ave_sites"] = ns_total / float(self.tune_cfg.no_samples)
                info["ave_pols"] = np_total / float(self.tune_cfg.no_samples)
            return lam_total, info

        # Fallback: CPU ProcessPool (shares parent lattice by FORK)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        global _QDLAT_GLOBAL
        _QDLAT_GLOBAL = self.qd_lattice

        ctx = mp.get_context("fork")
        jobs = [
            (int(idx), float(theta_pol), float(theta_site), self.tune_cfg.criterion,
             float(weights_map[int(idx)]))
            for idx in self.start_sites
        ]

        lam_sum = 0.0; nsites_sum = 0; npols_sum = 0
        max_workers = self.exec_plan.max_workers or os.cpu_count() or 1
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_rate_score_worker, job) for job in jobs]
            for fut in as_completed(futs):
                lam, ns, np_ = fut.result()
                lam_sum += lam; nsites_sum += ns; npols_sum += np_

        info = {}
        if score_info and self.tune_cfg.no_samples > 0:
            info["ave_sites"] = nsites_sum / float(self.tune_cfg.no_samples)
            info["ave_pols"] = npols_sum / float(self.tune_cfg.no_samples)
        return lam_sum, info



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

        # TODO : maybe put this somewhere else; we need this to close the GPU pool
        if self.backend.use_gpu:
            import time
            start = time.time()
            self.close_pool()
            end = time.time()
            print(f"time taking for stopping: {end-start:.6f}")

        
        return dict(theta_site=hi, theta_pol=tp_star, lambda_final=float(lam_star), 
                    nsites=int(info_star['ave_sites']), npols=int(info_star['ave_pols']))

    