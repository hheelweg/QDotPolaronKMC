import numpy as np
from numpy.random import default_rng
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import GeometryConfig, DisorderConfig, BathConfig, RunConfig
from . import const
from .hamiltonian_box import SpecDens
from .montecarlo import KMCRunner

# global variable to allow parallel workers to use the same QDLattice for convergence tests
_QDLAT_GLOBAL = None

# top-level worker for computing the rate scores from a single lattice site
# TODO: do we want to re-write this in terms of _rate_worker instead of _rate_score_worker?
def _rate_score_worker(args):
    (start_idx, theta_pol, theta_sites, criterion, weight) = args
    # Compute rates for this start index
    qd_lattice = _QDLAT_GLOBAL
    rates, final_sites, _, sel_info = KMCRunner._make_kmatrix_boxNEW(qd_lattice, start_idx,
                                                                     theta_pol=theta_pol, theta_sites=theta_sites,
                                                                     selection_info = True)
    
    # evaluate convergence criterion on rates vector
    if criterion == "rate-displacement":
        start_loc = qd_lattice.qd_locations[start_idx]                                                      # r(0)
        sq_displacments = ((qd_lattice.qd_locations[final_sites] - start_loc)**2).sum(axis = 1)             # ||Δr||^2 per destination
        lamda = (rates * sq_displacments).sum() / (2 * qd_lattice.geom.dims)
    else:
        raise ValueError("please specify valid convergence criterion for rates!")
    
    return lamda * weight, sel_info['nsites_sel'], sel_info['npols_sel']



class ConvergenceAnalysis(KMCRunner):

    def __init__(self, geom : GeometryConfig, dis : DisorderConfig, bath_cfg : BathConfig, run : RunConfig, no_samples : int):

        super().__init__(geom, dis, bath_cfg, run)
        # TODO : load some more class attributes here? (e.g. convergence criterion etc.)
        # NOTE : maybe also specify which algortihm type we use radial versus overlap cutoff?
        self.no_samples = no_samples

        # intialize environment to perform rate convergence analysis in
        self._build_rate_convergenc_env()
    

    def _build_rate_convergenc_env(self):

        # (0) build bath
        bath = SpecDens(self.bath_cfg.spectrum, const.kB * self.bath_cfg.temp)

        # (1) draw a lattice realization
        self.qd_lattice = self._build_grid_realization(bath, rid=0)

        # (2) produce no_samples starting indices from where to compute rate vectors
        ss_conv = self._ss_root.spawn(1)[0]
        rng_conv = default_rng(ss_conv)
        self.start_sites = rng_conv.integers(0, self.qd_lattice.geom.n_sites, size=self.no_samples)

        # (3) get Boltzmann weights for each start polaron in start_sites
        E = self.qd_lattice.full_ham.evals
        beta = self.qd_lattice.full_ham.beta
        w = np.exp(- beta * E[self.start_sites])
        Z = np.sum(np.exp(- beta * E))
        self.weights = w / Z

    
    # TODO : write input parameters so that we can also use this for r_hop/r_ove
    def _rate_score(self, theta_pol, theta_sites, 
                    criterion=None, score_info=False,
                    ):

        # get rates starting from each polaron starting index and analyze by criterion
        rates_criterion = None
        lambdas = np.zeros_like(self.start_sites, dtype=np.float32)
        nsites_sel, npols_sel = 0, 0
        for i, start_idx in enumerate(self.start_sites):

            # NEW way to obtain rates from theta_pol/theta_sites
            rates, final_sites, _, sel_info = KMCRunner._make_kmatrix_boxNEW(self.qd_lattice, start_idx,
                                                                             theta_sites=theta_sites,
                                                                             theta_pol=theta_pol,
                                                                             selection_info = True
                                                                             )
            
            # how many polarons/sites were selected 
            nsites_sel += sel_info['nsites_sel']
            npols_sel += sel_info['npols_sel']
            
            # evaluate convergence criterion on rates vector
            if criterion == "rate-displacement":
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
            info['ave_sites'] = nsites_sel / self.no_samples
            info['ave_pols'] = npols_sel / self.no_samples

        return rates_criterion, info


    # serial version to compute rate scores
    # TODO : write input parameters so that we can also use this for r_hop/r_ove
    def _rate_score_parallel(self, theta_pol, theta_sites, *,
                             criterion="rate-displacement", score_info = True,
                             max_workers=None):
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
        jobs = [(int(start_idx), float(theta_pol), float(theta_sites), criterion, weight_by_idx[start_idx]) 
                for start_idx in self.start_sites]

        rates_criterion = 0
        nsites_sel, npols_sel = 0, 0

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
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
            info['ave_sites'] = nsites_sel / self.no_samples
            info['ave_pols'] = npols_sel / self.no_samples

        return rates_criterion, info


    @staticmethod
    def _per_oct_gain(lam_from: float, lam_to: float, span_factor: float):
        rel = (lam_to - lam_from) / (abs(lam_from) + 1e-300)
        octaves = np.log(1.0 / max(span_factor, 1e-12)) / np.log(2.0)
        return rel / max(octaves, 1e-12)


    # --- inner: tune theta_pol at fixed theta_sites using fixed-span rule ---
    def _tune_theta_pol(
                        self,
                        theta_sites: float,
                        *,
                        theta_pol_start: float = 0.30,
                        theta_pol_min: float   = 0.02,
                        rho: float             = 0.7,
                        delta: float           = 0.015,
                        max_steps: int         = 8,
                        criterion: str         = "rate-displacement",
                        verbose                = True
                        ):
        tp = float(theta_pol_start)

        # evaluate Lambda at current theta_pol
        lam_from, _info = self._rate_score(tp, theta_sites, criterion=criterion, score_info=True)

        for _ in range(int(max_steps)):
            tp_next = max(float(theta_pol_min), rho * tp)
            if tp_next >= tp - 1e-15:
                break

            lam_to, _ = self._rate_score_parallel(tp_next, theta_sites, criterion=criterion, score_info=False, max_workers=8)

            # per-octave gain over a fixed span tp -> rho*tp
            gain = self._per_oct_gain(lam_from, lam_to, rho)

            # accept move
            tp, lam_from = tp_next, lam_to

            if verbose:
                print(f"[pol]  tp→{tp:.4f}  per-oct gain={gain*100:.2f}%/oct")

            # stop when additional tightening barely helps
            if gain < float(delta) or tp <= float(theta_pol_min) + 1e-12:
                break

        return tp, float(lam_from)

    # --- outer: bisection in log(theta_sites) using fixed-span rule; returns chosen thetas + Lambda ---
    def auto_tune_thetas(
                        self,
                        *,
                        theta_sites_lo: float = 0.10,   # loose (larger) starting value
                        theta_sites_hi: float = 0.01,   # tight (smaller) floor
                        theta_pol_start: float = 0.30,
                        theta_pol_min:   float = 0.02,
                        rho: float             = 0.7,   # fixed span to test gains
                        delta: float           = 0.015, # “plateau” target per-octave gain (≈1–2%)
                        max_outer: int         = 12,
                        criterion: str         = "rate-displacement",
                        verbose                = True
                    ):
        
        # ensure valid ordering
        lo = max(theta_sites_lo, theta_sites_hi)
        hi = min(theta_sites_lo, theta_sites_hi)
        if not (lo > hi):
            raise ValueError("Require theta_sites_lo > theta_sites_hi (looser > tighter).")

        # define a function: for a given theta_sites, tune theta_pol and return Lambda
        def eval_L(ts: float):
            tp_star, lam = self._tune_theta_pol(
                                                ts,
                                                theta_pol_start=theta_pol_start,
                                                theta_pol_min=theta_pol_min,
                                                rho=rho,
                                                delta=delta,
                                                criterion=criterion,
                                                )
            return lam, tp_star

        # helper: compute per-octave gain for sites by comparing ts and rho*ts (clamped at hi)
        def sites_gain(ts: float) -> tuple[float, float, float]:
            ts_tight = max(hi, rho * ts)
            lam_lo, _ = eval_L(ts)
            lam_hi, _ = eval_L(ts_tight)
            g = self._per_oct_gain(lam_lo, lam_hi, max(ts_tight / ts, 1e-12))
            return g, lam_lo, lam_hi

        # evaluate at ends (edge-case handling)
        g_lo, lam_lo, _ = sites_gain(lo)   # loose point gain toward tighter
        g_hi, _, lam_hi = sites_gain(hi)   # tight point gain toward even tighter (may be zero-span)

        # if even the tight end is still “steep”, return the tightest (best we can do)
        if g_hi > float(delta):
            tp_star, lam_star = self._tune_theta_pol(hi,
                                theta_pol_start=theta_pol_start,
                                theta_pol_min=theta_pol_min, rho=rho, delta=delta, criterion=criterion)
            return dict(theta_sites=hi, theta_pol=tp_star, lambda_final=float(lam_star))

        # if the loose end is already “flat”, keep the largest (cheapest) feasible theta_sites
        if g_lo <= float(delta):
            tp_star, lam_star = self._tune_theta_pol(lo,
                                theta_pol_start=theta_pol_start,
                                theta_pol_min=theta_pol_min, rho=rho, delta=delta, criterion=criterion)
            return dict(theta_sites=lo, theta_pol=tp_star, lambda_final=float(lam_star))

        # otherwise, bisection on log theta_sites to find largest theta with gain <= delta
        for _ in range(int(max_outer)):
            mid = float(np.sqrt(lo * hi))  # geometric midpoint (bisection in log-space)
            g_mid, _, _ = sites_gain(mid)

            if g_mid > float(delta):
                # still steep at mid → move tighter
                lo = mid
            else:
                # flat enough at mid → keep it as new “hi” (feasible)
                hi = mid
            
            if verbose:
                print(f"[sites] lo={lo:.4f} hi={hi:.4f} mid={mid:.4f} gain(mid)={g_mid*100:.2f}%/oct")

            if lo / hi <= 1.10:  # bracket within ~10% is enough
                break

        # finalize at hi (largest theta_sites in bracket with gain <= delta)
        tp_star, lam_star = self._tune_theta_pol_span(hi,
                            theta_pol_start=theta_pol_start,
                            theta_pol_min=theta_pol_min, rho=rho, delta=delta, criterion=criterion)
        
        return dict(theta_sites=hi, theta_pol=tp_star, lambda_final=float(lam_star))

    