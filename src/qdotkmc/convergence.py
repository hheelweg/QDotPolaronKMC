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

# top-level worker for computing the rates from a single lattice site
def _rate_worker(args):
    (start_idx, theta_pol, theta_sites) = args
    # Compute rates for this start index
    rates, final_sites, _, sel_info = KMCRunner._make_kmatrix_boxNEW(_QDLAT_GLOBAL, start_idx,
                                                                     theta_sites, theta_pol,
                                                                     selection_info = True)
    return start_idx, rates, final_sites, sel_info



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

        # dispatch configs + indices to parallelize over
        jobs = [(int(start_idx), float(theta_pol), float(theta_sites)) for start_idx in self.start_sites]

        rates_criterion = None
        nsites_sel, npols_sel = 0, 0

        # # Weighted sum uses the Boltzmann weights you precomputed per start index.
        # # Build a dict for O(1) lookup.
        # weight_by_idx = {int(i): float(w) for i, w in zip(self.start_sites, self.weights)}

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_rate_worker, job) for job in jobs]
            for fut in as_completed(futs):

                # (1) let worker obtain rates etc.
                start_idx, rates, final_sites, sel_info = fut.result()

                # (2) post-processing of information from _rate_worker to obtain scores etc.
                # (2.1) how many polarons/sites were selected 
                nsites_sel += sel_info['nsites_sel']
                npols_sel += sel_info['npols_sel']
                # (2.2) evaluate convergence criterion on rates vector
                if criterion == "rate-displacement":
                    start_loc = self.qd_lattice.qd_locations[start_idx]                                                      # r(0)
                    sq_displacments = ((self.qd_lattice.qd_locations[final_sites] - start_loc)**2).sum(axis = 1)             # ||Δr||^2 per destination
                    lamda = (rates * sq_displacments).sum() / (2 * self.qd_lattice.geom.dims)
                    rates_criterion = (self.weights * lamda).sum()
                else:
                    raise ValueError("please specify valid convergence criterion for rates!")

        # optional : store additional information
        info = {}
        if score_info:
            info['ave_sites'] = nsites_sel / self.no_samples
            info['ave_pols'] = npols_sel / self.no_samples

        return rates_criterion, info


        

    def _tune_theta_pol_simple(
                                self,
                                *,
                                theta_sites: float,
                                theta_pol_start: float = 0.30,
                                theta_pol_min: float = 0.02,
                                shrink: float = 0.7,              # fixed geometric shrink per step
                                delta_pol: float = 0.015,         # stop when per-octave ΔΛ/Λ < delta_pol
                                max_steps: int = 10,
                                criterion: str = "rate-displacement",
                              ):
        """
        Shrink theta_pol by a fixed factor until Λ stops changing materially
        (measured as per-octave relative gain). Uses ONLY self._rate_score(...).
        Returns dict with theta_pol, Lambda, cost proxy, and avg sizes.
        """
        def eval_score(tp: float):
            lam, info = self._rate_score(tp, theta_sites,
                                         criterion=criterion, score_info=True)
            # cheap cost proxy: dense-J ~ n^2 * P
            n = float(info.get('ave_sites', 1.0)) or 1.0
            P = float(info.get('ave_pols',  1.0)) or 1.0
            cost = n*n * P
            return float(lam), info, float(cost)

        tp = float(theta_pol_start)
        lam_prev, info_prev, cost_prev = eval_score(tp)

        for _ in range(int(max_steps)):
            tp_next = max(float(theta_pol_min), float(shrink) * tp)
            if tp_next >= tp - 1e-15:  # no progress possible
                break

            lam_next, info_next, cost_next = eval_score(tp_next)

            # per-octave relative gain (step-size robust)
            rel_gain = (lam_next - lam_prev) / (abs(lam_prev) + 1e-300)
            span_oct = np.log(max(1e-12, tp / tp_next)) / np.log(2.0)
            per_oct_gain = rel_gain / max(span_oct, 1e-12)

            # accept step
            tp, lam_prev, info_prev, cost_prev = tp_next, lam_next, info_next, cost_next

            if per_oct_gain < float(delta_pol) or tp <= float(theta_pol_min) + 1e-12:
                break

        return {
            "theta_pol": tp,
            "Lambda": lam_prev,
            "cost": cost_prev,
            "avg_sites": float(info_prev.get('ave_sites', np.nan)),
            "avg_pols": float(info_prev.get('ave_pols',  np.nan)),
        }
    

    def auto_tune_thetas_simple(
                                self,
                                *,
                                theta_sites_grid=(0.10, 0.07, 0.05, 0.035, 0.025),
                                theta_pol_start: float = 0.30,
                                theta_pol_min: float = 0.02,
                                shrink_pol: float = 0.7,
                                delta_pol: float = 0.015,          # per-octave ΔΛ/Λ stop for pol tuning
                                delta_octave: float = 0.015,       # knee threshold along sites (per-octave slope)
                                criterion: str = "rate-displacement",
                                analytics: bool = True,
                                ):
        """
        1) For each theta_sites on a small geometric grid, shrink theta_pol until Λ saturates.
        2) Compute per-octave slopes of Λ between neighbors and pick the first
           theta_sites where slope < delta_octave ("flat" region).
        Returns chosen (theta_sites, theta_pol), Λ, cost proxy, and a small diagnostics dict.
        """
        # (1) evaluate Λ for each theta_sites with minimal pol tuning
        rows = []
        for ts in map(float, theta_sites_grid):
            res = self._tune_theta_pol_simple(
                theta_sites=ts,
                theta_pol_start=float(theta_pol_start),
                theta_pol_min=float(theta_pol_min),
                shrink=float(shrink_pol),
                delta_pol=float(delta_pol),
                criterion=criterion,
            )
            rows.append({"theta_sites": ts, **res})

        # (2) per-octave slopes between neighbors; pick first slope < delta_octave
        th = np.array([r["theta_sites"] for r in rows], float)
        La = np.array([r["Lambda"]      for r in rows], float)
        Co = np.array([r["cost"]        for r in rows], float)

        assert np.all(th[:-1] > th[1:]), "theta_sites_grid must be strictly decreasing"

        slopes = []
        for i in range(len(th) - 1):
            dLa = La[i+1] - La[i]
            span_oct = np.log(th[i] / th[i+1]) / np.log(2.0)
            slopes.append((dLa / max(abs(La[i]), 1e-300)) / max(span_oct, 1e-12))
        slopes = np.array(slopes)

        if np.any(slopes < float(delta_octave)):
            idx = int(np.argmax(slopes < float(delta_octave)))  # first time we flatten
        else:
            idx = len(th) - 2  # never flattened; pick tightest bracket

        # Choose the cheaper of the two endpoints if Λ is ~equal (within 1%)
        i_best = idx if La[idx] >= La[idx+1] else idx+1
        j_other = idx+1 if i_best == idx else idx
        if abs(La[i_best] - La[j_other]) / (abs(La[i_best]) + 1e-300) <= 0.01:
            i_best = idx+1 if Co[idx+1] < Co[idx] else idx

        best = rows[i_best]

        out = dict(
            theta_sites=float(best["theta_sites"]),
            theta_pol=float(best["theta_pol"]),
            lambda_final=float(best["Lambda"]),
            cost_final=float(best["cost"]),
        )

        if analytics:
            out["diagnostics"] = dict(
                stageA_curve=[dict(theta_sites=float(r["theta_sites"]),
                                   theta_pol=float(r["theta_pol"]),
                                   Lambda=float(r["Lambda"]),
                                   cost=float(r["cost"]),
                                   avg_sites=float(r["avg_sites"]),
                                   avg_pols=float(r["avg_pols"]))
                              for r in rows],
                slopes=slopes.tolist(),
                grid=list(map(float, theta_sites_grid)),
                settings=dict(
                    delta_pol=float(delta_pol),
                    delta_octave=float(delta_octave),
                    theta_pol_start=float(theta_pol_start),
                    theta_pol_min=float(theta_pol_min),
                    shrink_pol=float(shrink_pol),
                ),
            )
        return out