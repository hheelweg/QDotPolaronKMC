
import numpy as np
from . import const
from . import utils
from qdotkmc.backend import get_backend
import time
from typing import Optional, List
import os

class Redfield():

    def __init__(self, hamiltonian, polaron_locations, site_locations, kappa, backend,
                time_verbose = True):

        self.ham = hamiltonian
        self.polaron_locations = polaron_locations                  # polaron locations (global frame)
        self.site_locations = site_locations                        # site locations (global frame)
        self.kappa = kappa                                          # kappa-polaron for polaron transformation
        self.time_verbose = time_verbose                            # set to true only when time to compute rates is desired

        # ---- physics caches -----
        # bath-correlation cache
        self._corr_cache = {}                                       # key: (lam, self.kappa, center_global) -> dict[int -> complex]
        # avoid recomputing J2 = J * J by caching
        self._J2_cache = {}                                         # key: tuple(site_g) -> J2 ndarray
        # cache for |U|^2
        self._W_abs2 = None                                         # cache |U|^2 (n, P) float64, C-contig

        # ----- GPU cache slots (for select_by_weight) -----
        self._Wg = None
        self._L2g = None
        self._gpu_cache_key = None                                  # (n, P, id(W_host), id(L2_host))

        # --- load backend (GPU/CPU) ---
        self.backend = backend                                       # keep the handle if you want helper methods later
        self.xp = backend.xp                                         # numpy or cupy                  

        # Configure cupy memory pools (no-op on CPU)
        if hasattr(self.backend, "setup_pools"):
            self.backend.setup_pools()

        # print which backend we end up using
        if self.time_verbose:
            mode = "GPU" if self.backend.use_gpu else "CPU"
            print(f"[Redfield] backend: {mode}  (use_c64={self.backend.gpu_use_c64})")

    
    def _get_W_abs2(self):
        """
        Return cached matrix W = |U|^2 (site × polaron weights).

        -   U is the site2polaron transformation matrix (n × P).
        -   Each entry W[m,ν] = |U[m,ν]|^2 gives the probability
            weight of polaron ν on site m.
        -   This is reused in many selections, so we cache it in
            self._W_abs2 after first computation.

        Returns
        -------
        W : ndarray, shape (n, P), float64, C-contiguous
            Cached weights matrix.
        """
        if self._W_abs2 is None:
            U = self.ham.Umat
            W = np.abs(U)**2                 # real
            self._W_abs2 = np.asarray(W, dtype=np.float64, order='C')
        return self._W_abs2
    
    def _get_L2(self):
        """
        Return cached full-system |J|^2 matrix (site-site couplings squared).

        - J is the inter-site coupling matrix (n × n).
        - L2 = |J|^2 (elementwise squared coupling strengths).
        - This is used to propagate site weights into effective
        contact scores (S) and exchange scores (s).
        - Heavy to compute repeatedly, so we call through
        self._get_J2_cached(...) and cache the result.

        Returns
        -------
        L2 : ndarray, shape (n, n), float64, C-contiguous
            squared coupling matrix.
        """
        Ns = self.ham.Umat.shape[0]
        L2 = self._get_J2_cached(self.ham.J_dense, np.arange(Ns))
        return np.asarray(L2, dtype=np.float64, order="C")
        
    def _ensure_WL2_gpu(self):
            """
            Ensure W and L2 are uploaded and cached on the GPU.

            - Converts host-side W = |U|^2 and L2 = |J|^2 (float64, C-contig) to CuPy,
            caching device arrays to avoid re-uploading every call.
            - A cache key based on shapes and host buffer addresses is used to detect
            when the lattice has changed and a refresh is needed.

            Returns
            -------
            Wg : cupy.ndarray, shape (n, P), float64, C-contiguous
                Device-resident |U|^2.
            L2g : cupy.ndarray, shape (n, P), float64, C-contiguous
                Device-resident |J|^2.
            n, P : int
                Dimensions of the weights matrix.
            """
            # NOTE: this should call your *existing* cached provider:
            Wh  = self._get_W_abs2()
            L2h = self._get_L2()

            Ns, Np = Wh.shape
            key = (Ns, Np, 
                int(Wh.__array_interface__['data'][0]), 
                int(L2h.__array_interface__['data'][0]))
            
            if self._gpu_cache_key != key:
                # set allocators once (safe to re-call)
                try:
                    self.xp.cuda.set_allocator(self.xp.cuda.MemoryPool().malloc)
                    self.xp.cuda.set_pinned_memory_allocator(self.xp.cuda.PinnedMemoryPool().malloc)
                except Exception:
                    pass
                # upload fresh copies to device
                self._Wg = self.xp.asarray(Wh,  dtype=self.xp.float64, order="C")
                self._L2g = self.xp.asarray(L2h, dtype=self.xp.float64, order="C")
                self._gpu_cache_key = key

            return self._Wg, self._L2g, Ns, Np
    

    def _top_prefix_by_coverage_cpu(self, 
                                    values: np.ndarray, 
                                    keep_fraction: float) -> np.ndarray:
        """
        greedy coverage selector (CPU version).

        Given a vector of nonnegative scores v, return the smallest
        prefix set of indices such that the cumulative sum reaches
        'keep_fraction' of the total sum.

        - Used in select_by_weight to:
            (1) pick the minimal set of destination polarons (θ_pol),
            (2) pick the minimal set of important sites (θ_site).
        - Avoids full sort: grows candidate set geometrically and
        refines until target coverage is reached.

        Parameters
        ----------
        values : ndarray
            Input score vector (1D, nonnegative).
        keep_fraction : float
            Fraction of total sum to cover, in [0,1].

        Returns
        -------
        indices : ndarray[int]
            Indices of selected entries (order determined by descending scores).
        """
        v = np.asarray(values, dtype=np.float64)
        total = float(v.sum())
        if total <= 0.0:
            return np.empty(0, dtype=np.intp)
        target = keep_fraction * total

        vmax = float(v.max(initial=0.0))
        if vmax <= 0.0:
            return np.empty(0, dtype=np.intp)

        # lower bound for k; grow geometrically until coverage reached
        k = max(1, int(target / vmax))
        k = min(k, v.size)

        while True:
            # select top-k by value (unsorted)
            topk_idx = np.argpartition(v, v.size - k)[-k:]        
            topk_vals = v[topk_idx]

            # sort that small subset descending
            order_local = np.argsort(-topk_vals)                  
            idx_sorted = topk_idx[order_local]

            # cumulative coverage
            csum = np.cumsum(topk_vals[order_local])
            pos = np.searchsorted(csum, target, side='left')

            # if target covered, return minimal prefix
            if pos < k:
                return idx_sorted[:pos+1].astype(np.intp)
            
            # if we already considered everything, just return all
            if k == v.size:
                return idx_sorted.astype(np.intp)
            
            # otherwise grow k geometrically and retry
            k = min(v.size, int(k*1.8)+1)
    
    def _top_prefix_by_coverage_gpu(self, values : np.ndarray,
                                    keep_fraction: float):
        """
        Greedy coverage selector (GPU/CuPy version).

        Same semantics as _top_prefix_by_coverage_cpu, but keeps all math on device:
        - Grow a candidate top-k using argpartition.
        - Sort only that subset.
        - Return the minimal prefix whose cumulative sum reaches target coverage.

        Parameters
        ----------
        values : array_like (CuPy or NumPy)
            Score vector (1D). Will be converted to a CuPy float array.
        keep_fraction : float
            Fraction of total sum to cover, in [0, 1].

        Returns
        -------
        indices : np.ndarray[int]
            Selected indices (NumPy intp). We convert to host at the end because
            downstream indexing typically expects NumPy indices.
        """
        # normalize to 1D, float, sanitize NaNs/Infs so cumsum is well-defined
        v = self.xp.asarray(values).ravel()
        if v.dtype.kind != 'f':
            v = v.astype(self.xp.float64, copy=False)
        v = self.xp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        tot = float(self.xp.sum(v).get())
        if tot <= 0.0:
            return np.empty(0, dtype=np.intp)

        target = float(keep_fraction) * tot
        vmax = float(self.xp.max(v).get())
        if vmax <= 0.0:
            return np.empty(0, dtype=np.intp)

        n = int(v.size)
        k = max(1, int(target / vmax)); k = min(k, n)

        while True:
            # unsorted top-k on device
            idx_topk = self.xp.argpartition(v, n - k)[-k:]
            vals     = v[idx_topk]

            # sort only the candidate set (deterministic descending)
            order    = self.xp.argsort(-vals)
            idx_sorted = idx_topk[order]
            csum = self.xp.cumsum(vals[order])

            # cp.searchsorted expects a device array 'v', not a python scalar
            target_dev = self.xp.asarray([target], dtype=csum.dtype)   
            pos_dev = self.xp.searchsorted(csum, target_dev, side="left") 
            pos = int(pos_dev.get()[0])

            # minimal prefix found, or we've already taken all
            if pos < k or k == n:
                return np.asarray(idx_sorted[:pos+1].get(), dtype=np.intp)
            
            # otherwise enlarge k geometrically and retry
            k = min(n, int(k*1.8)+1)
    

    # bath half-Fourier Transforms
    # K_λ(ω) in Eq. (15)
    def _corr_row(self, lam, center_global, pol_g):
        key = (float(lam), float(self.kappa), int(center_global))
        row_cache = self._corr_cache.get(key)
        if row_cache is None:
            row_cache = {}
            self._corr_cache[key] = row_cache

        # gather missing indices
        need = [int(i) for i in pol_g if int(i) not in row_cache]
        if need:
            omega = self.ham.omega_diff[need, int(center_global)]
            vals  = self.ham.spec.correlationFT(omega, lam, self.kappa)  # vectorized
            for i, v in zip(need, vals):
                row_cache[i] = v

        # assemble in pol_g order
        return np.array([row_cache[int(i)] for i in pol_g], dtype=np.complex128)


    # compute J2 := J * J
    def _get_J2_cached(self, J, site_g):
        key = tuple(map(int, site_g))
        J2 = self._J2_cache.get(key)
        if J2 is None or J2.shape != J.shape:
            J2 = (J * J).astype(np.float64, copy=False)
            self._J2_cache[key] = J2
        return J2

    
    # selection of sites, polarons for rates based on RADIUS
    # NOTE : this is our known implementation (similar to Kassal)
    def select_by_radius(self, center_global: int, *,
                         r_hop: int, 
                         r_ove: int,
                         pol_idxs_global: List[int], 
                         site_idxs_global: List[int],
                         periodic = False, 
                         grid_dims = None
                         ):
        """
        Returns subsets pol_g ⊆ pol_idxs_global and site_g ⊆ site_idxs_global
        that are within r_hop / r_ove (ina actual units) of center_global, preserving order.
        Also returns center_local (the position of center_global inside pol_g).

        Assumes:
        - self.polaron_locations are absolute coordinates (shape: [N, D])
        - self.ham.qd_lattice_rel are site coordinates in the same frame
        """

        pol_idxs_global  = np.asarray(pol_idxs_global,  dtype=np.intp)
        site_idxs_global = np.asarray(site_idxs_global, dtype=np.intp)

        # center position in the same frame used for distances
        center_coord = self.polaron_locations[int(center_global)]

        # periodic minimum-image displacement (vectorized)
        def _dist(pts):
            pts = np.atleast_2d(pts)
            disp = pts - center_coord
            if periodic:
                if grid_dims is None:
                    raise ValueError("grid_dims must be provided when periodic=True.")
                L = np.asarray(grid_dims, float)
                disp = disp - np.round(disp / L) * L
            return np.linalg.norm(disp, axis=1)

        # distances for *subset* (keep original order with boolean masks)
        dpol  = _dist(self.polaron_locations[pol_idxs_global])
        dsite = _dist(self.site_locations[site_idxs_global])

        keep_pol_mask  = (dpol  < r_hop)
        keep_site_mask = (dsite < r_ove)

        pol_g  = pol_idxs_global[keep_pol_mask]
        site_g = site_idxs_global[keep_site_mask]

        # center must remain inside pol_g (since it came from the box)
        assert center_global in pol_g, (
            "Center_global not in pol_idxs_global after box selection. "
            "Ensure the box always includes the center polaron."
        )

        return pol_g, site_g
 
    # selection of sites, polarons for rates based on WEIGHT
    def select_by_weight(self, center_global: int, *,
                     theta_site: float, 
                     theta_pol: float,
                     max_nuprime: Optional[int] = None,
                     verbose: bool = False):
        """
        Physics-aware, two-parameter selector for a single center polaron ν.
        Automatically switched between GPU and CPU executaion based on resources and backend

        What it does (in order):
        1) Rank destination polarons ν' by the J-aware contact score
                S_{ν→ν'} = w_ν^T |J|^2 w_{ν'} ,  with  w_α[m] = |U_{mα}|^2 .
            Keep the smallest set whose cumulative S reaches (1 - theta_pol) of total S.
            -> This defines pol_g = [ν] + kept ν'.

        2) Build a site-importance (exchange) score relative to the kept destinations:
                s_i = w_ν[i] * (|J|^2 w_D)[i] + w_D[i] * (|J|^2 w_ν)[i],
            where w_D = sum_{ν' in kept} w_{ν'}.
            Keep the smallest site set reaching (1 - theta_sites) of sum(s).
            -> This defines site_g.

        Notes:
        - Uses a cached full L2 = |J|^2 via _get_J2_cached().
        - No bath work here; _corr_row is only used later in make_redfield_box.
        - As θ_* to 0, selection monotonically approaches the full Redfield result.

        Parameters
        ----------
        theta_sites : float
            Coverage tolerance for sites. Smaller values keep MORE sites (higher cost, higher fidelity).
        theta_pol : float
            Coverage tolerance for destination polarons. Smaller values keep MORE ν' (linear cost).
        max_nuprime : Optional[int]
            Optional hard cap on the number of ν' considered after ranking by S (useful for speed).
        verbose : bool
            Print small diagnostics.

        Returns
        -------
        site_g : np.ndarray[int]
            Global site indices selected by exchange-score coverage.
        pol_g : np.ndarray[int]
            Global polaron indices with the center FIRST, followed by kept destinations.
        """
        if self.backend.use_gpu:
            return self.select_by_weight_gpu(center_global,
                                            theta_site=theta_site,
                                            theta_pol=theta_pol,
                                            max_nuprime=max_nuprime,
                                            verbose=verbose)
        else:
            return self.select_by_weight_cpu(center_global,
                                            theta_site=theta_site,
                                            theta_pol=theta_pol,
                                            max_nuprime=max_nuprime,
                                            verbose=verbose)

    # CPU version to select weights
    def select_by_weight_cpu(self, center_global: int, *,
                     theta_site: float, theta_pol: float,
                     max_nuprime: Optional[int] = None,
                     verbose: bool):
        """
        CPU version of weight-based selection.

        Steps:
        ------
        1. Destination polaron selection (θ_pol):
        - Define contact scores S_{ν→ν'} = w_ν^T |J|^2 w_{ν'}.
        - Use greedy coverage (_top_prefix_by_coverage_cpu) to
            keep just enough ν' to reach (1 - θ_pol) of total S.
        - Ensure center ν is always included.

        2. Site selection (θ_site):
        - Aggregate destination weight w_D = Σ_{ν'∈ kept} w_{ν'}.
        - Compute exchange score per site:
            s_i = w_ν[i]*(L2 w_D)[i] + w_D[i]*(L2 w_ν)[i]
        - Again use greedy coverage to keep minimal set reaching
            (1 - θ_site) of total s.

        Notes:
        ------
        - Uses cached W = |U|^2 and L2 = |J|^2 for efficiency.
        - As θ_pol, θ_site to 0, this approaches the full Redfield set.

        Parameters
        ----------
        center_global : int
            Index of source polaron ν.
        theta_site : float
            Site coverage tolerance (smaller = more sites kept).
        theta_pol : float
            Destination coverage tolerance (smaller = more ν' kept).
        max_nuprime : int, optional
            Hard cap on number of ν' considered (after ranking).
        verbose : bool
            Print diagnostic info.

        Returns
        -------
        site_g : ndarray[int]
            Selected site indices.
        pol_g : ndarray[int]
            Selected polaron indices, with center first.
        """

        nu = int(center_global)
        W  = self._get_W_abs2()                 # (n, P) float64
        L2 = self._get_L2()                     # (n, n) float64, C-contig
        n, P = W.shape

        # (1) ν' selection by S coverage
        w0 = W[:, nu]                           # (n,)
        t0 = L2 @ w0                            # (n,)
        S  = W.T @ t0                           # (P,)
        S[nu] = 0.0                             # exclude self

        kept = self._top_prefix_by_coverage_cpu(S, 1.0 - float(theta_pol))
        kept = kept[kept != nu]
        # optional: pre-cap by max_nuprime to reduce coverage work on long tails
        if max_nuprime is not None and kept.size > max_nuprime:
            kept = kept[np.argsort(-S[kept])[:max_nuprime]]
        # keep descending order for determinism
        kept = kept[np.argsort(-S[kept])]
        pol_g = np.concatenate(([nu], kept)).astype(np.intp)

        # (2) site selection by s coverage
        if kept.size:
            wD = W[:, kept].sum(axis=1)         # (n,)
            tD = L2 @ wD
            s  = w0 * tD + wD * t0
        else:
            s  = np.zeros(n, dtype=np.float64)

        if float(s.sum()) <= 0.0:
            site_set = set(utils._mass_core_by_theta(w0, theta_site).tolist())
            for j in kept:
                site_set |= set(utils._mass_core_by_theta(W[:, j], theta_site).tolist())
            site_g = np.array(sorted(site_set), dtype=np.intp)
            if verbose:
                print(f"[select] s_sum = 0 fallback; sites={site_g.size}")
            return site_g, pol_g

        kept_sites = self._top_prefix_by_coverage_cpu(s, 1.0 - float(theta_site))
        site_g = np.sort(kept_sites.astype(np.intp))

        if verbose:
            cov_pol   = S[kept].sum() / (S.sum() + 1e-300)
            cov_sites = s[site_g].sum() / (s.sum() + 1e-300)
            print(f"[select] nu' kept: {len(kept)}/{P-1}  S-coverage={cov_pol:.3f}")
            print(f"[select] sites kept: {site_g.size}/{n}  s-coverage={cov_sites:.3f}")

        return site_g, pol_g

    # GPU version to select weights
    def select_by_weight_gpu(self, center_global: int, *,
                     theta_site: float, theta_pol: float,
                     max_nuprime: Optional[int] = None,
                     verbose: bool):
        
        """
        GPU version of weight-based selection (CuPy).

        Steps (identical physics to CPU):
        1) Destination polaron selection (θ_pol):
        - S_{ν→ν'} = w_ν^T |J|^2 w_{ν'}  with w_α = |U|^2[:, α]
        - Keep smallest set of ν' whose cumulative S reaches (1 - θ_pol).
        2) Site selection (θ_site):
        - w_D = Σ_{ν'∈kept} |U|^2[:,ν']
        - s_i = w_ν[i]*(L2 w_D)[i] + w_D[i]*(L2 w_ν)[i]
        - Keep smallest site set reaching (1 - θ_site).

        GPU-specific notes:
        - Uses _ensure_WL2_gpu() to keep W=|U|^2 and L2=|J|^2 resident on device.
        - Minimizes host<>device copies (one cp.asnumpy of S and s for diagnostics/ordering).
        - Chooses between a column-gather sum and a masked GEMV for w_D depending on kept size.

        Returns
        -------
        site_g : np.ndarray[int]
            Selected site indices.
        pol_g : np.ndarray[int]
            Selected polaron indices, with center first.
        """

        nu = int(center_global)
        # ensure W (n × P) and L2 (n × n) on the GPU; reuse if unchanged
        W, L2, Ns, Np = self._ensure_WL2_gpu()  

        # (1) ν' selection
        w0 = W[:, nu]               # (n,)
        t0 = L2 @ w0                # (n,)
        S  = W.T @ t0               # (P,)
        S[nu] = 0.0                 # exclude self

        # one host copy for ordering/diagnostics; heavy math remains on device
        S_host = self.xp.asnumpy(S)  
        # optional: pre-cap by max_nuprime to reduce coverage work on long tails
        if max_nuprime is not None and max_nuprime < Np-1:
            cand = self.xp.argpartition(S, -(max_nuprime+1))[-(max_nuprime+1):]
            cand = self.xp.asnumpy(cand)
            cand = cand[cand != nu]
            cand = cand[np.argsort(-S_host[cand])]
            S_view = S_host[cand]
            csum = np.cumsum(S_view)
            k_pol = int(np.searchsorted(csum, (1.0 - float(theta_pol))*S_view.sum(), 'left')) + 1
            kept = cand[:k_pol]
        else:
            # device-side coverage (returns np indices), then drop self if present
            kept = self._top_prefix_by_coverage_gpu(S, 1.0 - float(theta_pol))
            kept = kept[kept != nu]

        # deterministic descending order by S (host)
        kept = kept[np.argsort(-S_host[kept])]
        pol_g = np.concatenate(([nu], kept)).astype(np.intp)

        # (2) site selection on device
        k = kept.size
        if k == 0:
            wD = self.xp.zeros(Ns, dtype=self.xp.float64)                 # (n,)
        elif k < (Np // 16):
            # for a small kept set, gathering columns is efficient
            wD = W[:, kept].sum(axis=1)                         # (n,)     
        else:
            # for a large kept set, masked GEMV is more coalesced/faster
            mask = self.xp.zeros((Np,), dtype=self.xp.float64)
            mask[kept] = 1.0
            wD = W @ mask                                       # (n,)            

        tD = L2 @ wD
        s  = w0 * tD + wD * t0

        s_host = self.xp.asnumpy(s)
        if float(s_host.sum()) <= 0.0:
            site_set = set(utils._mass_core_by_theta(self.xp.asnumpy(W)[:, nu], theta_site).tolist())
            for j in kept:
                site_set |= set(utils._mass_core_by_theta(self.xp.asnumpy(W)[:, j], theta_site).tolist())
            site_g = np.array(sorted(site_set), dtype=np.intp)
            if verbose:
                print(f"[select] s_sum = 0 fallback; sites={site_g.size}")
            return site_g, pol_g

        kept_sites = self._top_prefix_by_coverage_gpu(s, 1.0 - float(theta_site))
        site_g = np.sort(kept_sites.astype(np.intp))

        if verbose:
            cov_pol   = S_host[kept].sum() / (S_host.sum() + 1e-300)
            cov_sites = s_host[site_g].sum() / (s_host.sum() + 1e-300)
            print(f"[select] nu' kept: {len(kept)}/{Np-1}  S-coverage={cov_pol:.3f}")
            print(f"[select] sites kept: {site_g.size}/{Ns}  s-coverage={cov_sites:.3f}")

        return site_g, pol_g


    # function for computing 𝛾_+(𝜈') (exact, closed-form λ-contraction)
    # (a) for execution on CPU (np-based)
    @staticmethod
    def _build_gamma_plus_cpu(J, J2, Up, u0, bath_map):
        n, P = Up.shape
        Upc  = Up.conj()

        # shared matmuls
        Ju0 = J @ u0            # (n,)
        JUp = J @ Up            # (n,P)

        # row/col sums
        rowR = (u0.conj()[:, None]) * JUp          # (n,P)
        colR = (Ju0.conj()[:, None]) * Up          # (n,P)
        rowC = Upc * Ju0[:, None]                  # (n,P)
        colC = (u0[:, None]) * JUp.conj()          # (n,P)  # uses conj(JUp), avoids JUpc matmul

        # this builds R_{ab,𝜈'} = J_{ab} conj(U_{a𝜈}) U_{b𝜈'}
        # and C_{ab,𝜈'} = J_{ab} conj(U_{a𝜈'}) U_{b𝜈}

        # compute the full unconstrained sum T0
        sum_rowR = rowR.sum(axis=0)                # (P,)
        sum_rowC = rowC.sum(axis=0)                # (P,)
        T0  = sum_rowR * sum_rowC
        # sums with one equality enforced (i.e. Tac has a=c, etc.)
        Tac = (rowR * rowC).sum(axis=0)            # ac
        Tbd = (colR * colC).sum(axis=0)            # bd
        Tad = (rowR * colC).sum(axis=0)            # ad
        Tbc = (colR * rowC).sum(axis=0)            # bc

        # build Tpair (a = c & b = d) and (a = d & c = b)
        # pair buckets via one matmul + a couple of elementwise ops
        U0Up = u0[None, :] * Up.T                   # (P,n)   target vector batch (per p)
        Z    = (J2 @ U0Up.T)                        # (n,P)
        V    = u0.conj()[:, None] * Upc             # (n,P)
        Tpair  = (V * Z).sum(axis=0)                # (P,)  -> ac & bd

        Au0    = np.abs(u0)**2                      # (n,)
        AUp    = (np.abs(Up)**2)                    # (n,P)
        t_b    = J2 @ Au0                           # (n,)
        Tcross = (AUp.T @ t_b)                      # (P,)  -> ad & bc

        # renaming
        E_ac, E_bd, E_ad, E_bc = Tac, Tbd, Tad, Tbc
        E_acbd  = Tpair
        E_adbc  = Tcross
        # compute terms H_λ(𝜈') that are contracted with each K_λ(ω_{𝜈'𝜈}), i.e.
        # one term for each λ ∈ {-2,-1,0,1,2}, included Möbius inclusion-exclusion
        H2   = E_acbd
        Hm2  = E_adbc
        H1   = E_ac + E_bd - 2.0 * E_acbd
        Hm1  = E_ad + E_bc - 2.0 * E_adbc
        H0   = T0 - (H2 + Hm2 + H1 + Hm1)

        # compute 𝛾_+(𝜈') = ∑_λ K_λ(ω_{𝜈'𝜈})H_λ(𝜈') based on Eq. (14)
        # with ∑ over λ. 
        return (bath_map[-2.0] * Hm2
                + bath_map[-1.0] * Hm1
                + bath_map[ 0.0] * H0
                + bath_map[ 1.0] * H1
                + bath_map[ 2.0] * H2)

    # function for computing 𝛾_+(𝜈') (exact, closed-form λ-contraction)
    # (b) for execution on GPU (cp-based)
    @staticmethod
    def _build_gamma_plus_gpu(J, J2, Up, u0, bath_map, xp, *, use_c64=False):

        # dtypes (stay in 128-bit by default for accuracy)
        cupy_c = xp.complex64 if use_c64 else xp.complex128
        cupy_f = xp.float32   if use_c64 else xp.float64

        # upload once per call (minimal change version)
        Jg  = xp.asarray(J,  dtype=cupy_f)
        J2g = xp.asarray(J2, dtype=cupy_f)
        Upg = xp.asarray(Up, dtype=cupy_c)
        u0g = xp.asarray(u0, dtype=cupy_c)

        # shared matmuls
        Ju0 = Jg @ u0g           # (n,)
        JUp = Jg @ Upg           # (n,P)

        # T0 via two GEMV
        sum_rowR = JUp.T @ xp.conj(u0g)         # (P,)
        sum_rowC = xp.conj(Upg).T @ Ju0         # (P,)
        T0 = sum_rowR * sum_rowC

        # one-equality sums (Hadamard + reductions)
        Tac = xp.einsum('i,ip,ip->p', Ju0*xp.conj(u0g), JUp, xp.conj(Upg), optimize=True)
        Tbd = xp.einsum('i,ip,ip->p', xp.conj(Ju0)*u0g, Upg, xp.conj(JUp), optimize=True)

        Tad = (xp.abs(JUp)**2).T @ (xp.abs(u0g)**2)
        Tbc = (xp.abs(Upg)**2).T @ (xp.abs(Ju0)**2)


        # pair terms
        Y = u0g[:, None] * Upg                 # (n,P)
        Z = J2g @ Y                            # (n,P)
        X = xp.conj(u0g)[:, None] * xp.conj(Upg)
        E_acbd = xp.einsum('ip,ip->p', X, Z, optimize=True)

        t_b    = J2g @ (xp.abs(u0g)**2)
        E_adbc = (xp.abs(Upg)**2).T @ t_b

        # H buckets
        H2, Hm2 = E_acbd, E_adbc
        H1  = Tac + Tbd - 2.0*E_acbd
        Hm1 = Tad + Tbc - 2.0*E_adbc

        # K rows: convert CPU bath_map -> device only (small)
        K_m2 = xp.asarray(bath_map[-2.0], dtype=xp.complex128)
        K_m1 = xp.asarray(bath_map[-1.0], dtype=xp.complex128)
        K0   = xp.asarray(bath_map[ 0.0], dtype=xp.complex128)
        K1   = xp.asarray(bath_map[ 1.0], dtype=xp.complex128)
        K2   = xp.asarray(bath_map[ 2.0], dtype=xp.complex128)

        if xp.allclose(K0, 0.0):
            gamma = K_m2*Hm2 + K_m1*Hm1 + K1*H1 + K2*H2
        else:
            H0 = T0 - (H2 + Hm2 + H1 + Hm1)
            gamma = K_m2*Hm2 + K_m1*Hm1 + K0*H0 + K1*H1 + K2*H2

        return xp.asnumpy(gamma)

    # obtain redfield rates within box
    def make_redfield(self, *, pol_idxs_global, site_idxs_global, center_global, verbosity = False):
        """
        Compute outgoing Redfield rates from a fixed polaron (eigenstate) ν to all ν' in the
        current box, using an exact, closed-form λ-contraction.

        Physics
        --------------------------------------
        We evaluate the population Redfield tensor elements
            R_{νν'} = 2 Re[ Γ_{ν'ν, νν′}(ω_{ν'ν}) ]      for ν' ≠ ν,
        and return them as a 1D vector `rates` (the diagonal “loss” term is not needed for KMC).
        Here Γ is the bath‑contracted four‑index object:
            Γ_{μν, μ′ν′}(ω) = Σ_{m,n,m′,n′} J_{mn} J_{m′n′}
                            ⟨μ|m⟩⟨n|ν⟩ ⟨μ′|m′⟩⟨n′|ν′⟩ K_{mn,m′n′}(ω) ,
        with ω_{ν'ν} = E_{ν'} − E_{ν}.
        In our model the site-bath correlator K_{mn,m'n'}(ω) depends on the index
        pattern (equalities among m,n,m',n') only through five “buckets” labeled by
        λ ∈ {-2, -1, 0, 1, 2}. We therefore rewrite the four-index contraction as:
            Γ_{ν'ν, νν'}(ω_{ν'ν}) = Σ_λ  K_λ(ω_{ν'ν}) · H_λ(ν→ν′) ,
        where K_λ is the half-Fourier transform of the bath correlation,
        and H_λ are purely system (J,U) geometric prefactors.

        What this function computes
        ---------------------------
        1) Subspace selection:
        - `pol_idxs_global`  → eigenstates (“polarons”) inside the box; `center_global` is ν.
        - `site_idxs_global` → site indices that support those polarons.
        - We keep order; `center_loc` is ν's local index within `pol_idxs_global`.

        2) Bath half-Fourier rows:
        - For each λ ∈ {-2,-1,0,1,2}, build a length-P vector K_λ(ω_{ν'ν}) with
            `_corr_row(λ, center_global, pol_g)`. This is Eq. (15) (with Eq. (16) for the model),
            evaluated at ω_{ν'ν}. λ=0 contributes zero by symmetry in this model.

        3) System geometry (no big 3D tensors):
        - Extract site-submatrix J := J_dense[site_g, site_g] (real, zero diagonal),
            and eigenvector slices:
                u0 := U[site_g, ν]       (shape n)
                Up := U[site_g, ν' for all ν' in pol_g]  (shape n x P)
        - Using a few BLAS ops, compute the exact bucket prefactors H_λ(ν→ν') from:
                R_{ab,ν'} = J_{ab} · conj(U_{aν}) · U_{bν'}
                C_{ab,ν'} = J_{ab} · conj(U_{aν'}) · U_{bν}
            without materializing R or C. We form a small set of axis-sums
            (Tac, Tbd, Tad, Tbc, T0) and the two “pair” terms (E_acbd for a=c&b=d,
            E_adbc for a=d&b=c, using J^2), then apply a tiny Möbius inclusion-exclusion
            to obtain H_λ for λ = -2, -1, 0, 1, 2 exactly.

        4) Γ and rates:
        - γ_plus(ν') := Σ_λ K_λ(ω_{ν'ν}) · H_λ(ν→ν′)  (this is the contracted Γ).
        - Redfield population out-rates:  R_{ν→ν′} = 2·Re[γ_plus(ν')] / ħ  (ν' ≠ ν).
        - We delete the center entry (ν'=ν) and return the off-diagonal vector `rates`
            and the matching global destination indices `final_site_idxs`.

        Inputs
        ------
        pol_idxs_global : 1D array-like of int
            Global eigenstate indices inside the box (order preserved). Length P.
        site_idxs_global : 1D array-like of int
            Global site indices supporting the box. Length n.
        center_global : int
            The global eigenstate index ν that is the KMC “from” state.

        Internal shapes (after slicing)
        -------------------------------
        n  := len(site_idxs_global)      # sites
        P  := len(pol_idxs_global)       # polarons
        J      : (n, n)   real, diag(J)=0
        u0     : (n,)     complex  (column of U for ν)
        Up     : (n, P)   complex  (columns of U for ν' ∈ pol_g)
        K_λ    : (P,)     complex  (per λ), via `_corr_row`

        Returns
        -------
        rates : (P-1,) float64
            Outgoing rates R_{ν→ν′} for all ν' ≠ ν (in the order of `pol_idxs_global`
            with the center removed). Units: s^{-1}.
        final_site_idxs : (P-1,) int
            Global eigenstate indices ν' aligned with `rates`.
        tot_time : float
            Wall-clock time spent inside this routine (profiling aid).

        """

        t_all = time.time()
        self.time_verbose = verbosity
        time_verbose = getattr(self, "time_verbose", False)

        # (1) select the active subset in box
        # (1.1) candidate polaron desitination indices 𝜈' ∈ pol_g (including 𝜈)
        pol_g  = np.asarray(pol_idxs_global,  dtype=np.intp)
        # (1.2) site indices a,b,c,d ∈ site_g for overlap 
        site_g = np.asarray(site_idxs_global, dtype=np.intp)
        # (1.3) center_global idx 𝜈 (starting polaron)
        m0 = int(center_global)                     

        # (1.4) obtain center_global index in pol_g (local index)
        where = np.nonzero(pol_g == int(center_global))[0]
        assert where.size == 1, "center_global is not (uniquely) inside pol_idxs_global"
        center_loc = int(where[0])

        # NOTE : the follwing code is just for debugging and checking how many indices are in pol_g/site_g
        # the size of these arrays will obviously scale the computational costs
        npols  = int(pol_g.size)                        # n
        nsites = int(site_g.size)                       # P
        if time_verbose:
            print('npols, nsites', npols, nsites)

        # (2) build bath integrals K_λ(ω_{𝜈'𝜈}) (vectorized, aligned to pol_g order)
        #  we will combine them by λ ∈ {-2,-1,0,1,2}.
        t0 = time.time()
        lamvals = (-2.0, -1.0, 0.0, 1.0, 2.0)
        bath_map = {
            lam: (np.zeros(npols, np.complex128) if lam == 0.0
                else self._corr_row(lam, center_global, pol_g))
            for lam in lamvals
        }
        if time_verbose:
            print('time(bath integrals)', time.time() - t0, flush=True)

        # --- Build gamma_plus ------------
        t1 = time.time()

        # (3) build the system-bath pieces 
        # J coupling Hamiltonian on subset with indices site_g 
        J = np.asarray(
                        self.ham.J_dense[np.ix_(site_g, site_g)], dtype=np.float64, order='C'
                        )      
        J2 = self._get_J2_cached(J, site_g)             # J * J caching for memory efficieny 
        U = self.ham.Umat                               # unitary transformation for site/polaron mapping

        u0 = U[site_g, m0]                              #  overlap (a | 𝜈) for 𝜈
        Up = U[np.ix_(site_g, pol_g)]                   #  overlap (a | 𝜈') for all 𝜈' ∈ pol_g

        if time_verbose:
            print('time(site→eig rows/cols)', time.time() - t1, flush=True)


        # (4) build 𝛾_+(𝜈')
        t2 = time.time()
        # if loop GPU/CPU switch
        if self.backend.use_gpu:
            # run on GPU
            gamma_plus = Redfield._build_gamma_plus_gpu(J, J2, Up, u0, bath_map, xp=self.xp, use_c64=self.backend.gpu_use_c64)
        else:
            # run on CPU
            gamma_plus = Redfield._build_gamma_plus_cpu(J, J2, Up, u0, bath_map)  

        if time_verbose:
            print('time(gamma accumulation)', time.time() - t2, flush=True)

        # (5) compute only outgoing rates R_{𝜈𝜈'} = 2 Re𝛤_{𝜈'𝜈,𝜈𝜈'} = 2 Re 𝛾_+(𝜈') for (𝜈' = 𝜈)
        # need to remove center_loc, scale by ħ; return global final polaron indices 
        red_R_tensor = 2.0 * np.real(gamma_plus)
        rates = np.delete(red_R_tensor, center_loc) / const.hbar
        final_site_idxs = np.delete(pol_g, center_loc).astype(int)


        if time_verbose:
            print('time(total)', time.time() - t_all, flush=True)

        # NOTE : this is just for debugging
        if time_verbose:
            print('rates sum/shape', np.sum(rates), rates.shape)

        return rates, final_site_idxs, time.time() - t_all

    




