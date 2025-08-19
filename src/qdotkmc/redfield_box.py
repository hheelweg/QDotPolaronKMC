
import numpy as np
from . import const
from . import utils
import time
from scipy import sparse


class Redfield():

    def __init__(self, hamiltonian, polaron_locations, site_locations, kappa, r_hop, r_ove,
                time_verbose = True):

        self.ham = hamiltonian
        self.polaron_locations = polaron_locations          # polaron locations (global frame)
        self.site_locations = site_locations                # site locations (global frame)

        self.kappa=kappa

        # hopping and overlap radii
        self.r_hop = r_hop
        self.r_ove = r_ove

        # set to true only when time to compute rates is desired
        self.time_verbose = time_verbose

        # bath-correlation cache
        self._corr_cache = {}  # key: (lam, self.kappa, center_global) -> dict[int -> complex]

        # avoid recomputing J2 = J * J by caching
        self._J2_cache = {}  # key: tuple(site_g) -> J2 ndarray


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

    
    # NOTE : might move this to montecarlo.py since this is not effectively being used here
    def refine_by_radius(self, *,
                     pol_idxs_global, site_idxs_global, center_global,
                     periodic=False, grid_dims=None,
                     r_hop=None, r_ove=None):
        """
        Returns subsets pol_g ⊆ pol_idxs_global and site_g ⊆ site_idxs_global
        that are within r_hop / r_ove of center_global, preserving order.
        Also returns center_local (the position of center_global inside pol_g).

        Assumes:
        - self.polaron_locations are absolute coordinates (shape: [N, D])
        - self.ham.qd_lattice_rel are site coordinates in the same frame
        """
        if r_hop is None: r_hop = self.r_hop
        if r_ove is None: r_ove = self.r_ove

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
            "center_global not in pol_idxs_global after box selection; "
            "ensure the box always includes the center polaron."
        )

        return pol_g, site_g
 

    def _score_candidates(self, center_global, pol_pool=None, use_sites=None):
        """
        Return (score, W, S) for ν' in pol_pool (default: all), using:
        W_{ν'} = sum_λ |K_λ(ω_{ν'ν})|   via _corr_row (cached)
        S_{ν→ν'} = w_ν^T |J|^2 w_{ν'}   (contact)
        If use_sites is provided, restrict J^2 and U to those sites (cheap).
        """
        U = self.ham.Umat
        J = self.ham.J_dense
        N_sites, N_pol = U.shape
        nu = int(center_global)

        if pol_pool is None:
            pol_pool = np.arange(N_pol, dtype=np.intp)
        pol_pool = np.asarray(pol_pool, dtype=np.intp)

        # --- Bath weights: prefetch & cache rows for all ν' in pol_pool
        W = np.zeros(pol_pool.size, float)
        for lam in (-2.0, -1.0, 0.0, 1.0, 2.0):
            row = (np.zeros(pol_pool.size, np.complex128) if lam == 0.0
                else self._corr_row(lam, nu, pol_pool))
            W += np.abs(row)

        # --- Contact score S = w_ν^T |J|^2 w_{ν'}
        if use_sites is None:
            w_nu = np.abs(U[:, nu])**2
            J2_full = (np.abs(J)**2)
            t = J2_full @ w_nu                           # (N_sites,)
            S_all = (np.abs(U)**2).T @ t                 # (N_pol,)
        else:
            idx = np.asarray(use_sites, dtype=np.intp)
            U2s = (np.abs(U[idx, :])**2)                 # (n, N_pol)
            w_nu = U2s[:, nu]
            J2s = (np.abs(J[np.ix_(idx, idx)])**2)
            t = J2s @ w_nu                               # (n,)
            S_all = U2s.T @ t                            # (N_pol,)

        # Slice to pol_pool
        S = S_all[pol_pool]
        score = W * S
        # zero out self
        where_self = np.nonzero(pol_pool == nu)[0]
        if where_self.size:
            score[where_self[0]] = 0.0
        return score, W, S
    
    def _k_until_cov(self, vals_desc, eta):
        v = np.asarray(vals_desc, float)
        tot = v.sum()
        if tot <= 0: return 0
        k = np.searchsorted(np.cumsum(v), (1.0 - float(eta)) * tot, side="left") + 1
        return int(min(k, v.size))

    def select_by_eta_after_box(self, center_global, *, eta=0.02,
                                pol_pool=None, site_pool=None, verbose=False):
        """
        Use existing (site_pool, pol_pool) as the *universe*.
        Pick ν' and sites by (1-η)-coverage of a safe proxy, reusing _corr_row cache.
        Returns (site_g_sel, pol_g_sel[center first]).
        """
        nu = int(center_global)
        U = self.ham.Umat
        J = self.ham.J_dense
        N_sites, N_pol = U.shape

        if pol_pool is None:
            pol_pool = np.arange(N_pol, dtype=np.intp)
        if site_pool is None:
            site_pool = np.arange(N_sites, dtype=np.intp)

        # Score within the provided pools; bath rows are cached here
        score, W, S = self._score_candidates(nu, pol_pool=pol_pool, use_sites=site_pool)

        order_pool = pol_pool[np.argsort(-score)]
        k_pol = self._k_until_cov(score[np.argsort(-score)], eta)
        kept_pol = order_pool[:k_pol]
        pol_g_sel = np.concatenate(([nu], kept_pol)).astype(np.intp)

        # Per-ν' edge weights within site_pool
        idx = np.asarray(site_pool, dtype=np.intp)
        U2s = (np.abs(U[idx, :])**2)
        w_src = U2s[:, nu]
        J2s = (np.abs(J[np.ix_(idx, idx)])**2)

        endpoints = set()
        # source "mass core" by the SAME eta (one knob)
        w = w_src.copy()
        order_i = np.argsort(w)[::-1]
        ki = self._k_until_cov(w[order_i], eta)
        src_core = idx[order_i[:ki]]
        endpoints.update(src_core.tolist())

        for nu_p in kept_pol:
            # destination mass core by same eta
            wdst = U2s[:, nu_p]
            order_j = np.argsort(wdst)[::-1]
            kj = self._k_until_cov(wdst[order_j], eta)
            dst_core = idx[order_j[:kj]]

            # rank edges on src_core × dst_core by e_ij = |J|^2 * w_i * w'_j
            if src_core.size == 0 or dst_core.size == 0:
                continue
            J2_sub = J2s[np.ix_(np.isin(idx, src_core), np.isin(idx, dst_core))]
            e = (w[np.isin(idx, src_core)][:, None]) * J2_sub * (wdst[np.isin(idx, dst_core)][None, :])
            ef = e.ravel()
            if ef.size == 0 or ef.sum() == 0.0:
                continue
            order_e = np.argsort(-ef)
            k_e = self._k_until_cov(ef[order_e], eta)
            keep = order_e[:k_e]
            ii = keep // e.shape[1]
            jj = keep %  e.shape[1]
            endpoints.update(src_core[ii].tolist())
            endpoints.update(dst_core[jj].tolist())

        site_g_sel = np.array(sorted(endpoints), dtype=np.intp)

        if verbose:
            cov_pol = (score[np.argsort(-score)][:k_pol].sum() / (score.sum() + 1e-300))
            print(f"[η] kept ν': {len(pol_g_sel)-1}/{len(pol_pool)-1}, coverage≈{cov_pol:.3f}, sites={site_g_sel.size}")
        return site_g_sel, pol_g_sel


    # obtain redfield rates within box
    def make_redfield_box(self, *, pol_idxs_global, site_idxs_global, center_global):
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
        n  := len(site_idxs_global)      # sites in box
        P  := len(pol_idxs_global)       # polarons in box
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
        J2 = self._get_J2_cached(J, site_g)         # J * J caching for memory efficieny 
        U = self.ham.Umat                           # unitary transformation for site/polaron mapping

        u0 = U[site_g, m0]                          #  overlap (a | 𝜈) for 𝜈
        Up = U[np.ix_(site_g, pol_g)]               #  overlap (a | 𝜈') for all 𝜈' ∈ pol_g

        if time_verbose:
            print('time(site→eig rows/cols)', time.time() - t1, flush=True)

        # function for computing 𝛾_+(𝜈') (exact, closed-form λ-contraction)
        def _build_gamma_plus(J, J2, Up, u0, bath_map):
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

        # (4) build 𝛾_+(𝜈')
        t2 = time.time()
        gamma_plus = _build_gamma_plus(J, J2, Up, u0, bath_map)
        if time_verbose:
            print('time(gamma accumulation)', time.time() - t2, flush=True)

        # (5) compute only outgoing rates R_{𝜈𝜈'} = 2 Re𝛤_{𝜈'𝜈,𝜈𝜈'} = 2 Re 𝛾_+(𝜈') for (𝜈' = 𝜈)
        # need to remove center_loc, scale by ħ; return global final polaron indices 
        red_R_tensor = 2.0 * np.real(gamma_plus)
        rates = np.delete(red_R_tensor, center_loc) / const.hbar
        final_site_idxs = np.delete(pol_g, center_loc).astype(int)



        if time_verbose:
            print('time(total)', time.time() - t_all, flush=True)

        print('rates sum/shape', np.sum(rates), rates.shape)
        return rates, final_site_idxs, time.time() - t_all

    




    
    
    



