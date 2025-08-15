
import numpy as np
from . import const
from . import utils
import time
from scipy import sparse


class Redfield():

    def __init__(self, hamiltonian, polaron_locations, site_locations, kappa, r_hop, r_ove,
                time_verbose = True, enforce_db = True):

        self.ham = hamiltonian
        self.polaron_locations = polaron_locations          # polaron locations (global frame)
        self.site_locations = site_locations                # site locations (global frame)

        self.kappa=kappa

        # hopping and overlap radii
        self.r_hop = r_hop
        self.r_ove = r_ove

        # set to true only when time to compute rates is desired
        self.time_verbose = time_verbose

        # set to true when we want to enforce detailled balance on rates
        self.enforce_db = enforce_db

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

        # # (4) build 𝛾_+(𝜈')
        # t2 = time.time()
        # gamma_plus = _build_gamma_plus(J, J2, Up, u0, bath_map)
        # if time_verbose:
        #     print('time(gamma accumulation)', time.time() - t2, flush=True)

        # # (5) compute only outgoing rates R_{𝜈𝜈'} = 2 Re𝛤_{𝜈'𝜈,𝜈𝜈'} = 2 Re 𝛾_+(𝜈') for (𝜈' = 𝜈)
        # # need to remove center_loc, scale by ħ; return global final polaron indices 
        # red_R_tensor = 2.0 * np.real(gamma_plus)
        # rates = np.delete(red_R_tensor, center_loc) / const.hbar
        # final_site_idxs = np.delete(pol_g, center_loc).astype(int)



        # if time_verbose:
        #     print('time(total)', time.time() - t_all, flush=True)

        # print('rates sum/shape', np.sum(rates), rates.shape)
        # return rates, final_site_idxs, time.time() - t_all

        # (4) build 𝛾_+(ν′)
        t2 = time.time()
        gamma_plus = _build_gamma_plus(J, J2, Up, u0, bath_map)
        if time_verbose:
            print('time(gamma accumulation)', time.time() - t2, flush=True)

        # (4b) OPTIONAL: enforce pairwise detailed balance for explicit neighbors
        # We do this BEFORE removing center_loc, working on the full "pol_g"-aligned vector.
        red_R_tensor_full = 2.0 * np.real(gamma_plus) / const.hbar  # [P] incl. center

        if getattr(self, "enforce_db", False):
            beta = float(self.ham.beta)           # make sure self.ham.beta exists (1/kT)
            E    = np.asarray(self.ham.E)         # or use omega_diff below if E not available

            # helper: build bath rows for another center (reuse caches)
            def _bath_rows_for(center_idx):
                return {
                    lam: (np.zeros_like(red_R_tensor_full, dtype=np.complex128) if lam == 0.0
                        else self._corr_row(lam, int(center_idx), pol_g))
                    for lam in (-2.0, -1.0, 0.0, 1.0, 2.0)
                }

            # corrected copy we will fill
            red_R_tensor_corr = red_R_tensor_full.copy()

            # Loop over destinations (skip the center entry)
            for dest_loc in range(pol_g.size):
                if dest_loc == center_loc:
                    continue
                j_global = int(pol_g[dest_loc])

                # forward (i -> j): already have it
                k_ij_R = max(0.0, red_R_tensor_full[dest_loc])

                # reverse (j -> i): compute in THE SAME BOX by swapping u0 to the j-column
                # build bath rows with center = j, and use u1 = U[site_g, j]
                bath_map_rev = _bath_rows_for(j_global)
                u1 = U[site_g, j_global]                   # column for j in the global U
                gamma_plus_rev = _build_gamma_plus(J, J2, Up, u1, bath_map_rev)

                # find index of m0 inside pol_g to read the reverse element
                idx_m0_in_pol_g = center_loc
                k_ji_R = max(0.0, 2.0 * np.real(gamma_plus_rev[idx_m0_in_pol_g]) / const.hbar)

                # target ratio exp(-beta * ΔE); use energies if available, else omega_diff
                if hasattr(self.ham, "E"):
                    dE = float(E[j_global] - E[m0])
                else:
                    dE = float(self.ham.omega_diff[j_global, m0])  # E_j - E_i
                g = np.exp(-beta * dE)

                # Metropolis-like symmetric correction (keeps scale, enforces k_ij/k_ji = g)
                k_ij = np.sqrt(k_ij_R * k_ji_R * g) if (k_ij_R > 0.0 and k_ji_R > 0.0) else 0.0

                red_R_tensor_corr[dest_loc] = k_ij

            red_R_tensor_used = red_R_tensor_corr
        else:
            red_R_tensor_used = red_R_tensor_full

        # (5) now drop the center entry and return
        rates = np.delete(red_R_tensor_used, center_loc)
        final_site_idxs = np.delete(pol_g, center_loc).astype(int)

        if time_verbose:
            print('time(total)', time.time() - t_all, flush=True)

        print('rates sum/shape', np.sum(rates), rates.shape)
        return rates, final_site_idxs, time.time() - t_all
    




    
    
    



