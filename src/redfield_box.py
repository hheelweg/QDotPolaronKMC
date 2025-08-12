
import numpy as np
from . import const
from . import utils
import time
from numba import njit
from scipy import sparse

class Unitary(object):
    """A unitary evolution class
    """

    def __init__(self, hamiltonian):
        self.ham = hamiltonian

    def setup(self):
        pass

class Redfield(Unitary):
    """ class to compute the Redfield rates for KMC
    """

    def __init__(self, hamiltonian, polaron_locations, site_locations, kappa, r_hop, r_ove, time_verbose = True):

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

        # avoid recomputing J2 = J * J
        self._J2_cache = {}  # key: tuple(site_g) -> J2 ndarray

        # cache pruned A-maps
        self._A_lambda_cache = {}          # { nsites : {lam: CSR} }
        self._A_lambda_pruned = {}         # { (nsites, mask_sig) : {lam: CSR_pruned} }

    # bath correlation function
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

    # pruned A-maps
    def _get_pruned_A_map(self, nsites, A_map, ab_keep):
        key = (nsites, ab_keep.tobytes())  # mask signature
        P = self._A_lambda_pruned.get(key)
        if P is not None:
            return P
        # build and store once
        P = {}
        for lam, A in A_map.items():
            if A is None:
                P[lam] = None
            else:
                P[lam] = A[ab_keep][:, ab_keep]
        self._A_lambda_pruned[key] = P
        return P
    

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
 

    # (08/12/2015) this is a bit black box but super fast (and correct!)
    # def make_redfield_box(self, *, pol_idxs_global, site_idxs_global, center_global):
    #     """
    #     Exact-physics clone (global indexing), optimized with closed-form λ-contraction:
    #     - No CSR / no (ab,cd) pair loops
    #     - γ accumulation in O(nsites^2 * npols) using row/col/diag reductions + Möbius inversion
    #     """
    #     import time
    #     import numpy as np

    #     t_all = time.time()
    #     time_verbose = getattr(self, "time_verbose", False)

    #     # --- local views (preserve order)
    #     pol_g  = np.asarray(pol_idxs_global,  dtype=np.intp)
    #     site_g = np.asarray(site_idxs_global, dtype=np.intp)
    #     npols  = int(pol_g.size)
    #     nsites = int(site_g.size)
    #     if time_verbose:
    #         print('npols, nsites', npols, nsites)

    #     # Map center_global -> local index
    #     where = np.nonzero(pol_g == int(center_global))[0]
    #     assert where.size == 1, "center_global is not (uniquely) inside pol_idxs_global"
    #     center_loc = int(where[0])

    #     # --- Bath integrals (vectorized, aligned to pol_g order)
    #     #     We will combine them by λ ∈ {-2,-1,0,1,2} after the closed-form contraction.
    #     t0 = time.time()
    #     lamvals = (-2.0, -1.0, 0.0, 1.0, 2.0)
    #     bath_map = {
    #         lam: (np.zeros(npols, np.complex128) if lam == 0.0
    #             else self._corr_row(lam, center_global, pol_g))
    #         for lam in lamvals
    #     }
    #     if time_verbose:
    #         print('time(bath integrals)', time.time() - t0, flush=True)

    #     # --- Build dense R and C once (nsites, nsites, npols)
    #     t1 = time.time()
    #     J_mat = self.ham.J_dense[np.ix_(site_g, site_g)]  # (n, n)
    #     U = self.ham.Umat
    #     m0 = int(center_global)

    #     U_site_center = U[site_g, m0]                 # (n,)
    #     U_site_pol    = U[np.ix_(site_g, pol_g)]      # (n, P)
    #     U_site_pol_c  = np.conj(U_site_pol)           # (n, P)

    #     # R3D[a,b,p] = J[a,b] * conj(U[a,m0]) * U[b,p]
    #     # C3D[a,b,p] = J[a,b] * conj(U[a,p]) * U[b,m0]
    #     R3D = (J_mat[:, :, None]
    #         * np.conj(U_site_center)[:, None, None]
    #         * U_site_pol[None, :, :])              # (n, n, P)
    #     C3D = (J_mat[:, :, None]
    #         * U_site_pol_c[:, None, :]
    #         * U_site_center[None, :, None])        # (n, n, P)
    #     if time_verbose:
    #         print('time(site→eig rows/cols)', time.time() - t1, flush=True)

    #     # --- Closed-form λ-bucket contraction (no pairs, no CSR)
    #     # We need, for each p, the sums over indices in the 5 λ-classes. Do this by
    #     # computing T[mask] for the 4 equalities {ac, bd, ad, bc}, then Möbius inversion
    #     # to get exact-class sums, and finally aggregating by score s = (#ac + #bd) - (#ad + #bc).

    #     def _lambda_contraction(R3D, C3D, bath_map):
    #         n, _, P = R3D.shape
    #         R = R3D; C = C3D

    #         # Basic reductions (all vectorized over P)
    #         rowR = R.sum(axis=1)             # (n, P)  sum over b
    #         colR = R.sum(axis=0)             # (n, P)  sum over a
    #         rowC = C.sum(axis=1)             # (n, P)
    #         colC = C.sum(axis=0)             # (n, P)
    #         diagR = np.einsum('aap->ap', R)  # (n, P)
    #         diagC = np.einsum('aap->ap', C)  # (n, P)

    #         # T[mask] = sum over indices where equalities in 'mask' hold (others unconstrained)
    #         # mask bits order: 0=ac, 1=bd, 2=ad, 3=bc
    #         def m(*bits): return sum(1 << b for b in bits)
    #         T = {}

    #         # empty set (no constraints)
    #         T[m()]       = (R.sum(axis=(0,1)) * C.sum(axis=(0,1)))               # (P,)
    #         # singletons
    #         T[m(0)]      = (rowR * rowC).sum(axis=0)                             # ac
    #         T[m(1)]      = (colR * colC).sum(axis=0)                             # bd
    #         T[m(2)]      = (rowR * colC).sum(axis=0)                             # ad
    #         T[m(3)]      = (colR * rowC).sum(axis=0)                             # bc
    #         # pairs
    #         T[m(0,1)]    = (R * C).sum(axis=(0,1))                               # ac & bd -> (a=c, b=d)
    #         T[m(0,2)]    = (rowR * diagC).sum(axis=0)                            # ac & ad -> c=d=a
    #         T[m(0,3)]    = (diagR * rowC).sum(axis=0)                            # ac & bc -> a=b=c
    #         T[m(1,2)]    = (diagR * colC).sum(axis=0)                            # bd & ad -> a=b=d
    #         T[m(1,3)]    = (colR * diagC).sum(axis=0)                            # bd & bc -> b=c=d
    #         T[m(2,3)]    = (R * C.swapaxes(0,1)).sum(axis=(0,1))                 # ad & bc -> (a=d, b=c)
    #         # triples and quadruple (all indices equal)
    #         diag_prod    = (diagR * diagC).sum(axis=0)
    #         T[m(0,1,2)]  = diag_prod
    #         T[m(0,1,3)]  = diag_prod
    #         T[m(0,2,3)]  = diag_prod
    #         T[m(1,2,3)]  = diag_prod
    #         T[m(0,1,2,3)] = diag_prod

    #         # Möbius inversion on the 4-variable Boolean lattice:
    #         # Exact[mask] = T[mask] - sum_{superset ⊃ mask} Exact[superset]
    #         masks = sorted(T.keys(), key=lambda x: bin(x).count("1"), reverse=True)
    #         Exact = {}
    #         for mask in masks:
    #             val = T[mask]
    #             for sup in Exact:
    #                 if (sup & mask) == mask and sup != mask:  # sup is a strict superset
    #                     val = val - Exact[sup]
    #             Exact[mask] = val

    #         # Aggregate by λ score s = (#ac + #bd) - (#ad + #bc) ∈ {-2,-1,0,1,2}
    #         H = {s: np.zeros(npols, dtype=np.complex128) for s in (-2, -1, 0, 1, 2)}
    #         for mask, vec in Exact.items():
    #             countX = ((mask >> 0) & 1) + ((mask >> 1) & 1)  # ac + bd
    #             countZ = ((mask >> 2) & 1) + ((mask >> 3) & 1)  # ad + bc
    #             s = countX - countZ
    #             H[s] += vec

    #         # Combine with bath integrals per λ
    #         out = np.zeros(npols, dtype=np.complex128)
    #         for s, vec in H.items():
    #             out += bath_map[float(s)] * vec
    #         return out

    #     t2 = time.time()
    #     gamma_plus = _lambda_contraction(R3D, C3D, bath_map)  # (npols,)
    #     if time_verbose:
    #         print('time(gamma accumulation)', time.time() - t2, flush=True)

    #     # --- outgoing rates (remove center), scale by ħ; return GLOBAL final indices
    #     red_R_tensor = 2.0 * np.real(gamma_plus)
    #     rates = np.delete(red_R_tensor, center_loc) / const.hbar
    #     final_site_idxs = np.delete(pol_g, center_loc).astype(int)

    #     if time_verbose:
    #         print('time(total)', time.time() - t_all, flush=True)

    #     print('rates sum/shape', np.sum(rates), rates.shape)
    #     return rates, final_site_idxs, time.time() - t_all

    # this is even 10x faster than the one before
    def make_redfield_box(self, *, pol_idxs_global, site_idxs_global, center_global):
        """
        Exact-physics clone (global indexing), optimized with closed-form λ-contraction:
        - No CSR / no (ab,cd) pair loops
        - γ accumulation in O(nsites^2 * npols) using row/col/diag reductions + Möbius inversion
        """
        import time
        import numpy as np

        t_all = time.time()
        time_verbose = getattr(self, "time_verbose", False)

        # --- local views (preserve order)
        pol_g  = np.asarray(pol_idxs_global,  dtype=np.intp)
        site_g = np.asarray(site_idxs_global, dtype=np.intp)
        npols  = int(pol_g.size)
        nsites = int(site_g.size)
        if time_verbose:
            print('npols, nsites', npols, nsites)

        # Map center_global -> local index
        where = np.nonzero(pol_g == int(center_global))[0]
        assert where.size == 1, "center_global is not (uniquely) inside pol_idxs_global"
        center_loc = int(where[0])

        # --- Bath integrals (vectorized, aligned to pol_g order)
        #     We will combine them by λ ∈ {-2,-1,0,1,2} after the closed-form contraction.
        t0 = time.time()
        lamvals = (-2.0, -1.0, 0.0, 1.0, 2.0)
        bath_map = {
            lam: (np.zeros(npols, np.complex128) if lam == 0.0
                else self._corr_row(lam, center_global, pol_g))
            for lam in lamvals
        }
        if time_verbose:
            print('time(bath integrals)', time.time() - t0, flush=True)

        # --- Build gamma_plus
        t1 = time.time()

        # system-bath operator J_dense
        J = np.asarray(self.ham.J_dense[np.ix_(site_g, site_g)], dtype=np.float64, order='C')  # (n,n)
        J2 = self._get_J2_cached(J, site_g)
        U = self.ham.Umat
        m0 = int(center_global)
        u0 = U[site_g, m0]                          # (n,)
        Up = U[np.ix_(site_g, pol_g)]               # (n,P)

        if time_verbose:
            print('time(site→eig rows/cols)', time.time() - t1, flush=True)



        def _gamma_closed_form_fast(J, J2, Up, u0, bath_map):
            n, P = Up.shape
            Upc  = Up.conj()

            # Shared matmuls (1 fewer than before)
            Ju0 = J @ u0            # (n,)
            JUp = J @ Up            # (n,P)

            # Row/col sums
            rowR = (u0.conj()[:, None]) * JUp          # (n,P)
            colR = (Ju0.conj()[:, None]) * Up          # (n,P)
            rowC = Upc * Ju0[:, None]                  # (n,P)
            colC = (u0[:, None]) * JUp.conj()          # (n,P)  # uses conj(JUp), avoids JUpc matmul

            sum_rowR = rowR.sum(axis=0)                # (P,)
            sum_rowC = rowC.sum(axis=0)                # (P,)
            T0  = sum_rowR * sum_rowC
            Tac = (rowR * rowC).sum(axis=0)            # ac
            Tbd = (colR * colC).sum(axis=0)            # bd
            Tad = (rowR * colC).sum(axis=0)            # ad
            Tbc = (colR * rowC).sum(axis=0)            # bc

            # Pair buckets via one matmul + a couple of elementwise ops
            U0Up = u0[None, :] * Up.T                   # (P,n)   target vector batch (per p)
            Z    = (J2 @ U0Up.T)                        # (n,P)
            V    = u0.conj()[:, None] * Upc             # (n,P)
            Tpair  = (V * Z).sum(axis=0)                # (P,)  -> ac & bd

            Au0    = np.abs(u0)**2                      # (n,)
            AUp    = (np.abs(Up)**2)                    # (n,P)
            t_b    = J2 @ Au0                           # (n,)
            Tcross = (AUp.T @ t_b)                      # (P,)  -> ad & bc

            E_ac, E_bd, E_ad, E_bc = Tac, Tbd, Tad, Tbc
            E_acbd  = Tpair
            E_adbc  = Tcross
            H2   = E_acbd
            Hm2  = E_adbc
            H1   = E_ac + E_bd - 2.0 * E_acbd
            Hm1  = E_ad + E_bc - 2.0 * E_adbc
            H0   = T0 - (H2 + Hm2 + H1 + Hm1)

            return (bath_map[-2.0] * Hm2
                    + bath_map[-1.0] * Hm1
                    + bath_map[ 0.0] * H0
                    + bath_map[ 1.0] * H1
                    + bath_map[ 2.0] * H2)

        
        t2 = time.time()
        gamma_plus = _gamma_closed_form_fast(J, Up, u0, bath_map)
        if time_verbose:
            print('time(gamma accumulation)', time.time() - t2, flush=True)

        # --- outgoing rates (remove center), scale by ħ; return GLOBAL final indices
        red_R_tensor = 2.0 * np.real(gamma_plus)
        rates = np.delete(red_R_tensor, center_loc) / const.hbar
        final_site_idxs = np.delete(pol_g, center_loc).astype(int)

        if time_verbose:
            print('time(total)', time.time() - t_all, flush=True)

        print('rates sum/shape', np.sum(rates), rates.shape)
        return rates, final_site_idxs, time.time() - t_all
    


    # (08/12/2025) working code
    # NOTE : keep this as legacy code somewhere
    # def make_redfield_box(self, *, pol_idxs_global, site_idxs_global, center_global):
    #     """
    #     Exact-physics clone (global indexing), optimized:
    #     - reuses cached eigen-basis operators per (a,b)
    #     - builds R/C directly (no 4D Gs allocation)
    #     - bath integrals vectorized
    #     """
    #     t_all = time.time()
    #     time_verbose = getattr(self, "time_verbose", False)

    #     # --- local views of the provided global sets (preserve order)
    #     pol_g  = np.asarray(pol_idxs_global,  dtype=np.intp)
    #     site_g = np.asarray(site_idxs_global, dtype=np.intp)
    #     npols  = int(pol_g.size)
    #     nsites = int(site_g.size)
    #     if time_verbose:
    #         print('npols, nsites', npols, nsites)

    #     # Map center_global -> local index (center_loc)
    #     where = np.nonzero(pol_g == int(center_global))[0]
    #     assert where.size == 1, "center_global is not (uniquely) inside pol_idxs_global"
    #     center_loc = int(where[0])

    #     # --- λ index cache (by nsites), identical to your baseline
    #     if not hasattr(self, "_lam_idx_cache"):
    #         self._lam_idx_cache = {}
    #     lamdalist = (-2.0, -1.0, 0.0, 1.0, 2.0)
    #     if nsites not in self._lam_idx_cache:
    #         ident = np.identity(nsites)
    #         ones  = np.ones((nsites, nsites, nsites, nsites))
    #         lamdas = ( np.einsum('ac, abcd->abcd', ident, ones)
    #                 + np.einsum('bd, abcd->abcd', ident, ones)
    #                 - np.einsum('ad, abcd->abcd', ident, ones)
    #                 - np.einsum('bc, abcd->abcd', ident, ones) )
    #         idx_dict = {}
    #         for lam in lamdalist:
    #             idxs = np.argwhere(lamdas == lam)
    #             if idxs.size == 0:
    #                 idx_dict[lam] = (np.array([], dtype=int),
    #                                 np.array([], dtype=int),
    #                                 np.array([], dtype=int),
    #                                 np.array([], dtype=int))
    #             else:
    #                 a_idx, b_idx, c_idx, d_idx = idxs.T
    #                 idx_dict[lam] = (a_idx, b_idx, c_idx, d_idx)
    #         del lamdas
    #         self._lam_idx_cache[nsites] = idx_dict
    #     idx_dict = self._lam_idx_cache[nsites]

    #     # --- optional CSR map cache (by nsites),
    #     if not hasattr(self, "_A_lambda_cache"):
    #         self._A_lambda_cache = {}
    #     if nsites not in self._A_lambda_cache:
    #         AB = nsites * nsites
    #         A_map = {}
    #         for lam in lamdalist:
    #             a_idx, b_idx, c_idx, d_idx = idx_dict[lam]
    #             if a_idx.size == 0:
    #                 A_map[lam] = None
    #             else:
    #                 ab_flat = (a_idx * nsites + b_idx).astype(np.intp)
    #                 cd_flat = (c_idx * nsites + d_idx).astype(np.intp)
    #                 data = np.ones_like(ab_flat, dtype=np.float64)
    #                 A_map[lam] = sparse.csr_matrix((data, (ab_flat, cd_flat)), shape=(AB, AB))
    #         self._A_lambda_cache[nsites] = A_map
    #     A_map = self._A_lambda_cache[nsites]

    #     # --- Bath integrals (vectorized, global ω-row aligned to pol_g order)
    #     t0 = time.time()
    #     # cached way of computing bath correletion FT
    #     bath_integrals = [
    #                         np.zeros(npols, np.complex128) if lam == 0.0 else self._corr_row(lam, center_global, pol_g)
    #                         for lam in lamdalist
    #                      ]
        
    #     if time_verbose:
    #         print('time(bath integrals)', time.time() - t0, flush=True)

    #     # --- (BIG WIN) Build R and C directly from cached full eigen-operators
    #     t1 = time.time()


    #     J_mat = self.ham.J_dense[np.ix_(site_g, site_g)]  # (nsites, nsites)
    #     U = self.ham.Umat
    #     m0 = int(center_global)

    #     U_site_center = U[site_g, m0]                 # (nsites,)
    #     U_site_pol    = U[np.ix_(site_g, pol_g)]      # (nsites, npols)
    #     U_site_pol_c  = np.conj(U_site_pol)           # (nsites, npols)

    #     # Broadcast to 3D: (nsites, nsites, npols)
    #     # R3D[a,b,:] = J[a,b] * conj(U[a,m0]) * U[b,:]
    #     R3D = (J_mat[:, :, None]
    #         * np.conj(U_site_center)[:, None, None]
    #         * U_site_pol[None, :, :])

    #     # C3D[a,b,:] = J[a,b] * conj(U[a,:]) * U[b,m0]
    #     C3D = (J_mat[:, :, None]
    #         * U_site_pol_c[:, None, :]
    #         * U_site_center[None, :, None])

    #     # Flatten (a,b) -> ab, Fortran order helps CSR @ dense that comes next
    #     AB = nsites * nsites
    #     R = np.asfortranarray(R3D.reshape(AB, npols), dtype=np.complex128)
    #     C = np.asfortranarray(C3D.reshape(AB, npols), dtype=np.complex128)

    #     # no transpose needed anymore
    #     if time_verbose:
    #         print('time(site→eig rows/cols)', time.time() - t1, flush=True)

    #     # --- prune exactly-zero rows/cols 
    #     ab_keep = np.any(R != 0, axis=1) | np.any(C != 0, axis=1)
    #     if ab_keep.sum() < AB:
    #         R = R[ab_keep, :]
    #         C = C[ab_keep, :]
    #         A_map = self._get_pruned_A_map(nsites, A_map, ab_keep)
    #         AB = ab_keep.sum()

    #     # --- gamma accumulation via CSR×dense and einsum (identical algebra/order)
    #     t2 = time.time()
    #     Y = np.empty_like(R, order='F')
    #     tmp = np.empty_like(R, order='F')

    #     gamma_plus = np.zeros(npols, dtype=np.complex128)
    #     for k, lam in enumerate(lamdalist):
    #         A = A_map[lam]
    #         if A is None:
    #             continue
    #         Y[:] = A.dot(C)                # or A.dot(C, out=Y) on newer SciPy
    #         np.multiply(R, Y, out=tmp)
    #         contrib = tmp.sum(axis=0)
    #         gamma_plus += bath_integrals[k] * contrib

    #     if time_verbose:
    #         print('time(gamma accumulation)', time.time() - t2, flush=True)

    #     # --- outgoing rates (remove center), scale by ħ; return GLOBAL final indices
    #     red_R_tensor = 2.0 * np.real(gamma_plus)
    #     rates = np.delete(red_R_tensor, center_loc) / const.hbar
    #     final_site_idxs = np.delete(pol_g, center_loc).astype(int)

    #     if time_verbose:
    #         print('time(total)', time.time() - t_all, flush=True)

    #     print('rates sum/shape', np.sum(rates), rates.shape)
    #     return rates, final_site_idxs, time.time() - t_all


    
        # legcy code, should be removed eventually. 
    def get_idxsNEW(self, center_idx):
        """
        Return:
            polaron_idxs: array of polaron indices within r_hop of center
            site_idxs: list of arrays; for each polaron_idx, the site indices within r_ove of both center and that polaron
        """
        center_coord = self.polaron_locations[center_idx]

        # Find polarons within r_hop of center
        polaron_coords = np.array(self.polaron_locations)
        distances_to_center = np.linalg.norm(polaron_coords - center_coord, axis=1)
        polaron_idxs = np.where(distances_to_center < self.r_hop)[0]

        # Precompute distances from all sites to center
        site_coords = np.array(self.ham.qd_lattice_rel)
        d_center = np.linalg.norm(site_coords - center_coord, axis=1)

        # For each polaron_idx, get intersection of site indices within r_ove of both polaron and center
        site_idxs = []
        for pol_idx in polaron_idxs:
            d_pol = np.linalg.norm(site_coords - polaron_coords[pol_idx], axis=1)
            overlap_idxs = np.where((d_center < self.r_ove) & (d_pol < self.r_ove))[0]
            site_idxs.append(overlap_idxs)

        return polaron_idxs, site_idxs
    

    # VERSION 4 : this is actually doing it w.r.t. some overlap region (08/08/2025)
    # def make_redfield_box(self, center_idx):
    #     """
    #     General path using get_idxsNew(center_idx).
    #     - Per-destination site sets S_i (polaron-dependent nsites).
    #     - EXACT same physics as your validated vectorized function:
    #         * same λ list, same weighting
    #         * same row/col orientation
    #         * same LOCAL omega indexing (omega_diff[i, center_idx])
    #     but the a,b,c,d sums are restricted to S_i x S_i for each i.
    #     Returns:
    #         rates, final_pol_idxs (GLOBAL eigenstate indices), walltime
    #     """
    #     t0_all = time.time()

    #     # --- neighborhoods (polaron subset and per-destination site sets)
    #     pol_idxs, site_idxs_list = self.get_idxsNew(center_idx)   # NEW function
    #     npols = len(pol_idxs)
    #     if self.time_verbose:
    #         print('npols', npols, [len(arr) for arr in site_idxs_list])
    #     if npols == 0:
    #         return np.array([], dtype=float), np.array([], dtype=int), time.time() - t0_all

    #     # local index of the center eigenstate within this polaron subset
    #     center_i = int(np.where(pol_idxs == center_idx)[0][0])

    #     # Build the union of sites across all S_i so we can transform once
    #     union_sites = np.unique(np.concatenate(site_idxs_list)) if len(site_idxs_list) else np.array([], dtype=int)
    #     Nunion = len(union_sites)

    #     # Map global site index -> position in union [0..Nunion-1]
    #     site_pos = -np.ones(len(self.ham.qd_lattice_rel), dtype=int)
    #     site_pos[union_sites] = np.arange(Nunion, dtype=int)

    #     # --- Bath integrals (KEEP the baseline's LOCAL omega indexing!)
    #     lamdalist = (-2, -1, 0, 1, 2)
    #     t_bath = time.time()
    #     bath_integrals = []
    #     for lam in lamdalist:
    #         vec = np.zeros(npols, dtype=np.complex128)
    #         if lam != 0:
    #             for i in range(npols):  # LOCAL i (0..npols-1) on purpose to match baseline
    #                 omega_ij = self.ham.omega_diff[i, center_idx]
    #                 vec[i] = self.ham.spec.correlationFT(omega_ij, lam, self.kappa)
    #         bath_integrals.append(vec)
    #     if self.time_verbose:
    #         print('time(bath integrals):', time.time() - t_bath, flush=True)

    #     # --- Transform sysbath operators to eigenbasis ONCE over the union sites
    #     #     and restrict to the local polaron subspace (pol_idxs).
    #     #     We'll later pick [center_i, i] and [i, center_i] entries from these.
    #     t_g = time.time()
    #     # Gs_union[a,b,:,:] = (U† V_ab U)[pol_idxs, pol_idxs]
    #     Gs_union = np.empty((Nunion, Nunion, npols, npols), dtype=np.complex128)
    #     for ai, a in enumerate(union_sites):
    #         for bi, b in enumerate(union_sites):
    #             G_full = self.ham.site2eig(self.ham.sysbath[a][b])      # full eigbasis operator
    #             Gs_union[ai, bi] = G_full[np.ix_(pol_idxs, pol_idxs)]   # restrict to local polaron box
    #     if self.time_verbose:
    #         print('time(site→eig over union):', time.time() - t_g, flush=True)

    #     # Prepare convenient row/col views over the union (contiguous helps)
    #     # row_full[ai,bi,:] = G[a,b][center_i, :]   (vector over destination states ν′)
    #     # col_full[ai,bi,:] = G[a,b][:, center_i]   (vector over destination states ν′)
    #     row_full = np.ascontiguousarray(Gs_union[:, :, center_i, :])   # (Nunion, Nunion, npols)
    #     col_full = np.ascontiguousarray(Gs_union[:, :, :, center_i])   # (Nunion, Nunion, npols)

    #     # --- Cache λ index tuples per "nsites" value (since |S_i| varies with i)
    #     # We reproduce your λ tensor per size once, then reuse its index lists.
    #     if not hasattr(self, "_lam_idx_cache_varsize"):
    #         self._lam_idx_cache_varsize = {}  # key: nsites (int) -> dict[lam] = (a_idx,b_idx,c_idx,d_idx)

    #     def get_lam_indices(nsites_i: int):
    #         if nsites_i not in self._lam_idx_cache_varsize:
    #             ident = np.identity(nsites_i, dtype=int)
    #             ones  = np.ones((nsites_i, nsites_i, nsites_i, nsites_i), dtype=int)
    #             lamdas = (np.einsum('ac, abcd->abcd', ident, ones)
    #                     + np.einsum('bd, abcd->abcd', ident, ones)
    #                     - np.einsum('ad, abcd->abcd', ident, ones)
    #                     - np.einsum('bc, abcd->abcd', ident, ones)).astype(np.int8)
    #             idx_dict = {}
    #             for lam in lamdalist:
    #                 idxs = np.argwhere(lamdas == lam)
    #                 if idxs.size == 0:
    #                     idx_dict[lam] = (np.array([], dtype=np.intp),
    #                                     np.array([], dtype=np.intp),
    #                                     np.array([], dtype=np.intp),
    #                                     np.array([], dtype=np.intp))
    #                 else:
    #                     a_idx, b_idx, c_idx, d_idx = idxs.T
    #                     idx_dict[lam] = (a_idx.astype(np.intp), b_idx.astype(np.intp),
    #                                     c_idx.astype(np.intp), d_idx.astype(np.intp))
    #             del lamdas
    #             self._lam_idx_cache_varsize[nsites_i] = idx_dict
    #         return self._lam_idx_cache_varsize[nsites_i]

    #     # --- Accumulate gamma_plus EXACTLY like the baseline, but per-destination i
    #     t_acc = time.time()
    #     gamma_plus = np.zeros(npols, dtype=np.complex128)

    #     for i in range(npols):
    #         # site subset S_i for this destination (map to union positions)
    #         Sg = site_idxs_list[i]
    #         if Sg.size == 0:
    #             continue
    #         S = site_pos[Sg]   # positions in [0..Nunion-1]
    #         nsi = S.size

    #         # Views of the needed scalars for this i:
    #         # rows_S[a,b] = G[a,b][center_i, i]
    #         # cols_S[c,d] = G[c,d][i, center_i]
    #         rows_S = row_full[np.ix_(S, S, [i])][:, :, 0]  # (nsi, nsi)
    #         cols_S = col_full[np.ix_(S, S, [i])][:, :, 0]  # (nsi, nsi)

    #         # Get λ index tuples for this size nsi
    #         lam_idx = get_lam_indices(nsi)

    #         # Sum over (a,b,c,d) in S_i × S_i with the λ constraints (EXACT baseline algebra)
    #         for pos, lam in enumerate(lamdalist):
    #             a_idx, b_idx, c_idx, d_idx = lam_idx[lam]
    #             if a_idx.size == 0:
    #                 continue
    #             # contrib_i = sum_k rows_S[a_k, b_k] * cols_S[c_k, d_k]
    #             contrib_i = np.sum(rows_S[a_idx, b_idx] * cols_S[c_idx, d_idx])
    #             gamma_plus[i] += bath_integrals[pos][i] * contrib_i

    #     if self.time_verbose:
    #         print('time(gamma accumulation per-dest):', time.time() - t_acc, flush=True)

    #     # --- Outgoing rates exactly as before
    #     self.red_R_tensor = 2.0 * np.real(gamma_plus)
    #     rates = np.delete(self.red_R_tensor, center_i) / const.hbar
    #     final_pol_idxs = np.delete(pol_idxs, center_i)

    #     if self.time_verbose:
    #         print('time(total):', time.time() - t0_all, flush=True)

    #     print('rates', rates)

    #     return rates, final_pol_idxs, time.time() - t0_all








class OldRedfield(Unitary):
    """ class to compute the Redfield tensor
    """

    def __init__(self, hamiltonian, polaron_locations, kappa, r_hop, r_ove, time_verbose = True):

        self.ham = hamiltonian
        self.polaron_locations = polaron_locations         # polaron locations (in relative frame)

        self.kappa=kappa
        self.r_hop = r_hop
        self.r_ove = r_ove
        # set to true only when time to compute rates is desired
        self.time_verbose = time_verbose
        
 
    def get_idxs(self, center_idx):
        # location of center polaron i (given by idx center_idx) around which we have constructed the box
        center_coord = self.polaron_locations[center_idx]
        # (1) get indices of all polaron states j that are within r_hop of the center polaron 
        polaron_idxs = np.where(np.array([np.linalg.norm(polaron_pos - center_coord) for polaron_pos in self.polaron_locations]) < self.r_hop )[0]
        # (2) get indices of the site basis states that are within r_ove of the center polaron
        site_idxs = np.where(np.array([np.linalg.norm(site_pos - center_coord) for site_pos in self.ham.qd_lattice_rel]) < self.r_ove )[0]
        return polaron_idxs, site_idxs
    


    def debug_bath(self, center_idx):
        # transitions from center ν to each ν'
        pol_idxs, _ = self.get_idxs(center_idx)
        omega_vec = self.ham.omega_diff[pol_idxs, center_idx]

        # for lam in (-2, -1, 0, 1, 2):
        #     K = self.ham.spec.correlationFT(omega_vec, lam, self.kappa)  # shape (npols,)
        #     S = 2.0 * np.real(K)  # symmetrized spectrum; should be >= 0 elementwise
        #     print(f"λ={lam}: min 2Re[K]={S.min(): .3e}, max {S.max(): .3e}")


    def make_redfield_box(self, center_idx):

        # find polaron and site states r_hop and r_ove, respectively, away from center_idx
        pol_idxs, site_idxs = self.get_idxs(center_idx)
        npols = len(pol_idxs)
        nsites = len(site_idxs)
        print('npols, nsites', npols, nsites)
        # center idx in pol_idxs
        center_i = np.where(pol_idxs == center_idx)[0][0]

        # self.debug_bath(center_idx)


        # compute lambda tensor (Eq. (16))
        ident = np.identity(nsites)
        ones = np.ones((nsites, nsites, nsites, nsites))
        lamdas= (  np.einsum('ac, abcd->abcd', ident, ones) + np.einsum('bd, abcd->abcd', ident, ones)
                 - np.einsum('ad, abcd->abcd', ident, ones) - np.einsum('bc, abcd->abcd', ident, ones)
                )
        
        # compute integral of bath correlation function
        start = time.time()
        start_tot = start
        lamdalist = [-2.0, -1.0, 0.0, 1.0, 2.0]
        bath_integrals = []
        for lam in lamdalist:
            matrix = np.zeros(npols, dtype = np.complex128)
            if lam == 0:
                bath_integrals.append(matrix)
            else:
                for i in range(npols):
                    omega_ij = self.ham.omega_diff[i, center_idx]
                    matrix[i] = self.ham.spec.correlationFT(omega_ij, lam, self.kappa)
                bath_integrals.append(matrix)

        end = time.time()
        if self.time_verbose:
            print('time difference1', end - start, flush=True)

        # transform sysbath operators to eigenbasis
        start = time.time()
        Gs = np.zeros((nsites, nsites), dtype=object)
        for a, a_idx in enumerate(site_idxs):
            for b, b_idx in enumerate(site_idxs):
                Gs[a][b] = self.ham.site2eig( self.ham.sysbath[a_idx][b_idx] )[pol_idxs, :][:, pol_idxs]
        end = time.time()
        if self.time_verbose:
            print('time difference2', end - start, flush=True)
        
        #gamma_plus = np.zeros((ns, ns), dtype = np.complex128)
        start = time.time()
        gamma_plus = np.zeros(npols, dtype = np.complex128)
        for lamda in [-2, -1, 0, 1, 2]:
            indices = np.argwhere(lamdas == lamda)
            for abcd in indices:
                gamma_plus += np.multiply(bath_integrals[lamda + 2], 
                                  np.multiply(Gs[abcd[2]][abcd[3]].T[center_i], Gs[abcd[0]][abcd[1]][center_i]))
        end = time.time()
        if self.time_verbose:
            print('time difference3', end - start, flush=True)

        # only outgoing rates are relevant so we can disregard the delta-function
        # term in Eq. (19), we also need to remove the starting state (center_idx)
        self.red_R_tensor = 2 * np.real(gamma_plus)
        rates = np.delete(self.red_R_tensor, center_i) / const.hbar
        final_site_idxs = np.delete(pol_idxs, center_i)

        print('rates', rates)

        end_tot = time.time()
        if self.time_verbose:
            print('time difference (tot)', end_tot - start_tot, flush=True)


        # return (outgoing) rates and corresponding polaron idxs (final sites)
        return rates, final_site_idxs, end_tot - start_tot


    
    
    



