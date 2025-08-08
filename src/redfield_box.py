
import numpy as np
from . import const
import time

class Unitary(object):
    """A unitary evolution class
    """

    def __init__(self, hamiltonian):
        self.ham = hamiltonian

    def setup(self):
        pass

class NewRedfield(Unitary):
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
    
    def get_idxsNew(self, center_idx):
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
        
    # def make_redfield_box(self, center_idx):
    #     # --- setup
    #     pol_idxs, site_idxs = self.get_idxs(center_idx)
    #     pol1_idx, _ = self.get_idxsNew(center_idx)
    #     print('npols new', len(pol1_idx))
    #     npols = len(pol_idxs); nsites = len(site_idxs)
    #     if self.time_verbose:
    #         print('npols, nsites', npols, nsites)
    #     start_tot = time.time()
    #     center_i = int(np.where(pol_idxs == center_idx)[0][0])

    #     # --- cache λ-index sets for this nsites
    #     if not hasattr(self, "_lam_idx_cache"):
    #         self._lam_idx_cache = {}
    #     lamdalist = (-2.0, -1.0, 0.0, 1.0, 2.0)
    #     if nsites not in self._lam_idx_cache:
    #         ident = np.identity(nsites)
    #         ones  = np.ones((nsites, nsites, nsites, nsites))
    #         lamdas = (np.einsum('ac, abcd->abcd', ident, ones)
    #                 + np.einsum('bd, abcd->abcd', ident, ones)
    #                 - np.einsum('ad, abcd->abcd', ident, ones)
    #                 - np.einsum('bc, abcd->abcd', ident, ones))
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

    #     # --- (optional) cache flattened (a,b) indices to avoid recomputing each call
    #     if not hasattr(self, "_flat_idx_cache"):
    #         self._flat_idx_cache = {}
    #     if nsites not in self._flat_idx_cache:
    #         flat = {}
    #         for lam in lamdalist:
    #             a_idx, b_idx, c_idx, d_idx = idx_dict[lam]
    #             flat[lam] = ((a_idx * nsites + b_idx).astype(np.intp),
    #                         (c_idx * nsites + d_idx).astype(np.intp))
    #         self._flat_idx_cache[nsites] = flat
    #     flat = self._flat_idx_cache[nsites]

    #     # --- bath integrals (KEEP your exact local indexing)
    #     t0 = time.time()
    #     bath_integrals = []
    #     for lam in lamdalist:
    #         vec = np.zeros(npols, dtype=np.complex128)
    #         if lam != 0.0:
    #             for i in range(npols):  # local index on purpose
    #                 omega_ij = self.ham.omega_diff[i, center_idx]
    #                 vec[i] = self.ham.spec.correlationFT(omega_ij, lam, self.kappa)
    #         bath_integrals.append(vec)
    #     if self.time_verbose:
    #         print('time(bath integrals)', time.time() - t0, flush=True)

    #     # --- transform sysbath operators to eigenbasis (same slicing as before)
    #     t1 = time.time()
    #     Gs = np.empty((nsites, nsites, npols, npols), dtype=np.complex128)
    #     for aa, a_idx in enumerate(site_idxs):
    #         for bb, b_idx in enumerate(site_idxs):
    #             G_full = self.ham.site2eig(self.ham.sysbath[a_idx][b_idx])
    #             Gs[aa, bb] = G_full[np.ix_(pol_idxs, pol_idxs)]
    #     if self.time_verbose:
    #         print('time(site→eig)', time.time() - t1, flush=True)

    #     # --- PREP: make center row/col contiguous and flatten (a,b)→ab
    #     AB = nsites * nsites
    #     # center row: Gs[:, :, center_i, :] -> (ns,ns,npols) -> (AB,npols)
    #     Gs_c_row_flat = np.ascontiguousarray(Gs[:, :, center_i, :].reshape(AB, npols))
    #     # center col: Gs[:, :, :, center_i] -> (ns,ns,npols) -> (AB,npols)
    #     Gs_c_col_flat = np.ascontiguousarray(Gs[:, :, :, center_i].reshape(AB, npols))

    #     # --- vectorized accumulation over λ using flattened takes + einsum
    #     t2 = time.time()
    #     gamma_plus = np.zeros(npols, dtype=np.complex128)

    #     for lam_idx, lam in enumerate(lamdalist):
    #         ab_flat, cd_flat = flat[lam]
    #         if ab_flat.size == 0:
    #             continue
    #         rows = Gs_c_row_flat.take(ab_flat, axis=0)  # (K, npols)
    #         cols = Gs_c_col_flat.take(cd_flat, axis=0)  # (K, npols)
    #         # contrib[n] = sum_k rows[k,n]*cols[k,n]
    #         contrib = np.einsum('kn,kn->n', rows, cols, optimize=True)
    #         gamma_plus += bath_integrals[lam_idx] * contrib

    #     if self.time_verbose:
    #         print('time(gamma accumulation)', time.time() - t2, flush=True)

    #     # --- outgoing rates (unchanged)
    #     self.red_R_tensor = 2.0 * np.real(gamma_plus)
    #     rates = np.delete(self.red_R_tensor, center_i) / const.hbar
    #     final_site_idxs = np.delete(pol_idxs, center_i)

    #     if self.time_verbose:
    #         print('time(total)', time.time() - start_tot, flush=True)

    #     return rates, final_site_idxs, time.time() - start_tot

    def make_redfield_box(self, center_idx):
        # --- neighborhoods from the new function
        pol_idxs, site_idxs_list = self.get_idxsNew(center_idx)   # NEW
        npols = len(pol_idxs)
        if self.time_verbose:
            print('npols', npols)
        start_tot = time.time()
        if npols == 0:
            return np.array([], dtype=float), np.array([], dtype=int), 0.0

        # local index of center in local polaron list
        center_i = int(np.where(pol_idxs == center_idx)[0][0])

        # ---- build union of all site indices referenced by any S_i, and a map to 0..Nunion-1
        if len(site_idxs_list):
            union_sites = np.unique(np.concatenate(site_idxs_list))
        else:
            union_sites = np.array([], dtype=int)
        Nunion = len(union_sites)
        # map global site index -> position in union
        site_pos = -np.ones(len(self.ham.qd_lattice_rel), dtype=int)
        site_pos[union_sites] = np.arange(Nunion, dtype=int)

        # ---- λ index bins as INTEGERS (avoid float equality)
        lamdalist = (-2, -1, 0, 1, 2)

        # ---- bath integrals: vector over final states i for each λ
        # NOTE: Use GLOBAL ω indexing (safer): ω_{ν'ν} with ν' = pol_idxs[i]
        t0 = time.time()
        B = []
        for lam in lamdalist:
            if lam == 0:
                B.append(np.zeros(npols, dtype=np.complex128))
            else:
                # if your correlationFT accepts arrays, this is fast; otherwise loop i
                omegas = self.ham.omega_diff[pol_idxs, center_idx]    # GLOBAL ν'
                B.append(self.ham.spec.correlationFT(omegas, lam, self.kappa))
        if self.time_verbose:
            print('time(bath integrals)', time.time() - t0, flush=True)

        # ---- build only the needed slices of G_ab in eigenbasis for all (a,b) in union
        # Row[a,b,i] = G_ab[center_i, i], Col[a,b,i] = G_ab[i, center_i]
        t1 = time.time()
        Row = np.empty((Nunion, Nunion, npols), dtype=np.complex128)
        Col = np.empty((Nunion, Nunion, npols), dtype=np.complex128)
        for ai, a in enumerate(union_sites):
            for bi, b in enumerate(union_sites):
                Vab_eig = self.ham.site2eig(self.ham.sysbath[a][b])     # full eig-op in eigenbasis
                G_loc   = Vab_eig[np.ix_(pol_idxs, pol_idxs)]            # restrict to local polaron box
                Row[ai, bi, :] = G_loc[center_i, :]                      # row at ν=center
                Col[ai, bi, :] = G_loc[:, center_i]                      # col at ν=center
        if self.time_verbose:
            print('time(site→eig slices)', time.time() - t1, flush=True)

        # ---- δ-reduced λ contraction on each S_i (exact algebra, fewer terms)
        # For a given i with subset S (mapped to union positions), the structure factor is:
        #   term_ac = (Σ_{a∈S} Col[a,a,i]) (Σ_{b∈S} Row[b,b,i])
        #   term_ad = Σ_{d∈S} ( Σ_{n∈S} Row[d,n,i] ) ( Σ_{m∈S} Col[m,d,i] )
        #   S_struct(i) = 2 * (term_ac - term_ad)
        t2 = time.time()
        S_struct = np.zeros(npols, dtype=np.complex128)
        for i in range(npols):
            Sg = site_idxs_list[i]
            if Sg.size == 0:
                continue
            S = site_pos[Sg]  # map to union positions

            # diagonals over S
            diag_row = Row[S, S, i]   # shape (|S|,)
            diag_col = Col[S, S, i]   # shape (|S|,)

            # sums over S for each fixed d∈S
            Rs = Row[np.ix_(S, S, [i])]  # (|S|, |S|, 1)
            Cs = Col[np.ix_(S, S, [i])]  # (|S|, |S|, 1)
            row_sum = Rs[:, :, 0].sum(axis=1)   # (|S|,)
            col_sum = Cs[:, :, 0].sum(axis=0)   # (|S|,)

            term_ac = diag_col.sum() * diag_row.sum()
            term_ad = np.dot(row_sum, col_sum)
            S_struct[i] = 2.0 * (term_ac - term_ad)
        if self.time_verbose:
            print('time(δ-reduced accumulation)', time.time() - t2, flush=True)

        # ---- assemble γ⁺(i) = Σ_λ B_λ(i) * S_struct(i)
        gamma_plus = np.zeros(npols, dtype=np.complex128)
        for k in range(len(lamdalist)):
            gamma_plus += B[k] * S_struct

        # ---- outgoing rates (drop self term, divide by ħ)
        self.red_R_tensor = 2.0 * np.real(gamma_plus)
        rates = np.delete(self.red_R_tensor, center_i) / const.hbar
        final_pol_idxs = np.delete(pol_idxs, center_i)

        if self.time_verbose:
            print('time(total)', time.time() - start_tot, flush=True)
        
        print('rates', len(rates))

        return rates, final_pol_idxs, time.time() - start_tot












class Redfield(Unitary):
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
        
    def make_redfield_box(self, center_idx):

        # find polaron and site states r_hop and r_ove, respectively, away from center_idx
        pol_idxs, site_idxs = self.get_idxs(center_idx)
        npols = len(pol_idxs)
        nsites = len(site_idxs)
        print('npols, nsites', npols, nsites)
        # center idx in pol_idxs
        center_i = np.where(pol_idxs == center_idx)[0][0]


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


    
    
    



