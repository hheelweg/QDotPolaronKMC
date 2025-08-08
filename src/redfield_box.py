
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
    #     print('site_idxs', site_idxs)
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
        """
        General path using get_idxsNew(center_idx).
        - Per-destination site sets S_i (polaron-dependent nsites).
        - EXACT same physics as your validated vectorized function:
            * same λ list, same weighting
            * same row/col orientation
            * same LOCAL omega indexing (omega_diff[i, center_idx])
        but the a,b,c,d sums are restricted to S_i x S_i for each i.
        Returns:
            rates, final_pol_idxs (GLOBAL eigenstate indices), walltime
        """
        t0_all = time.time()

        # --- neighborhoods (polaron subset and per-destination site sets)
        pol_idxs, site_idxs_list = self.get_idxsNew(center_idx)   # NEW function
        npols = len(pol_idxs)
        if self.time_verbose:
            print('npols', npols, [len(arr) for arr in site_idxs_list])
        if npols == 0:
            return np.array([], dtype=float), np.array([], dtype=int), time.time() - t0_all

        # local index of the center eigenstate within this polaron subset
        center_i = int(np.where(pol_idxs == center_idx)[0][0])

        # Build the union of sites across all S_i so we can transform once
        union_sites = np.unique(np.concatenate(site_idxs_list)) if len(site_idxs_list) else np.array([], dtype=int)
        Nunion = len(union_sites)

        # Map global site index -> position in union [0..Nunion-1]
        site_pos = -np.ones(len(self.ham.qd_lattice_rel), dtype=int)
        site_pos[union_sites] = np.arange(Nunion, dtype=int)

        # --- Bath integrals (KEEP the baseline's LOCAL omega indexing!)
        lamdalist = (-2, -1, 0, 1, 2)
        t_bath = time.time()
        bath_integrals = []
        for lam in lamdalist:
            vec = np.zeros(npols, dtype=np.complex128)
            if lam != 0:
                for i in range(npols):  # LOCAL i (0..npols-1) on purpose to match baseline
                    omega_ij = self.ham.omega_diff[i, center_idx]
                    vec[i] = self.ham.spec.correlationFT(omega_ij, lam, self.kappa)
            bath_integrals.append(vec)
        if self.time_verbose:
            print('time(bath integrals):', time.time() - t_bath, flush=True)

        # --- Transform sysbath operators to eigenbasis ONCE over the union sites
        #     and restrict to the local polaron subspace (pol_idxs).
        #     We'll later pick [center_i, i] and [i, center_i] entries from these.
        t_g = time.time()
        # Gs_union[a,b,:,:] = (U† V_ab U)[pol_idxs, pol_idxs]
        Gs_union = np.empty((Nunion, Nunion, npols, npols), dtype=np.complex128)
        for ai, a in enumerate(union_sites):
            for bi, b in enumerate(union_sites):
                G_full = self.ham.site2eig(self.ham.sysbath[a][b])      # full eigbasis operator
                Gs_union[ai, bi] = G_full[np.ix_(pol_idxs, pol_idxs)]   # restrict to local polaron box
        if self.time_verbose:
            print('time(site→eig over union):', time.time() - t_g, flush=True)

        # Prepare convenient row/col views over the union (contiguous helps)
        # row_full[ai,bi,:] = G[a,b][center_i, :]   (vector over destination states ν′)
        # col_full[ai,bi,:] = G[a,b][:, center_i]   (vector over destination states ν′)
        row_full = np.ascontiguousarray(Gs_union[:, :, center_i, :])   # (Nunion, Nunion, npols)
        col_full = np.ascontiguousarray(Gs_union[:, :, :, center_i])   # (Nunion, Nunion, npols)

        # --- Cache λ index tuples per "nsites" value (since |S_i| varies with i)
        # We reproduce your λ tensor per size once, then reuse its index lists.
        if not hasattr(self, "_lam_idx_cache_varsize"):
            self._lam_idx_cache_varsize = {}  # key: nsites (int) -> dict[lam] = (a_idx,b_idx,c_idx,d_idx)

        def get_lam_indices(nsites_i: int):
            if nsites_i not in self._lam_idx_cache_varsize:
                ident = np.identity(nsites_i, dtype=int)
                ones  = np.ones((nsites_i, nsites_i, nsites_i, nsites_i), dtype=int)
                lamdas = (np.einsum('ac, abcd->abcd', ident, ones)
                        + np.einsum('bd, abcd->abcd', ident, ones)
                        - np.einsum('ad, abcd->abcd', ident, ones)
                        - np.einsum('bc, abcd->abcd', ident, ones)).astype(np.int8)
                idx_dict = {}
                for lam in lamdalist:
                    idxs = np.argwhere(lamdas == lam)
                    if idxs.size == 0:
                        idx_dict[lam] = (np.array([], dtype=np.intp),
                                        np.array([], dtype=np.intp),
                                        np.array([], dtype=np.intp),
                                        np.array([], dtype=np.intp))
                    else:
                        a_idx, b_idx, c_idx, d_idx = idxs.T
                        idx_dict[lam] = (a_idx.astype(np.intp), b_idx.astype(np.intp),
                                        c_idx.astype(np.intp), d_idx.astype(np.intp))
                del lamdas
                self._lam_idx_cache_varsize[nsites_i] = idx_dict
            return self._lam_idx_cache_varsize[nsites_i]

        # --- Accumulate gamma_plus EXACTLY like the baseline, but per-destination i
        t_acc = time.time()
        gamma_plus = np.zeros(npols, dtype=np.complex128)

        for i in range(npols):
            # site subset S_i for this destination (map to union positions)
            Sg = site_idxs_list[i]
            if Sg.size == 0:
                continue
            S = site_pos[Sg]   # positions in [0..Nunion-1]
            nsi = S.size

            # Views of the needed scalars for this i:
            # rows_S[a,b] = G[a,b][center_i, i]
            # cols_S[c,d] = G[c,d][i, center_i]
            rows_S = row_full[np.ix_(S, S, [i])][:, :, 0]  # (nsi, nsi)
            cols_S = col_full[np.ix_(S, S, [i])][:, :, 0]  # (nsi, nsi)

            # Get λ index tuples for this size nsi
            lam_idx = get_lam_indices(nsi)

            # Sum over (a,b,c,d) in S_i × S_i with the λ constraints (EXACT baseline algebra)
            for pos, lam in enumerate(lamdalist):
                a_idx, b_idx, c_idx, d_idx = lam_idx[lam]
                if a_idx.size == 0:
                    continue
                # contrib_i = sum_k rows_S[a_k, b_k] * cols_S[c_k, d_k]
                contrib_i = np.sum(rows_S[a_idx, b_idx] * cols_S[c_idx, d_idx])
                gamma_plus[i] += bath_integrals[pos][i] * contrib_i

        if self.time_verbose:
            print('time(gamma accumulation per-dest):', time.time() - t_acc, flush=True)

        # --- Outgoing rates exactly as before
        self.red_R_tensor = 2.0 * np.real(gamma_plus)
        rates = np.delete(self.red_R_tensor, center_i) / const.hbar
        final_pol_idxs = np.delete(pol_idxs, center_i)

        if self.time_verbose:
            print('time(total):', time.time() - t0_all, flush=True)

        print('rates', rates)

        return rates, final_pol_idxs, time.time() - t0_all















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


    
    
    



