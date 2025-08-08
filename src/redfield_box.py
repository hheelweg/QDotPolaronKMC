
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
        
    def make_redfield_box(self, center_idx):
        # --- (0) setup
        pol_idxs, site_idxs = self.get_idxs(center_idx)
        npols, nsites = len(pol_idxs), len(site_idxs)
        if self.time_verbose:
            print('npols, nsites', npols, nsites)
        t0 = time.time()

        # local index of the center eigenstate within the polaron subspace
        center_i = int(np.where(pol_idxs == center_idx)[0][0])

        # --- (1) Bath integrals: use linear-in-λ shortcut
        # Build B1(ω_{ν'ν}) for all ν' (global index for ω!)
        B1 = np.zeros(npols, dtype=np.complex128)
        for i in range(npols):
            g = pol_idxs[i]                              # global ν'
            omega = self.ham.omega_diff[g, center_idx]   # ω_{ν'ν} = E_{ν'} - E_ν
            B1[i] = self.ham.spec.correlationFT(omega, 1, self.kappa)
        # If you ever switch bath method, you can fall back to a slower path by checking linearity:
        # assert np.allclose(self.ham.spec.correlationFT(omega, 2, self.kappa), 2*B1[i])

        if self.time_verbose:
            print('time(bath) =', time.time() - t0, flush=True)

        # --- (2) Build only the row/column you actually use from each G_ab
        # Row[a,b,:] = G_ab[center_i, :]    (your Gs[ab][center_i])
        # Col[c,d,:] = G_cd[:, center_i]    (your Gs[cd].T[center_i] == G_cd[:, center_i])
        Row = np.empty((nsites, nsites, npols), dtype=np.complex128)
        Col = np.empty((nsites, nsites, npols), dtype=np.complex128)

        for ai, a in enumerate(site_idxs):
            for bi, b in enumerate(site_idxs):
                Vab_site = self.ham.sysbath[a][b]
                Vab_eig  = self.ham.site2eig(Vab_site)    # full eigenbasis operator
                # grab only needed slices w.r.t. global eigen indices
                Row[ai, bi, :] = Vab_eig[center_idx, pol_idxs]   # row at ν=center
                Col[ai, bi, :] = Vab_eig[pol_idxs,   center_idx] # column at ν=center

        if self.time_verbose:
            print('time(G-slices) =', time.time() - t0, flush=True)

        # --- (3) Contract λ-structure WITHOUT building λ_{abcd}
        # term1: +δ_ac     ->  (sum_a Col[a,*,:]) dot (sum_b Row[*,b,:]) but only along the diagonals a==c
        # In practice: use ONLY diagonals for δ_ac and δ_bd
        diag_Row = Row[np.arange(nsites), np.arange(nsites), :]  # shape (nsites, npols)
        diag_Col = Col[np.arange(nsites), np.arange(nsites), :]

        sum_diag_row = np.sum(diag_Row, axis=0)   # shape (npols,)
        sum_diag_col = np.sum(diag_Col, axis=0)   # shape (npols,)

        term1 = sum_diag_col.conj() * sum_diag_row    # +δ_ac
        term2 = term1                                 # +δ_bd  (same algebraic form)

        # term3: -δ_ad  ->  sum_{d} (sum_n A_{dn}^*) * (sum_m A_{md})
        term3 = np.zeros(npols, dtype=np.complex128)
        for d in range(nsites):
            row_star = np.sum(Row[d, :, :], axis=0).conj()
            col_sum  = np.sum(Col[:, d, :], axis=0)
            term3 += row_star * col_sum

        # term4: -δ_bc  ->  sum_{c} (sum_{n'} A_{cn'}^*) * (sum_{m'} A_{m'c})
        term4 = np.zeros(npols, dtype=np.complex128)
        for c in range(nsites):
            row_star = np.sum(Row[c, :, :], axis=0).conj()
            col_sum  = np.sum(Col[:, c, :], axis=0)
            term4 += row_star * col_sum

        S = (term1 + term2) - (term3 + term4)          # structure factor (npols,)

        # --- (4) Assemble γ⁺ and outgoing rates
        gamma_plus = B1 * S                             # uses linear-in-λ identity
        R_out = 2.0 * np.real(gamma_plus)              # Eq. (19) real part
        rates = np.delete(R_out, center_i) / const.hbar
        final_site_idxs = np.delete(pol_idxs, center_i)

        if self.time_verbose:
            print('time(total) =', time.time() - t0, flush=True)

        return rates, final_site_idxs, time.time() - t0




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


    
    
    



