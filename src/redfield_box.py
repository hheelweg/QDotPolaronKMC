
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
        # --- setup
        pol_idxs, site_idxs = self.get_idxs(center_idx)
        npols, nsites = len(pol_idxs), len(site_idxs)
        if self.time_verbose:
            print('npols, nsites', npols, nsites)
        t0 = time.time()

        # local index of center eigenstate in the polaron subspace (unchanged)
        center_i = int(np.where(pol_idxs == center_idx)[0][0])

        # pointers into eigenvector matrix
        U = self.ham.Umat                      # shape (Nsite_total, Neig)
        u_c = U[:, center_idx]                 # site-space vector for center state
        U_P = U[:, pol_idxs]                   # site→local polaron subspace

        # --- bath integrals (keep your original local-ω indexing exactly)
        lamdalist = (-2.0, -1.0, 0.0, 1.0, 2.0)
        bath_integrals = []
        for lam in lamdalist:
            vec = np.zeros(npols, dtype=np.complex128)
            if lam != 0.0:
                # WARNING: preserves your original local i → omega_diff[i, center_idx]
                for i in range(npols):
                    omega_ij = self.ham.omega_diff[i, center_idx]
                    vec[i] = self.ham.spec.correlationFT(omega_ij, lam, self.kappa)
            bath_integrals.append(vec)

        if self.time_verbose:
            print('time(bath) =', time.time() - t0, flush=True)

        # --- build just the needed Row/Col slices using re-association
        # Row[a,b,:] = u_c^† V_ab U_P         (1 × npols)
        # Col[a,b,:] = U_P^† (V_ab u_c)       (npols × 1)
        Row = np.empty((nsites, nsites, npols), dtype=np.complex128)
        Col = np.empty((nsites, nsites, npols), dtype=np.complex128)

        t1 = time.time()
        # To help BLAS, ensure contiguity
        u_cC = np.ascontiguousarray(u_c)
        U_PC = np.ascontiguousarray(U_P)

        for ai, a in enumerate(site_idxs):
            for bi, b in enumerate(site_idxs):
                Vab = self.ham.sysbath[a][b]                    # site×site
                # Compute once per (a,b):
                #   T1 = Vab @ U_P      (site × npols)
                #   t2 = Vab @ u_c      (site,)
                T1 = Vab @ U_PC
                t2 = Vab @ u_cC
                # Row: (u_c^†) @ T1  -> shape (npols,)
                Row[ai, bi, :] = u_cC.conj().T @ T1
                # Col: (U_P^†) @ t2  -> shape (npols,)
                Col[ai, bi, :] = U_PC.conj().T @ t2

        if self.time_verbose:
            print('time(Row/Col) =', time.time() - t1, flush=True)

        # --- EXACT λ (Kronecker-delta) reduction, identical algebra to your lamdas tensor
        # S(ν') =  Σ_{ab,cd} (δ_ac + δ_bd - δ_ad - δ_bc) * Col[c,d,ν'] * Row[a,b,ν']
        # Expand to four aggregated terms (no extra conj, same order as your code):

        diag_Row = Row[np.arange(nsites), np.arange(nsites), :]     # (nsites, npols)
        diag_Col = Col[np.arange(nsites), np.arange(nsites), :]

        term_ac = np.sum(diag_Col, axis=0) * np.sum(diag_Row, axis=0)   # +δ_ac
        term_bd = term_ac                                               # +δ_bd (same shape)

        # -δ_ad: sum_d ( sum_b Row[d,b,:] ) * ( sum_a Col[a,d,:] )
        sum_row_over_b = np.sum(Row, axis=1)    # (nsites, npols)  collapse b
        sum_col_over_a = np.sum(Col, axis=0)    # (nsites, npols)  collapse a
        term_ad = np.sum(sum_row_over_b * sum_col_over_a, axis=0)

        # -δ_bc: symmetric to above (same tensors), so identical:
        term_bc = term_ad

        S = (term_ac + term_bd) - (term_ad + term_bc)                 # (npols,)

        # --- assemble γ⁺ exactly like your original lamda loop
        gamma_plus = np.zeros(npols, dtype=np.complex128)
        for k, lam in enumerate(lamdalist):
            gamma_plus += bath_integrals[k] * S

        # --- outgoing rates (unchanged)
        self.red_R_tensor = 2.0 * np.real(gamma_plus)
        rates = np.delete(self.red_R_tensor, center_i) / const.hbar
        final_site_idxs = np.delete(pol_idxs, center_i)

        if self.time_verbose:
            print('time(total) =', time.time() - t0, flush=True)
        
        print('rates', rates)

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


    
    
    



