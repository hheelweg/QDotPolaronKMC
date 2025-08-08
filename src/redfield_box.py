
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
        start_tot = time.time()
        center_i = int(np.where(pol_idxs == center_idx)[0][0])

        # --- cache λ-index sets once per nsites (identical to your original λ-structure)
        if not hasattr(self, "_lam_idx_cache"):
            self._lam_idx_cache = {}
        lamdalist = (-2.0, -1.0, 0.0, 1.0, 2.0)
        if nsites not in self._lam_idx_cache:
            ident = np.identity(nsites)
            ones  = np.ones((nsites, nsites, nsites, nsites))
            lamdas = (np.einsum('ac, abcd->abcd', ident, ones)
                    + np.einsum('bd, abcd->abcd', ident, ones)
                    - np.einsum('ad, abcd->abcd', ident, ones)
                    - np.einsum('bc, abcd->abcd', ident, ones))
            idx_dict = {}
            for lam in lamdalist:
                idxs = np.argwhere(lamdas == lam)
                if idxs.size == 0:
                    idx_dict[lam] = (np.array([], dtype=int),
                                    np.array([], dtype=int),
                                    np.array([], dtype=int),
                                    np.array([], dtype=int))
                else:
                    a_idx, b_idx, c_idx, d_idx = idxs.T
                    idx_dict[lam] = (a_idx, b_idx, c_idx, d_idx)
            del lamdas
            self._lam_idx_cache[nsites] = idx_dict
        idx_dict = self._lam_idx_cache[nsites]

        # --- bath integrals: vectorized over local final-state index (exact same indexing as before)
        t0 = time.time()
        omegas_local = self.ham.omega_diff[:npols, center_idx]  # preserves your local-i convention
        bath_integrals = []
        for lam in lamdalist:
            if lam == 0.0:
                bath_integrals.append(np.zeros(npols, dtype=np.complex128))
            else:
                bath_integrals.append(self.ham.spec.correlationFT(omegas_local, lam, self.kappa))
        if self.time_verbose:
            print('time(bath integrals)', time.time() - t0, flush=True)

        # --- build only the needed slices Gs[a,b][center_i,:] and Gs[a,b][:,center_i]
        #     cache per (site_idxs, pol_idxs, center_idx) box
        t1 = time.time()
        key = (tuple(site_idxs.tolist()), tuple(pol_idxs.tolist()), int(center_idx))
        if not hasattr(self, "_Gslice_cache"):
            self._Gslice_cache = {}
        cached = self._Gslice_cache.get(key, None)
        if cached is not None:
            Row, Col = cached
        else:
            Row = np.empty((nsites, nsites, npols), dtype=np.complex128)  # rows: G[a,b][center_i,:]
            Col = np.empty((nsites, nsites, npols), dtype=np.complex128)  # cols: G[a,b][:,center_i]
            for aa, a_idx in enumerate(site_idxs):
                for bb, b_idx in enumerate(site_idxs):
                    G_full = self.ham.site2eig(self.ham.sysbath[a_idx][b_idx])   # full eig op
                    G_loc  = G_full[np.ix_(pol_idxs, pol_idxs)]                   # restrict to box
                    Row[aa, bb, :] = G_loc[center_i, :]                           # exact row as before
                    Col[aa, bb, :] = G_loc[:, center_i]                           # exact col as before
            self._Gslice_cache[key] = (Row, Col)
        if self.time_verbose:
            print('time(site→eig slices)', time.time() - t1, flush=True)

        # --- vectorized accumulation over λ (identical algebra/order to your original)
        t2 = time.time()
        gamma_plus = np.zeros(npols, dtype=np.complex128)
        for lam_idx, lam in enumerate(lamdalist):
            a_idx, b_idx, c_idx, d_idx = idx_dict[lam]
            if a_idx.size == 0:
                continue
            rows = Row[a_idx, b_idx, :]      # shape (K, npols)  == Gs[a][b][center_i]
            cols = Col[c_idx, d_idx, :]      # shape (K, npols)  == Gs[c][d].T[center_i]
            contrib = np.sum(rows * cols, axis=0)   # (npols,)
            gamma_plus += bath_integrals[lam_idx] * contrib

        if self.time_verbose:
            print('time(gamma accumulation)', time.time() - t2, flush=True)

        # --- outgoing rates (unchanged)
        self.red_R_tensor = 2.0 * np.real(gamma_plus)
        rates = np.delete(self.red_R_tensor, center_i) / const.hbar
        final_site_idxs = np.delete(pol_idxs, center_i)

        if self.time_verbose:
            print('time(total)', time.time() - start_tot, flush=True)
        
        print('rates', rates)

        return rates, final_site_idxs, time.time() - start_tot









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


    
    
    



