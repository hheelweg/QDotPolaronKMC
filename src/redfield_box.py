
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
        # get local polaron/site indices and locate the local index of the center state
        pol_idxs, site_idxs = self.get_idxs(center_idx)
        npols, nsites = len(pol_idxs), len(site_idxs)
        center_i = int(np.where(pol_idxs == center_idx)[0][0])

        if self.time_verbose:
            print('npols, nsites', npols, nsites)

        t0 = time.time()

        # ---- (1) Build G_ab(ν,ν') = <ν| V_ab |ν'> in eigenbasis, restricted to the polaron box
        #     Store as a dense array: Gs[a,b,ν,ν']
        Gs = np.empty((nsites, nsites, npols, npols), dtype=np.complex128)
        U = self.ham.Umat  # if you have it handy; otherwise site2eig does the transform
        for ai, a in enumerate(site_idxs):
            for bi, b in enumerate(site_idxs):
                # site-basis system-bath operator block V_ab
                Vab_site = self.ham.sysbath[a][b]
                # eigenbasis block (full)
                Vab_eig = self.ham.site2eig(Vab_site)
                # restrict to local polaron indices
                Gs[ai, bi, :, :] = Vab_eig[np.ix_(pol_idxs, pol_idxs)]

        if self.time_verbose:
            print('time(G build) =', time.time() - t0, flush=True)

        # ---- (2) Precompute bath integrals B_λ(ω_{ν'ν}) for λ ∈ {-2,-1,0,1,2}
        lamvals = (-2, -1, 0, 1, 2)
        B = {lam: np.zeros(npols, dtype=np.complex128) for lam in lamvals}
        for i in range(npols):
            g = pol_idxs[i]                               # global ν'
            omega = self.ham.omega_diff[g, center_idx]    # ω_{ν'ν} = E_{ν'}-E_ν
            for lam in lamvals:
                if lam == 0:
                    B[lam][i] = 0.0
                else:
                    B[lam][i] = self.ham.spec.correlationFT(omega, lam, self.kappa)

        if self.time_verbose:
            print('time(B integrals) =', time.time() - t0, flush=True)

        # ---- (3) Build the needed rows/columns at the initial state ν=center_i
        # A[a,b,:] = row vector over ν' of G_ab(ν,ν')
        A = Gs[:, :, center_i, :]                         # shape (nsites, nsites, npols)

        # Diagonals A[a,a,:] (we’ll need sums of them)
        diagA = A[np.arange(nsites), np.arange(nsites), :]  # shape (nsites, npols)

        # ---- (4) Contract λ-structure without forming λ_{..} explicitly
        # term1:  +δ_{mn}     ->  (sum_m A_mm^*) * (sum_n A_nn)
        sum_diag_conj = np.sum(diagA.conj(), axis=0)      # shape (npols,)
        sum_diag      = np.sum(diagA,       axis=0)       # shape (npols,)
        term1 = sum_diag_conj * sum_diag                  # npols-vector

        # term2:  +δ_{m'n'}   ->  identical structure as term1
        term2 = term1

        # term3:  -δ_{mn'}    ->  sum_{n'} (sum_n A_{n'n}^*) * (sum_m A_{mn'})
        # implement by summing rows/cols at fixed n'
        term3 = np.zeros(npols, dtype=np.complex128)
        for nprime in range(nsites):
            row_star = np.sum(A[nprime, :, :].conj(), axis=0)  # sum over n
            col      = np.sum(A[:, nprime, :],       axis=0)  # sum over m
            term3 += row_star * col

        # term4:  -δ_{m'n}    ->  symmetric to term3
        term4 = np.zeros(npols, dtype=np.complex128)
        for n_ in range(nsites):
            row_star = np.sum(A[:, n_, :].conj(), axis=0)      # sum over n'
            col      = np.sum(A[n_, :, :],       axis=0)       # sum over m'
            term4 += row_star * col

        # Structure factor S(ν') after λ-combination (all four pieces share same G-combination)
        S = (term1 + term2) - (term3 + term4)                  # shape (npols,)

        # ---- (5) Assemble Γ^+ (ν'←ν) by summing over λ≠0
        gamma_plus = np.zeros(npols, dtype=np.complex128)
        for lam in lamvals:
            if lam != 0:
                gamma_plus += B[lam] * S

        # ---- (6) Outgoing rates: R_{ν'ν} = 2 Re[γ^+] / ħ, drop ν'=ν
        R_out = 2.0 * np.real(gamma_plus)
        rates = np.delete(R_out, center_i) / const.hbar
        final_pol_idxs = np.delete(pol_idxs, center_i)

        if self.time_verbose:
            print('time(total) =', time.time() - t0, flush=True)
        
        print('rates shape', rates.shape)

        return rates, final_pol_idxs, time.time() - t0



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

        end_tot = time.time()
        if self.time_verbose:
            print('time difference (tot)', end_tot - start_tot, flush=True)


        # return (outgoing) rates and corresponding polaron idxs (final sites)
        return rates, final_site_idxs, end_tot - start_tot


    
    
    



