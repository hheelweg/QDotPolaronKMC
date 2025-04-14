
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

class Redfield(Unitary):
    """class to compute the Redfield tensor
    """

    def __init__(self, hamiltonian, kappa, r_hop, r_ove, time_verbose = False):

        self.ham = hamiltonian
        self.kappa=kappa
        self.r_hop = r_hop
        self.r_ove = r_ove
        # set to true only when time to compute rates is desired
        self.time_verbose = time_verbose
        
        # polaron locations (in relative frame)
        self.polaron_locations = np.matmul(self.ham.Umat ** 2, self.ham.qd_lattice_rel)
 
    
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
            print('time difference1', end - start)

        # transform sysbath operators to eigenbasis
        start = time.time()
        Gs = np.zeros((nsites, nsites), dtype=object)
        for a, a_idx in enumerate(site_idxs):
            for b, b_idx in enumerate(site_idxs):
                Gs[a][b] = self.ham.site2eig( self.ham.sysbath[a_idx][b_idx] )[pol_idxs, :][:, pol_idxs]
        end = time.time()
        if self.time_verbose:
            print('time difference2', end - start)
        
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
            print('time difference3', end - start)

        # only outgoing rates are relevant so we can disregard the delta-function
        # term in Eq. (19), we also need to remove the starting state (center_idx)
        self.red_R_tensor = 2 * np.real(gamma_plus)
        rates = np.delete(self.red_R_tensor, center_i) / const.hbar
        final_site_idxs = np.delete(pol_idxs, center_i)

        # return (outgoing) rates and corresponding polaron idxs (final sites)
        return rates, final_site_idxs

class NewRedfield(Unitary):
    """ New class to compute the Redfield tensor
    """

    def __init__(self, hamiltonian, polaron_locations, kappa, r_hop, r_ove, time_verbose = False):

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
            print('time difference1', end - start)

        # transform sysbath operators to eigenbasis
        start = time.time()
        Gs = np.zeros((nsites, nsites), dtype=object)
        for a, a_idx in enumerate(site_idxs):
            for b, b_idx in enumerate(site_idxs):
                Gs[a][b] = self.ham.site2eig( self.ham.sysbath[a_idx][b_idx] )[pol_idxs, :][:, pol_idxs]
        end = time.time()
        if self.time_verbose:
            print('time difference2', end - start)
        
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
            print('time difference3', end - start)

        # only outgoing rates are relevant so we can disregard the delta-function
        # term in Eq. (19), we also need to remove the starting state (center_idx)
        self.red_R_tensor = 2 * np.real(gamma_plus)
        rates = np.delete(self.red_R_tensor, center_i) / const.hbar
        final_site_idxs = np.delete(pol_idxs, center_i)

        # return (outgoing) rates and corresponding polaron idxs (final sites)
        return rates, final_site_idxs

# HH : disregard this unless you want to compute the full Redfield rate matrix
class RedfieldFull(Unitary):
    

    def __init__(self, hamiltonian, kappa):

        self.ham = hamiltonian
        self.kappa=kappa
    
    def make_redfield_full(self, center_idx):
        
        
        # (1) compute lambda tensor (Eq. (16))
        ns = self.ham.nsite
        ident = np.identity(ns)
        ones=np.ones((ns, ns, ns, ns))
        lamdas= (  np.einsum('ac, abcd->abcd', ident, ones) + np.einsum('bd, abcd->abcd', ident, ones)
                 - np.einsum('ad, abcd->abcd', ident, ones) - np.einsum('bc, abcd->abcd', ident, ones)
                          )
        
        # (2) compute integral of bath correlation function (Eq. (15))
        # (2a) compute omega_diffs for all polarons in polaron_idxs
        omega_diff = np.zeros((ns, ns)) 
        for i in range(ns):
            for j in range(ns):
                omega_diff[i,j] = (self.ham.evals[i] - self.ham.evals[j])
        
        # (2b) compute bath integral for all possible lambda's and omega_diff's 
        lamdalist = [-2.0, -1.0, 0.0, 1.0, 2.0]
        bath_integrals = []
        for lam in lamdalist:
            matrix = np.zeros((ns, ns), dtype = np.complex128)
            if lam == 0:
                bath_integrals.append(matrix)
            else:
                for i in range(ns):
                    for j in range(ns):
                        omega_ij = omega_diff[i,j]
                        matrix[i,j] = self.ham.spec.correlationFT(omega_ij, lam, self.kappa)  
                bath_integrals.append(matrix)
        
        
        # (3) compute only gammas we need for Eq. (19) according to Eq. (14)
        # (3a) rates of type gamma_{ji, ij} in Eq. (19)
        # (3b) rates of type gamma_{ij, ji} in Eq. (19) 
        gammas_ij = np.zeros(ns, dtype = np.complex128) 
        gammas_ji = np.zeros(ns, dtype = np.complex128)
        for j in range(ns):
            for k in range(ns):
                for l in range(ns):
                    for m in range(ns):
                        for n in range(ns):
                            # (3a)
                            gammas_ij[j] += self.ham.sysbath[k, l]*np.conj(self.ham.Umat[k][j])*self.ham.Umat[l][center_idx]*\
                                            self.ham.sysbath[m, n]*np.conj(self.ham.Umat[m][center_idx])*self.ham.Umat[n][j]*\
                                            bath_integrals[int(lamdas[k, l, m, n]+2)][j, center_idx]
                            # (3b)
                            gammas_ji[j] += self.ham.sysbath[k, l]*np.conj(self.ham.Umat[k][center_idx])*self.ham.Umat[l][j]*\
                                            self.ham.sysbath[m, n]*np.conj(self.ham.Umat[m][j])*self.ham.Umat[n][center_idx]*\
                                            bath_integrals[int(lamdas[k, l, m, n]+2)][j, center_idx]
        

        # (4) compute rate vector going out of polaron i to all other polaronic states j in polaron_idxs (Eq. 19)
        rates = 2*np.real(gammas_ij)
        rates[center_idx] += - np.sum(2*np.real(gammas_ji))
        

        # (5) only consider outgoing rates, i.e. delete rates at polaron index i
        rates = np.delete(rates, center_idx)
        rates = rates/const.hbar
        
        return rates
    
    def make_redfield_tensor(self, center_idx):
        """Make and store the Redfield tensor
            (Markovian)
        """
        ns = self.ham.nsite

        kappa=self.kappa # kappa for polaron transform
        
        # compute lambda tensor (Eq. (16))
        ident = np.identity(ns)
        ones=np.ones((ns, ns, ns, ns))
        lamdas= (  np.einsum('ac, abcd->abcd', ident, ones) + np.einsum('bd, abcd->abcd', ident, ones)
                 - np.einsum('ad, abcd->abcd', ident, ones) - np.einsum('bc, abcd->abcd', ident, ones)
                  )
        
        # compute gamma_plus and gamma_minus
        gamma_plus = np.zeros((ns,ns,ns,ns), dtype = np.complex128)
        
        
        # compute integral of bath correlation function
        start = time.time()
        lamdalist = [-2.0, -1.0, 0.0, 1.0, 2.0]
        # bath_integrals = []
        bath_integrals_new = []
        for lam in lamdalist:
            # matrix = np.zeros((ns, ns), dtype = np.complex128)
            matrix_new = np.zeros(ns, dtype = np.complex128)
            if lam == 0:
                # bath_integrals.append(matrix)
                bath_integrals_new.append(matrix_new)
            else:
                # for i in range(ns):
                #     for j in range(ns):
                #         omega_ij = self.ham.omega_diff[i,j]
                #         matrix[i,j] = self.ham.spec.correlationFT(omega_ij, lam, kappa)  
                # bath_integrals.append(matrix)
                for i in range(ns):
                    omega_ij = self.ham.omega_diff[i,center_idx]
                    matrix_new[i] = self.ham.spec.correlationFT(omega_ij, lam, kappa)
                bath_integrals_new.append(matrix_new)

        end = time.time()
        print('time difference1', end - start)
        #print(bath_integrals[1].T[center_idx].shape, bath_integrals[2].T[center_idx].shape, bath_integrals[3].T[center_idx].shape)
        #print(bath_integrals_new[1].shape, bath_integrals_new[2].shape, bath_integrals_new[3].shape)

        # transform sysbath operators to eigenbasis
        start = time.time()
        Gs = np.zeros((ns, ns), dtype=object)
        for a in range(ns):
            for b in range(ns):
                Gs[a][b] = self.ham.site2eig( self.ham.sysbath[a][b] )
        end = time.time()
        print('time difference2', end - start)
        
        #gamma_plus = np.zeros((ns, ns), dtype = np.complex128)
        start = time.time()
        gamma_plus_new = np.zeros(ns, dtype = np.complex128)
        for lamda in [-2, -1, 0, 1, 2]:
            indices = np.argwhere(lamdas == lamda)
            for abcd in indices:
                #gamma_plus += np.multiply(np.transpose(Gs[abcd[0]][abcd[1]]), 
                #              np.multiply(Gs[abcd[2]][abcd[3]], bath_integrals[lamda + 2]))
                gamma_plus_new += np.multiply(bath_integrals_new[lamda + 2], 
                                  np.multiply(Gs[abcd[2]][abcd[3]].T[center_idx], Gs[abcd[0]][abcd[1]][center_idx]))
        end = time.time()
        print('time difference3', end - start)
        # compute population transfer matrix (reduced Redfield tensor)
        # rates computation (all information kept)

        # onlyoutgoin rates are relevant
        #self.red_R_tensor = 2 * np.real(gamma_plus.T)
        self.red_R_tensor1 = 2 * np.real(gamma_plus_new)
        print(self.red_R_tensor1)
        # print(self.red_R_tensor[center_idx])
        # print(2 * np.real(gamma_plus_new))
        # np.fill_diagonal(self.red_R_tensor, np.zeros(ns))
        # np.fill_diagonal(self.red_R_tensor, -np.sum(self.red_R_tensor, axis = 0))
        # rates = self.red_R_tensor[center_idx]
        # rates computation (onle center_idx)
        # self.red_R_tensor1 = 2 * np.real(gamma_plus_new)
        # self.red_R_tensor1[center_idx] = 0.0
        # self.red_R_tensor1[center_idx] = - np.sum(self.red_R_tensor1)
        #np.fill_diagonal(self.red_R_tensor1, np.zeros(ns))
        #np.fill_diagonal(self.red_R_tensor1, -np.sum(self.red_R_tensor1, axis = 0))

        # print differences
        # print(self.red_R_tensor[center_idx])
        # print(self.red_R_tensor1)

        # # get only rates at center_idx
        # test = self.red_R_tensor/const.hbar
        # test = test[~np.eye(test.shape[0],dtype=bool)].reshape(test.shape[0],-1)
        # test = np.sum(test.T, axis = 1)
        # print('test rates sum')
        # print(test)

        # get only outgoing rates at center idx
        #rates = self.red_R_tensor[center_idx]
        #rates = np.delete(rates, center_idx)
        #rates = rates/const.hbar

        rates1 = np.delete(self.red_R_tensor1, center_idx) / const.hbar

        return rates1
    
    
    



