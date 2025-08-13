import numpy as np
import math
from scipy import integrate

# class to set up QD Lattice 
class QDLattice():

    def __init__(self, dims, sidelength, qd_spacing,
                 nrg_center, inhomog_sd, dipolegen, relative_spatial_disorder,
                 seed,
                 r_hop, r_ove
                 ):

        # geometric attributes
        self.dims = dims
        self.sidelength = sidelength
        self.n = sidelength ** dims
        self.qd_spacing = qd_spacing
        self.boundary = sidelength * qd_spacing
        self.lattice_dimension = np.array([sidelength] * dims) * self.qd_spacing            # dimensions of lattice

        # energetic attributes
        self.nrg_center = nrg_center
        self.inhomog_sd = inhomog_sd
        self.relative_spatial_disorder = relative_spatial_disorder
        # parameters for randomness of Hamiltonian
        self.dipolegen = dipolegen
        self.seed = seed

        # initialize the box dimensions we consider for the KMC simulation
        self._init_box_dims(r_hop, r_ove)


    # NOTE: old make_qd_array method (unchanged)
    def _make_lattice(self):    
        # set locations for each QD
        self.qd_locations = np.zeros((self.n, self.dims))
        # if seed = None, then we draw a new random configuration every single time we call this method
        np.random.seed(self.seed)
        for i in np.arange(self.n):
            if self.dims == 2:
                self.qd_locations[i, :] = (i%self.sidelength * self.qd_spacing, np.floor(i/self.sidelength) * self.qd_spacing) \
                    + np.random.normal(0, self.qd_spacing * self.relative_spatial_disorder, [1, self.dims])
            elif self.dims == 1:
                self.qd_locations[i, :] = (i%self.sidelength * self.qd_spacing) \
                    + np.random.normal(0, self.qd_spacing * self.relative_spatial_disorder, [1, self.dims])
        self.qd_locations[self.qd_locations < 0] = self.qd_locations[self.qd_locations < 0] + self.sidelength * self.qd_spacing
        self.qd_locations[self.qd_locations > self.sidelength * self.qd_spacing] = \
            self.qd_locations[self.qd_locations > self.sidelength * self.qd_spacing] - self.sidelength * self.qd_spacing
        # set nrgs for each QD
        self.qdnrgs = np.random.normal(self.nrg_center, self.inhomog_sd, self.n)
        
        # set dipole moment orientation for each QD
        if self.dipolegen == 'random':
            self.qddipoles = np.random.normal(0, 1, [self.n, 3])
            self.qddipoles = self.qddipoles/np.array([[i] for i in np.linalg.norm(self.qddipoles, axis = 1)])
        elif self.dipolegen == 'alignZ':
            self.qddipoles = np.zeros([self.n, 3])
            self.qddipoles[:, 2] = np.ones(self.n)
        else:
            raise Exception("Invalid dipole generation type") 
        self.stored_npolarons_box = np.zeros(self.n)
        self.stored_polaron_sites = [np.array([]) for i in np.arange(self.n)]
        self.stored_rate_vectors = [np.array([]) for i in np.arange(self.n)]
    


    # NOTE : this uses box_radius = min(r_hop, r_ove) rounded to the next higher integer
    def _init_box_dims(self, r_hop, r_ove):
        # convert to actual units
        self.r_hop = r_hop * self.qd_spacing
        self.r_ove = r_ove * self.qd_spacing
        # box radius and dimensions:
        self.box_radius = math.ceil(min(r_hop, r_ove))
        # self.box_radius = r_box
        self.box_length = 2 * self.box_radius + 1
        # raise wanring if lattice dimensions are exceeded
        if self.box_length > self.sidelength:
            raise Warning('the lattice dimensions are exceeded! \
                          Please choose r_hop and r_ove accordingly!')
    

    

    
    def _setup(self, temp, ):
        self.get_kappa_polaron()
        self.get_hamil()


    # NOTE : currently only implemented for cubic-exp spectral density
    def get_kappa_polaron(self, freq_max = 1):
        lamda = self.spectrum[1]
        omega_c = self.spectrum[2]
        
        # TODO: update this to account for different spectrum funcs
        spectrum_func = lambda w: (np.pi*lamda/(2*omega_c**3))*w**3*np.exp(-w/omega_c)
        integrand = lambda freq : 1/np.pi * spectrum_func(freq)/np.power(freq, 2) * 1/np.tanh(self.beta * freq/2)
        self.kappa_polaron = np.exp(-integrate.quad(integrand, 0, freq_max)[0])


