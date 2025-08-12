import numpy as np
from scipy import integrate
from . import const
from . import utils
from . import hamiltonian_box, redfield_box
import time
import math



class KMCRunner():
    
    def __init__(self, dims, sidelength, qd_spacing, nrg_center, inhomog_sd, dipolegen, seed, relative_spatial_disorder, \
                 J_c, spectrum, temp, ntrajs, r_hop, r_ove, r_box):
        
        self.dims = dims
        self.sidelength = sidelength
        self.n = sidelength ** dims
        self.qd_spacing = qd_spacing
        self.boundary = sidelength * qd_spacing
        self.nrg_center = nrg_center
        self.inhomog_sd = inhomog_sd
        
        # parameters for randomness of Hamiltonian
        self.dipolegen = dipolegen
        self.seed = seed
        self.relative_spatial_disorder = relative_spatial_disorder
        
        # bath parameters
        self.J_c = J_c
        self.spectrum = spectrum
        
        # number of trajectories
        self.ntrajs = ntrajs 

        # get box information based on r_hop and r_ove (in units of the lattice spacing)
        self.make_box_dimensions(r_hop, r_ove, r_box)
        # print('box side length', self.box_length)
        
        # make QD lattice
        self.make_qd_array()
        # print('qd_spacing', self.qd_spacing)

        # dimensions of the lattice
        self.lattice_dimension = np.array([sidelength] * dims) * self.qd_spacing
        # print('lattice dim', self.lattice_dimension)
        
        # get temperature and Hamiltonian
        self.temp = temp
        self.set_temp(temp)
        
        # new way of defining the grid
        if self.dims == 1:
            self.grid = self.qd_locations.reshape((self.sidelength, self.dims))/self.qd_spacing
        if self.dims == 2:
            self.grid = self.qd_locations.reshape((self.sidelength, self.sidelength, self.dims))/self.qd_spacing
    
    # HH : here is the definition of the box_radius based on the minimum of 
    # r_hop, r_ove rounded to the next higher integer (this is arbitary and
    # we might want to modify this moving forward)
    def make_box_dimensions(self, r_hop, r_ove, r_box = 3):
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

    def get_relative_grid(self, center):
        """
        make relative grid around center coordinates while accounting for periodic boundary conditions.
        The recentered grid has (0,0) at coordinates center
        """
        # (1) recenter the grid
        recentered_grid = self.qd_locations - center
        # (2) account for periodic boundary conditions
        for i, coords in enumerate(recentered_grid):
            for j in range(self.dims):
                if coords[j] > self.box_radius*self.qd_spacing > 0: coords[j] = int(coords[j] - self.boundary)
                elif coords[j] < -self.box_radius*self.qd_spacing < 0: coords[j] = int(coords[j] + self.boundary)
        return recentered_grid

    def get_box(self, center):
        
        """
        create box around center coordinate and get box Hamiltonian as well as the eigenstates, 
        eigenenergies and locations of the polarons in that box
        """

        # create relative grid
        relative_grid = self.get_relative_grid(center)
        grid = [list(x) for x in relative_grid] # transform relative grid to list for index search

        box_idxs = []
        # TODO : add dims == 3.
        if self.dims == 1:
            for i in range(self.box_length):
                box_coord = list(np.array([self.qd_spacing])*(i-self.box_radius))
                box_idxs.append(grid.index(box_coord))
        if self.dims == 2:
            for i in range(self.box_length):
                for j in range(self.box_length):
                    box_coord = list(np.array([self.qd_spacing, 0])*(i-self.box_radius)+np.array([0, self.qd_spacing])*(j-self.box_radius))
                    box_idxs.append(grid.index(box_coord))
        
        # for the array slicing to work, the indices in box_indxs need to be sorted
        box_idxs = sorted(box_idxs)
        
        # extract non-spatial information for the box
        self.n_box = self.box_length**self.dims
        self.hamil_box = self.hamil[box_idxs, :][:, box_idxs]
        # compute eigenenergies and polaron eigenstates in box
        [self.eignrgs_box, self.eigstates_box] = utils.diagonalize(self.hamil_box) 
        # retrieve spatial information
        self.sites_locs = self.qd_locations[box_idxs]
        self.sites_locs_rel = relative_grid[box_idxs]
        # compute positions of polaron eigenstates
        self.eigstates_locs = np.matmul(self.eigstates_box ** 2, self.sites_locs_rel) + center
        
    # QD array setup
    # TODO make this compatible with non-periodic boundary conditions
    def make_qd_array(self):    
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
        return
    
    def get_disp_vector_matrix(self, positions):
        positions_xcopy = np.array([np.array([np.ones(self.n) * positions[:, i] for j in np.arange(self.n)]) for i in np.arange(self.dims)])
        positions_ycopy = np.array([np.transpose(positions_xcopy[i, :, :]) for i in np.arange(self.dims)])
        displacement_vector_matrix = positions_ycopy - positions_xcopy
        too_high_indices = displacement_vector_matrix > self.boundary/2
        too_low_indices = displacement_vector_matrix < -self.boundary/2
        displacement_vector_matrix[too_high_indices] = displacement_vector_matrix[too_high_indices] - self.boundary
        displacement_vector_matrix[too_low_indices] = displacement_vector_matrix[too_low_indices] + self.boundary
        return displacement_vector_matrix
    

    def get_kappa(self, mu_d, mu_a, loc_d, loc_a):
        dist = np.zeros(3)
        # convert distance to 3D
        dist[0:np.size(loc_d)] = loc_a - loc_d
        dist /= np.linalg.norm(dist)
        mu_d_norm = mu_d/np.linalg.norm(mu_d)
        mu_a_norm = mu_a/np.linalg.norm(mu_a)
        return (np.dot(mu_d_norm, mu_a_norm) - 3 * np.dot(mu_d_norm, dist) * np.dot(mu_a_norm, dist))
    

    def set_temp(self, temp):
        self.temp = temp
        self.beta = 1/(const.kB * self.temp) # 1/eV
        self.get_kappa_polaron()
        self.get_hamil()


    def get_kappa_polaron(self, freq_max = 1):
        lamda = self.spectrum[1]
        omega_c = self.spectrum[2]
        
        # TODO: update this to account for different spectrum funcs
        spectrum_func = lambda w: (np.pi*lamda/(2*omega_c**3))*w**3*np.exp(-w/omega_c)
        integrand = lambda freq : 1/np.pi * spectrum_func(freq)/np.power(freq, 2) * 1/np.tanh(self.beta * freq/2)
        self.kappa_polaron = np.exp(-integrate.quad(integrand, 0, freq_max)[0])
  

    # polaron-transformed Hamiltonian, eigenenergies, and polaron positions
    def get_hamil(self, periodic = True):

        self.hamil = np.diag(self.qdnrgs)
        displacement_vector_matrix = self.get_disp_vector_matrix(self.qd_locations)
        for i in np.arange(self.n):
            for j in np.arange(i+1, self.n):
                self.hamil[i, j] = self.J_c * self.kappa_polaron * self.get_kappa(self.qddipoles[i, :], self.qddipoles[j, :], self.qd_locations[i, :], self.qd_locations[j, :]) \
                    * 1/np.linalg.norm(displacement_vector_matrix[:, i, j])**3
                self.hamil[j, i] = self.hamil[i, j]
        [self.eignrgs, self.eigstates] = utils.diagonalize(self.hamil)
        # get the positions of the eigenstates
        if periodic:
            # use circular average for periodic boundary conditions to properly account for wraparound
            # convert all coordinates to vectors with tips on the unit circle 
            locations_unit_circle = self.qd_locations / self.boundary * 2 * np.pi
            unit_circle_ycoords = np.sin(locations_unit_circle)
            unit_circle_xcoords = np.cos(locations_unit_circle)
            # add corresponding vectors of each eigenstate
            unit_circle_eig_xcoords = np.transpose(np.matmul(np.transpose(unit_circle_xcoords), self.eigstates**2))
            unit_circle_eig_ycoords = np.transpose(np.matmul(np.transpose(unit_circle_ycoords), self.eigstates**2))
            # convert back to location coordinates for qds
            eigstate_positions = np.arctan2(unit_circle_eig_ycoords, unit_circle_eig_xcoords) * self.boundary /(2 * np.pi)
            eigstate_positions[eigstate_positions < 0] = eigstate_positions[eigstate_positions < 0] + self.boundary
            self.polaron_locs = eigstate_positions
        else:
            self.polaron_locs = np.transpose(np.matmul(np.transpose(self.qd_locations), self.eigstates**2))
        
        # does ham_sysbath do anything
        J = self.hamil - np.diag(np.diag(self.hamil)) 
        ham_sysbath = []
        for i in range(self.n):
            ham_list=[]
            for j in range(self.n):
                ham_coupl=np.zeros((self.n, self.n))
                ham_coupl[i,j]=J[i,j]
                ham_list.append(ham_coupl)
            ham_sysbath.append(ham_list)
        
        # Build ONE global Hamiltonian and ONE Redfield
        self.full_ham = hamiltonian_box.Hamiltonian(self.eignrgs, self.eigstates, self.qd_locations,
                                             ham_sysbath, self.spectrum, const.kB * self.temp)
        # do we need this?
        self.spectrum_calc = self.full_ham.spec
        self.redfield = redfield_box.NewRedfield(
                                                 self.full_ham, self.polaron_locs, self.kappa_polaron, self.r_hop, self.r_ove,
                                                 time_verbose=True
                                                )


    def make_kmatrix_box(self, center):
        
        """
        make rates and return indices of final polaron states, as well as index i
        of the start polaron
        """
        # system-bath Hamiltonian
        J = self.hamil_box - np.diag(np.diag(self.hamil_box)) 
        dim = len(self.hamil_box)
        ham_sysbath = []
        for i in range(dim):
            ham_list=[]
            for j in range(dim):
                ham_coupl=np.zeros((dim,dim))
                ham_coupl[i,j]=J[i,j]
                ham_list.append(ham_coupl)
            ham_sysbath.append(ham_list)   
        
        my_ham = hamiltonian_box.Hamiltonian(self.eignrgs_box, self.eigstates_box, self.sites_locs_rel,
                                             ham_sysbath, self.spectrum, const.kB * self.temp)
        my_redfield = redfield_box.Redfield(my_ham, self.kappa_polaron, self.r_hop, self.r_ove)

        # get rates and indices of the potential final polaron states we can jump to
        self.rates, self.final_states = my_redfield.make_redfield_box(center)

    # need to add this function! 
    def NEW_kmatrix_box(self, center):
        # TODO: consider checking that enough fraction of an eigenstate is included in the box before summing
        """
        make rates and return indices of final polaron states, as well as index i
        of the start polaron
        """
        # system-bath Hamiltonian
        J = self.hamil_box - np.diag(np.diag(self.hamil_box)) 
        dim = len(self.hamil_box)
        ham_sysbath = []
        for i in range(dim):
            ham_list=[]
            for j in range(dim):
                ham_coupl=np.zeros((dim,dim))
                ham_coupl[i,j]=J[i,j]
                ham_list.append(ham_coupl)
            ham_sysbath.append(ham_list)   
        
        my_ham = hamiltonian_box.Hamiltonian(self.eignrgs_box, self.eigstates_box, self.sites_locs_rel,
                                             ham_sysbath, self.spectrum_calc, const.kB * self.temp)
        my_redfield = redfield_box.NewRedfield(my_ham, self.eigstates_locs, self.kappa_polaron, self.r_hop, self.r_ove)

        # get rates and indices of the potential final polaron states we can jump to
        self.rates, self.final_states, tot_time = my_redfield.make_redfield_box(center, self.site_idxs_last, self.pol_idxs_last)
        overall_idx_start = self.get_closest_idx(self.eigstates_locs_abs[center], self.polaron_locs)
        self.stored_npolarons_box[overall_idx_start] = len(self.hamil_box)
        self.stored_polaron_sites[overall_idx_start] = np.copy(self.final_states)
        self.stored_rate_vectors[overall_idx_start] = np.copy(self.rates)
        
        return tot_time


    # def NEW_kmatrix_box(self, center_local):
    #     # 1) Use the indices prepared by NEW_get_box (periodic/relative)
    #     pol_idxs  = self.pol_idxs_last          # 1D global indices, from NEW_get_box
    #     site_idxs = self.site_idxs_last         # 1D global indices, from NEW_get_box
    #     center_global = self.center_global        # global index inside pol_idxs


    #     pol_g, site_g = self.redfield.refine_by_radius(
    #                 pol_idxs_global=self.pol_idxs_last,
    #                 site_idxs_global=self.site_idxs_last,
    #                 center_global=self.center_global,      # global index of the center polaron
    #                 periodic=True,                         # or False to match your physics
    #                 grid_dims=[self.sidelength] * int(self.dims)       # needed if periodic=True
    #                 )

    #     # 2) Compute rates on those exact indices (no re-derivation)
    #     self.rates, self.final_states, tot_time = self.redfield.make_redfield_box_global(
    #         pol_idxs_global=pol_g, site_idxs_global=site_g, center_global=center_global
    #     )

    #     # # Baseline-style:
    #     # cB = center_local
    #     print('polB, siteB, cB', pol_g, site_g, center_global)

    #     # 3) Cache by global center index
    #     overall_idx_start = center_global
    #     self.stored_npolarons_box[overall_idx_start] = len(pol_idxs)
    #     self.stored_polaron_sites[overall_idx_start] = np.copy(self.final_states)   # global indices
    #     self.stored_rate_vectors[overall_idx_start]  = np.copy(self.rates)

    #     return tot_time


    # make box around center position where we are currently at
    # TODO : incorporate periodic boundary conditions explicty (boolean)
    def NEW_get_box(self, center, periodic=True):
        # box dimensions (same as original)
        self.box_size = self.box_length * self.qd_spacing

        # ---- helpers (same logic as your original) ----
        def find_indices_within_box(points, center, grid_dimensions, box_size, periodic=True):
            half_box = box_size / 2
            dim = len(center)
            assert (len(points[0]) == len(center) == len(grid_dimensions))

            if dim == 1:
                dx = np.abs(points - center[0])
                if periodic:
                    dx = np.minimum(dx, grid_dimensions[0] - dx)
                mask = (dx <= half_box)
                return np.where(mask)[0]

            elif dim == 2:
                dx = np.abs(points[:, 0] - center[0])
                dy = np.abs(points[:, 1] - center[1])
                if periodic:
                    dx = np.minimum(dx, grid_dimensions[0] - dx)
                    dy = np.minimum(dy, grid_dimensions[1] - dy)
                mask = (dx <= half_box) & (dy <= half_box)
                return np.where(mask)[0]

            else:
                raise NotImplementedError("find_indices_within_box: only 1D/2D supported.")

        def get_relative_positions(points, center, grid_dimensions):
            dim = len(center)
            assert (len(points[0]) == len(center) == len(grid_dimensions))

            if dim == 1:
                dx = points - center[0]
                if periodic:
                    dx = (dx + grid_dimensions[0] / 2) % grid_dimensions[0] - grid_dimensions[0] / 2
                return dx

            elif dim == 2:
                dx = points[:, 0] - center[0]
                dy = points[:, 1] - center[1]
                if periodic:
                    dx = (dx + grid_dimensions[0] / 2) % grid_dimensions[0] - grid_dimensions[0] / 2
                    dy = (dy + grid_dimensions[1] / 2) % grid_dimensions[1] - grid_dimensions[1] / 2
                return np.column_stack((dx, dy))

            else:
                raise NotImplementedError("get_relative_positions: only 1D/2D supported.")

        # ---- (1) indices by BOX MASK ONLY (no spherical refinement) ----
        pol_idxs  = find_indices_within_box(self.polaron_locs, center, self.lattice_dimension, self.box_size, periodic)
        site_idxs = find_indices_within_box(self.qd_locations, center, self.lattice_dimension, self.box_size, periodic)

        # Keep order exactly as produced (no np.unique), but store contiguous for later
        self.pol_idxs_last  = np.ascontiguousarray(pol_idxs.astype(np.intp))
        self.site_idxs_last = np.ascontiguousarray(site_idxs.astype(np.intp))

        # ---- (2) absolute positions (unchanged) ----
        self.eigstates_locs_abs = self.polaron_locs[self.pol_idxs_last]
        self.site_locs          = self.qd_locations[self.site_idxs_last]

        # ---- (3) relative positions with periodic wrap (unchanged) ----
        self.eigstates_locs = get_relative_positions(self.eigstates_locs_abs, center, self.lattice_dimension)
        self.sites_locs_rel = get_relative_positions(self.site_locs,         center, self.lattice_dimension)

        # ---- (4) eigen-objects in the box (unchanged) ----
        self.eignrgs_box   = self.eignrgs[self.pol_idxs_last]
        self.eigstates_box = self.eigstates[self.site_idxs_last, :][:, self.pol_idxs_last]

        # ---- (5) local box Hamiltonian (unchanged) ----
        self.hamil_box = self.hamil[np.ix_(self.site_idxs_last, self.site_idxs_last)]


        center_global_idx = self.get_closest_idx(center, self.polaron_locs)  # search in FULL global coords
        self.center_global = int(center_global_idx)

        # you can still store the local index if you want to keep it for legacy code
        self.center_local = np.where(self.pol_idxs_last == self.center_global)[0][0]





    
    # def NEW_get_box(self, center, periodic = True):

    #     # box dimensions
    #     self.box_size = self.box_length * self.qd_spacing

    #     # (1)function that finds indices of points array within box_size of center
    #     def find_indices_within_box(points, center, grid_dimensions, box_size, periodic = True):
    #         # half box size
    #         half_box = box_size /2
    #         # get dimensiona of lattice and make sure all dimensions are equal
    #         dim = len(center)
    #         assert(len(points[0]) == len(center) == len(grid_dimensions))

    #         # TODO : implement for 3D
    #         if dim == 1:
    #             # Compute periodic distances in x direction
    #             dx = np.abs(points - center[0])

    #             # Apply periodic boundary conditions
    #             dx = np.minimum(dx, grid_dimensions[0] - dx)  # Distance considering periodic wrapping

    #             # Find indices where points are within the box
    #             mask = (dx <= half_box)
    #             return np.where(mask)[0]

    #         elif dim == 2:
    #             # Compute periodic distances in x and y directions
    #             dx = np.abs(points[:, 0] - center[0])
    #             dy = np.abs(points[:, 1] - center[1])

    #             # Apply periodic boundary conditions
    #             dx = np.minimum(dx, grid_dimensions[0] - dx)  # Distance considering periodic wrapping
    #             dy = np.minimum(dy, grid_dimensions[1] - dy)

    #             # Find indices where points are within the square box
    #             mask = (dx <= half_box) & (dy <= half_box)
    #             return np.where(mask)[0]

    #     # (2) function that finds relative position of points array w.r.t. centerx
    #     def get_relative_positions(points, center, grid_dimensions):

    #         # get dimensiona of lattice and make sure all dimensions are equal
    #         dim = len(center)
    #         assert(len(points[0]) == len(center) == len(grid_dimensions))
    #         if dim == 1:
    #             # Compute raw relative distances
    #             dx = points - center[0]

    #             # Apply periodic boundary conditions: adjust distances for wrap-around
    #             dx = (dx + grid_dimensions[0] / 2) % grid_dimensions[0] - grid_dimensions[0] / 2  # Wrap around midpoint
    #             return dx
    #         elif dim == 2:
    #             # Compute raw relative distances
    #             dx = points[:, 0] - center[0]
    #             dy = points[:, 1] - center[1]

    #             # Apply periodic boundary conditions: adjust distances for wrap-around
    #             dx = (dx + grid_dimensions[0] / 2) % grid_dimensions[0] - grid_dimensions[0] / 2  # Wrap around midpoint
    #             dy = (dy + grid_dimensions[1] / 2) % grid_dimensions[1] - grid_dimensions[1] / 2
    #             return np.column_stack((dx, dy)) 

    #     # get indices of polarons that are inside the box
    #     pol_idxs = find_indices_within_box(self.polaron_locs, center, self.lattice_dimension, self.box_size)
    #     # get indices of sites that are within the box
    #     site_idxs = find_indices_within_box(self.qd_locations, center, self.lattice_dimension, self.box_size)

    #     # get absolute positions of polarons and sites in box
    #     self.eigstates_locs_abs = self.polaron_locs[pol_idxs]
    #     self.site_locs = self.qd_locations[site_idxs]

    #     # get relative positions of polarons and sites in box
    #     self.eigstates_locs = get_relative_positions(self.eigstates_locs_abs, center, self.lattice_dimension)
    #     self.sites_locs_rel = get_relative_positions(self.site_locs, center, self.lattice_dimension)

    #     # get eigenstate energies and eigenstates in box
    #     self.eignrgs_box = self.eignrgs[pol_idxs]
    #     self.eigstates_box = self.eigstates[site_idxs, :][:, pol_idxs]

    #     # get box Hamiltonian
    #     # TODO : how can we use the new Hamiltonian for Redfield?
    #     self.hamil_box = self.hamil[site_idxs, :][:, site_idxs]


    def NEW_make_kmc_step(self, polaron_start_site):
        
        # (1) create box around polaron start_site
        self.NEW_get_box(polaron_start_site)

        # polA, siteA, cA = np.copy(self.pol_idxs_last), np.copy(self.site_idxs_last), int(self.center_local)
        # print('polA, siteA, cA', polA, siteA, cA)
        
        # (2) get idx of polaron eigenstate in box
        overall_idx_start = self.get_closest_idx(polaron_start_site, self.polaron_locs)
        box_idx_start = self.get_closest_idx(polaron_start_site, self.eigstates_locs_abs)
        start_pol = self.eigstates_locs_abs[box_idx_start]
        
        # (3) get rates from this polaron (box center) to potential final states
        if self.stored_npolarons_box[overall_idx_start] == 0:
            tot_time = self.NEW_kmatrix_box(box_idx_start)
        else:
            tot_time = 0
            self.final_states = self.stored_polaron_sites[overall_idx_start]
            self.rates = self.stored_rate_vectors[overall_idx_start]
        
        # (4) rejection-free KMC step
        # (4a) get cumulative rates
        cum_rates = np.array([np.sum(self.rates[:i+1]) for i in range(len(self.rates))])
        S = cum_rates[-1]
        # (4b) draw random number u and determine j s.t. cumrates[j-1] < u*T < cum_rates[j]
        u = np.random.uniform()
        self.j = np.searchsorted(cum_rates, u * S)
        # (4b) update time clock
        self.time += - np.log(np.random.uniform()) / S

        # (5) obtain spatial coordinates of final polaron state j
        # original version
        end_pol = self.eigstates_locs_abs[self.final_states[self.j]]

        # modified version 
        #end_pol = self.polaron_locs[self.final_states[self.j]]
        
        return start_pol, end_pol, tot_time

    # need to continue here
    # def NEW_make_kmc_step(self, polaron_start_site):

    #     # find global center index as before (using absolute polaron positions)
    #     center_idx = int(self.get_closest_idx(polaron_start_site, self.polaron_locs))

    #     # use the ORIGINAL selection (lives in NewRedfield and matches old physics)
    #     pol_idxs, site_idxs = self.redfield.get_idxs(center_idx)

    #     # store for downstream use
    #     self.pol_idxs_last  = pol_idxs.astype(np.intp)
    #     self.site_idxs_last = site_idxs.astype(np.intp)

    #     # local center index inside pol_idxs
    #     where = np.nonzero(self.pol_idxs_last == center_idx)[0]
    #     if where.size != 1:
    #         raise RuntimeError("Center not uniquely in pol_idxs")
    #     self.center_local = int(where[0])

    #     # absolute positions for output (unchanged)
    #     self.eigstates_locs_abs = self.polaron_locs[self.pol_idxs_last]
    #     start_pol = self.eigstates_locs_abs[self.center_local]

    #     # compute/use rates
    #     overall_idx_start = int(self.pol_idxs_last[self.center_local])
    #     if self.stored_npolarons_box[overall_idx_start] == 0:
    #         tot_time = self.NEW_kmatrix_box(self.center_local)  # uses stored indices
    #     else:
    #         tot_time = 0.0
    #         self.final_states = self.stored_polaron_sites[overall_idx_start]
    #         self.rates        = self.stored_rate_vectors[overall_idx_start]

    #     # KMC step
    #     cum_rates = np.cumsum(self.rates)
    #     S = cum_rates[-1]
    #     if S <= 0.0:
    #         self.j = 0
    #         end_pol = start_pol
    #         return start_pol, end_pol, tot_time

    #     u = np.random.uniform()
    #     self.j = int(np.searchsorted(cum_rates, u * S))
    #     self.time += -np.log(np.random.uniform()) / S

    #     end_pol = self.eigstates_locs_abs[self.final_states[self.j]]
    #     return start_pol, end_pol, tot_time
    
    
    
    def make_kmc_step(self, start_site):
        
        """
        perform a single KMC step from site coordinate (start_site) by locally diagonalizing
        the Hamiltonian in a box around that site.
        -----
        Returns:
            start_pol: coordinates of closest polaron site in box at start of step (closest to start_site)
            end_pol: coordinates of polaron after KMC step
        """
        
        np.random.seed(None)
        
        # (1) create box around start_site (but only if the charge has moved
        # in the previous step, i.e. if new_diagonalization = True)
        if self.new_diagonalization:
            self.get_box(start_site)
        
        # (2) get idx of closest polaron eigenstate in box 
        idx_start = self.get_closest_idx(start_site, self.eigstates_locs)
        start_pol = self.eigstates_locs[idx_start]
        self.start_pol = start_pol
        
        # (3) get rates from this polaron (box center) to potential final states
        self.make_kmatrix_box(idx_start)
        # (4) rejection-free KMC step
        # (4a) get cumulative rates
        cum_rates = np.array([np.sum(self.rates[:i+1]) for i in range(len(self.rates))])
        S = cum_rates[-1]
        # (4b) draw random number u and determine j s.t. cumrates[j-1] < u*T < cum_rates[j]
        u = np.random.uniform()
        self.j = np.searchsorted(cum_rates, u * S)
        # (4b) update time clock
        self.time += - np.log(np.random.uniform()) / S

        # (5) obtain spatial coordinates of final polaron state j
        end_pol = self.eigstates_locs[self.final_states[self.j]]
        
        return start_pol, end_pol
    
    # HH : for some arrays of r_hop and r_ove, comopute the rates and check
    # for convergence (you are more than invited to play around with this!)
    def get_rate_convergence(self, r_hops, r_oves):
        # determine convergence of rates
        rateConvergence = np.zeros((len(r_hops), len(r_oves)))
        boxLengths = np.zeros((len(r_hops), len(r_oves)))
        # more microscopoic rate convergence information
        rateMeanFinal = np.zeros((len(r_hops), len(r_oves), self.dims))
        rateThreeBiggest = np.zeros((len(r_hops), len(r_oves), 3))
        rateCumThree = np.zeros((len(r_hops), len(r_oves)))

        # consider outgoing rates from the most central site:
        start_site = np.array([int(self.sidelength // 2) * self.qd_spacing, int(self.sidelength // 2) * self.qd_spacing])

        # loop over all combinations of r_ove and r_hop 
        for i, r_hop in enumerate(r_hops):
            for j, r_ove in enumerate(r_oves):

                # (0) get box properties
                self.make_box_dimensions(r_hop, r_ove)
                boxLengths[i, j] = self.box_length

                # (1) get rates
                # (1.1) create box
                self.get_box(start_site)
                # (1.2) get initial polaron state (polaron basis)
                idx_start = self.get_closest_idx(start_site, self.eigstates_locs)
                self.start_pol = self.eigstates_locs[idx_start]
                # (1.3) obtain rates
                self.make_kmatrix_box(idx_start)

                # (2) analyze those rates based on desired criteria
                # (2.1) cumulative rates
                cum_rates = np.sum(self.rates)
                # (2.2) more microscopic details on the rates
                mean_final, three_biggest, cum_three_biggest_norm  = self.analyze_rates()

                # (3) store
                rateConvergence[i, j] = cum_rates
                rateMeanFinal[i, j, :] = mean_final
                #rateThreeBiggest[i, j, :] = three_biggest
                #rateCumThree[i, j] = cum_three_biggest_norm

        return rateConvergence, boxLengths, rateMeanFinal, rateThreeBiggest, rateCumThree
    
    # HH : this gives some more microscopic insights into self.rates
    def analyze_rates(self):
        # normalize rates array
        rates_norm = self.rates / np.sum(self.rates)
        # find three biggest entries and also the percentage of rates
        # (1) get mean final state difference from original state
        locs = self.eigstates_locs[self.final_states]
        mean_final = np.matmul(locs.T, rates_norm) - self.start_pol
        # (2) get three biggest (normalized) rates
        rates_norm.sort()
        three_biggest = rates_norm[-3:][::-1]
        cum_three_biggest_norm = np.sum(three_biggest)
        return mean_final, three_biggest, cum_three_biggest_norm   

        
    # HH : helper function that prints you the rates from start_site coordinate   
    def test_box_rates(self, start_site, rates_verbose = True):
        # create box
        self.get_box(start_site)
        # get initial polaron state (polaron basis)
        idx_start = self.get_closest_idx(start_site, self.eigstates_locs)
        start_pol = self.eigstates_locs[idx_start]
        self.start_pol = start_pol

        # obtain rates
        self.make_kmatrix_box(idx_start)

        # print rates
        if rates_verbose:
            print('rates ', self.rates)
            print('cumulative rates: ', np.sum(self.rates))
            print('length of rates vector:', len(self.rates))

        # # get location of final state with biggest rate for plotting purposes
        self.final_max = self.eigstates_locs[self.final_states[np.argmax(self.rates)]]
        

    def simulate_kmc(self, t_final):

        times_msds = np.linspace(0, t_final, int(t_final * 100))    # time ranges to use for computation of msds
                                                                    # note: can adjust the coarseness of time grid (here: 1000)
        msds = np.zeros(len(times_msds))                            # mean squared displacements
        
        for n in range(self.ntrajs):
            
            self.time = 0                                       # reset clock for each trajectory
            self.step_counter = 0                               # keeping track of the # of KMC steps
            self.trajectory_start = np.zeros(self.dims)         # initialize trajectory start point
            self.trajectory_current = np.zeros(self.dims)       # initialize current trajectopry state
            self.sds = np.zeros(len(times_msds))                # store sq displacements on times_msd time grid
            time_idx = 0

            # re-initialize Hamiltonian (i.e. different realizations of noise)
            self.make_qd_array()
            self.set_temp(self.temp)
            
            while self.time < t_final:

                # (1) determine what site we construct the box around
                if self.step_counter == 0:
                    # draw initial center of the box (here: 'uniform') in the exciton site basis
                    # TODO : might want to add other initializations
                    start_site = self.qd_locations[np.random.randint(0, self.n-1)]
                    #self.times.append(self.time)
                    self.new_diagonalization = True
                else:
                    # start_site is final_site from previous step
                    start_site = end_site
            
                # (2) perform KMC step and obtain coordinates of polaron at beginning (start_pol) and end (end_pol) of the step
                start_pol, end_pol = self.make_kmc_step(start_site)
                
                # (3) update trajectory and compute squared displacements 
                if self.step_counter == 0:
                    self.trajectory_start = start_pol
                    self.trajectory_current = start_pol
                if self.time < t_final:
                    # get current location in trajectory and compute squared displacement
                    self.trajectory_current = self.trajectory_current + end_pol-start_pol
                    sq_displacement = np.linalg.norm(self.trajectory_current-self.trajectory_start)**2 
                
                    # add squared displacement at correct position in times_msd grid
                    time_idx += np.searchsorted(times_msds[time_idx:], self.time)
                    self.sds[time_idx:] = sq_displacement
                        
                # (4) find lattice site closest to end_pol position and only diagonalize again if start_site != final_site 
                end_site = self.qd_locations[self.get_closest_idx(end_pol, self.qd_locations)]
                self.new_diagonalization = not (start_site == end_site).all()
                
                self.step_counter += 1 # update step counter
                
            # compute mean squared displacement as a running average instead of storing all displacement vectors
            msds = n/(n+1)*msds + 1/(n+1)*self.sds
            
            # return progress
            # print("{} KMC trajectories evolved, with {} KMC steps and an sds of {} before t_final is reached!". format(n+1, self.step_counter, self.sds[-1]))
            # if self.sds[-1] > 10000:
                # print("uh oh {}".format(self.sds[-1]))
        return times_msds, msds
    
    def NEW_simulate_kmc(self, t_final, qd_array_refresh = 100):

        times_msds = np.linspace(0, t_final, int(t_final * 100))    # time ranges to use for computation of msds
                                                                    # note: can adjust the coarseness of time grid (here: 1000)
        msds = np.zeros(len(times_msds))                            # mean squared displacements

        self.simulated_time = 0
        
        for n in range(self.ntrajs):
                
            self.time = 0                                       # reset clock for each trajectory
            self.step_counter = 0                               # keeping track of the # of KMC steps
            self.trajectory_start = np.zeros(self.dims)         # initialize trajectory start point
            self.trajectory_current = np.zeros(self.dims)       # initialize current trajectopry state
            self.sds = np.zeros(len(times_msds))                # store sq displacements on times_msd time grid
            
            comp_time = time.time()
            time_idx = 0
            sq_displacement = 0

            # re-initialize Hamiltonian (i.e. different realizations of noise)
            # NOTE (08/11/2025): do we need this?
            if n > 0 and n % qd_array_refresh == 0:
               self.make_qd_array()
               self.set_temp(self.temp)
            
            while self.time < t_final:

                # print(f'---------------TRAJ {n}----------------', flush = True)
                # print(f'time step: {self.step_counter}', flush = True)

                # (1) determine what polaron site we are at currently
                if self.step_counter == 0:
                    # draw initial center of the box (here: 'uniform') in the exciton site basis
                    # TODO : might want to add other initializations
                    start_site = self.qd_locations[np.random.randint(0, self.n-1)]
                    start_pol = self.polaron_locs[self.get_closest_idx(start_site, self.polaron_locs)]
                    #self.times.append(self.time)
                    self.new_diagonalization = True
                else:
                    # start_site is final_site from previous step
                    start_pol = end_pol
            
                # (2) perform KMC step and obtain coordinates of polaron at beginning (start_pol) and end (end_pol) of the step
                start_pol, end_pol, tot_time = self.NEW_make_kmc_step(start_pol)
                self.simulated_time += tot_time
                
                # (3) update trajectory and compute squared displacements 
                if self.step_counter == 0:
                    self.trajectory_start = start_pol
                    self.trajectory_current = start_pol
                if self.time < t_final:
                    # get current location in trajectory and compute squared displacement
                    self.trajectory_current = self.trajectory_current + end_pol - start_pol
                    sq_displacement = np.linalg.norm(self.trajectory_current-self.trajectory_start)**2 
                
                    # add squared displacement at correct position in times_msd grid
                    time_idx += np.searchsorted(times_msds[time_idx:], self.time)
                    self.sds[time_idx:] = sq_displacement
                # if sq_displacement > 10000:
                #     print("uh oh {}".format(self.sds[-1]))
                        
                # (4) find lattice site closest to end_pol position and only diagonalize again if start_site != final_site 
                end_site = self.qd_locations[self.get_closest_idx(end_pol, self.qd_locations)]
                self.new_diagonalization = not (start_pol == end_pol).all()
                
                self.step_counter += 1 # update step counter
                
            # compute mean squared displacement as a running average instead of storing all displacement vectors
            msds = n/(n+1)*msds + 1/(n+1)*self.sds
            
            # return progress
            # print("{} KMC trajectories evolved, with {} KMC steps and an sds of {} before t_final is reached! Computed in {} s". format(n+1, self.step_counter, self.sds[-1], time.time()-comp_time))
            # if self.sds[-1] > 10000:
            #    print("uh oh {}".format(self.sds[-1]))
        
        print('----------------------------------')
        print('---- SIMULATED TIME SUMMARY -----')
        print(f'total simulated time {self.simulated_time:.3f}')
        print('----------------------------------')
        return times_msds, msds

    
    
    # (08/09/2025) more efficient version
    def get_closest_idx(self, pos, array):
        """
        Find the index in `array` closest to `pos` under periodic boundary conditions.
        """
        # Vectorized periodic displacement
        delta = array - pos  # shape (N, dims)

        # Apply periodic boundary condition (minimum image convention)
        delta -= np.round(delta / self.boundary) * self.boundary

        # Compute squared distances
        dists_squared = np.sum(delta**2, axis=1)

        return np.argmin(dists_squared)
    

    def get_diffusivity_hh(self, msds, times, dims):
        # note : I here assume that the whole time arrange is approx. linear (might break down)
        fit_params, cov = np.polyfit(times, msds, 1, cov=True)
        diff = fit_params[0]/(2*dims)
        # obtain error on diffusvity as from error on slope parameter 
        diff_err = np.sqrt(np.diag(cov))[0]/(2*dims)
        return diff, diff_err
    
    def get_ipr(self):
        # returns ipr of one column vector, or mean ipr of multiple column vectors
        return np.mean(1/np.sum(self.eigstates ** 4, axis = 0))
    




