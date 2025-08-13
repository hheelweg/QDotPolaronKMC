import numpy as np
from scipy import integrate
from scipy.linalg import eigh
from . import const
from . import utils
from . import hamiltonian_box, redfield_box
import time
import math



class KMCRunner():
    
    def __init__(self, dims, sidelength, qd_spacing, nrg_center, inhomog_sd, dipolegen, seed, relative_spatial_disorder, \
                 J_c, spectrum, temp, ntrajs, nrealizations, r_hop, r_ove):
        
        # geometric attributes
        self.dims = dims
        self.sidelength = sidelength
        self.n = sidelength ** dims
        self.qd_spacing = qd_spacing
        self.boundary = sidelength * qd_spacing
        self.lattice_dimension = np.array([sidelength] * dims) * self.qd_spacing            # dimensions of lattice
        # print('lattice dim', self.lattice_dimension)

        # energetic attributes
        self.nrg_center = nrg_center
        self.inhomog_sd = inhomog_sd
        # parameters for randomness of Hamiltonian
        self.dipolegen = dipolegen
        self.seed = seed
        self.relative_spatial_disorder = relative_spatial_disorder
        
        # bath parameters
        self.J_c = J_c
        self.spectrum = spectrum
        
        # number of trajectories per realization
        self.ntrajs = ntrajs 
        self.nrealizations = nrealizations

        # get box information based on r_hop and r_ove (in units of the lattice spacing)
        self.make_box_dimensions(r_hop, r_ove)
        # print('box side length', self.box_length)
        
        # make QD lattice
        self.make_qd_array()
        # print('qd_spacing', self.qd_spacing)

        
        # get temperature and Hamiltonian
        self.temp = temp
        self.set_temp(temp)
        
        # new way of defining the grid
        # NOTE : do we need this?
        if self.dims == 1:
            self.grid = self.qd_locations.reshape((self.sidelength, self.dims))/self.qd_spacing
        if self.dims == 2:
            self.grid = self.qd_locations.reshape((self.sidelength, self.sidelength, self.dims))/self.qd_spacing
    

    # HH : here is the definition of the box_radius based on the minimum of 
    # r_hop, r_ove rounded to the next higher integer (this is arbitary and
    # we might want to modify this moving forward)
    def make_box_dimensions(self, r_hop, r_ove):
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
    

    
    # NOTE : maybe rename this into setup_lattice because we do more than setting temperature
    def set_temp(self, temp):
        self.temp = temp
        self.beta = 1/(const.kB * self.temp) # 1/eV
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
  
    # NOTE : former get_disp_vector_matrix
    def _pairwise_displacements(self, qd_pos, boundary):
        """
        Match get_disp_vector_matrix(): wrapped displacement for magnitude.
        qd_pos: (n, d) with d in {1,2}
        boundary: scalar box length
        Returns: rij_wrap (n, n, 3) with wrap applied on first d coords
        """
        import numpy as np
        n, d = qd_pos.shape
        L = float(boundary)

        # unwrapped per-axis differences (j - i), shape (n,n,d)
        rij_d = qd_pos[None, :, :] - qd_pos[:, None, :]

        # exact same wrap rule as original code (> L/2 and < -L/2)
        too_high = rij_d >  (L / 2.0)
        too_low  = rij_d < -(L / 2.0)
        rij_d = rij_d.copy()
        rij_d[too_high] -= L
        rij_d[too_low]  += L

        # embed into 3D (dipoles are 3D)
        rij_wrap = np.zeros((n, n, 3), dtype=np.float64)
        rij_wrap[:, :, :d] = rij_d
        return rij_wrap


    # function to build couplings 
    def _build_J(self, qd_pos, qd_dip, J_c, kappa_polaron, boundary=None):
        """
        Vectorized but physics-identical to the original loops:
        J_ij = J_c * kappa_polaron * [ μ_i·μ_j - 3(μ_i·r̂_unwrapped)(μ_j·r̂_unwrapped) ] / (‖r_wrap‖^3),
        with pairwise normalization of μ_i, μ_j, and r̂_unwrapped (as in get_kappa).
        """
        import numpy as np

        n, d = qd_pos.shape
        assert d in (1, 2)

        # --- Magnitude uses WRAPPED displacement (minimum image), exactly like get_disp_vector_matrix
        if boundary is not None:
            rij_wrap = self._pairwise_displacements(qd_pos, boundary)  # (n,n,3)
        else:
            rij_wrap = np.zeros((n, n, 3), dtype=np.float64)
            rij_wrap[:, :, :d] = qd_pos[None, :, :] - qd_pos[:, None, :]

        r2 = np.einsum('ijk,ijk->ij', rij_wrap, rij_wrap)  # (n,n)
        np.fill_diagonal(r2, np.inf)                        # avoid div by zero
        r = np.sqrt(r2)

        # --- direction uses UNWRAPPED displacement (exactly what get_kappa did)
        rij_unwrap = np.zeros((n, n, 3), dtype=np.float64)
        rij_unwrap[:, :, :d] = qd_pos[None, :, :] - qd_pos[:, None, :]
        r2_dir = np.einsum('ijk,ijk->ij', rij_unwrap, rij_unwrap)
        np.fill_diagonal(r2_dir, 1.0)                       # any nonzero to prevent NaN on diagonal
        rhat_dir = rij_unwrap / np.sqrt(r2_dir)[:, :, None] # unit vector from UNWRAPPED coords

        # --- pairwise dipole normalization (mirror get_kappa)
        mu = qd_dip.astype(np.float64, copy=False)
        mu_unit = mu / np.linalg.norm(mu, axis=1, keepdims=True)

        # angular factor κ_ij using r̂_unwrapped
        mui_dot_muj = mu_unit @ mu_unit.T                                   # (n,n)
        mui_dot_r   = np.einsum('id,ijd->ij', mu_unit, rhat_dir)            # (n,n)
        muj_dot_r   = np.einsum('jd,ijd->ij', mu_unit, rhat_dir)            # (n,n)
        kappa = mui_dot_muj - 3.0 * (mui_dot_r * muj_dot_r)

        # 1 / ‖r_wrap‖^3   (exactly matches 1/np.linalg.norm(disp_ij)**3 in your loop)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_r3 = 1.0 / (r2 * r)  # r^3 = r2 * r

        J = (J_c * kappa_polaron) * kappa * inv_r3
        np.fill_diagonal(J, 0.0)
        return J
    

    def get_hamil(self, periodic=True):

        # (1) set up polaron-transformed Hamiltonian 
        # (1.1) coupling terms in Hamiltonian
        J = self._build_J(
                        qd_pos=self.qd_locations,
                        qd_dip=self.qddipoles,
                        J_c=self.J_c,
                        kappa_polaron=self.kappa_polaron,
                        boundary=(self.boundary if periodic else None)
                        )
        # (1.2) site energies and total Hamiltonian
        self.hamil = np.diag(self.qdnrgs).astype(np.float64, copy=False)
        self.hamil += J


        # (2) keep original diagonalization routine
        # NOTE : can we improve this function somehow? 
        self.eignrgs, self.eigstates = utils.diagonalize(self.hamil)

        # (3) polaron positions 
        if periodic:
            locations_unit_circle = (self.qd_locations / self.boundary) * (2*np.pi)  # (n,d)
            unit_circle_ycoords = np.sin(locations_unit_circle)
            unit_circle_xcoords = np.cos(locations_unit_circle)
            psi2 = self.eigstates**2                                                 # (n,n)
            unit_circle_eig_xcoords = (unit_circle_xcoords.T @ psi2).T               # == your transpose/matmul pattern
            unit_circle_eig_ycoords = (unit_circle_ycoords.T @ psi2).T
            eigstate_positions = np.arctan2(unit_circle_eig_ycoords, unit_circle_eig_xcoords) * (self.boundary / (2*np.pi))
            eigstate_positions[eigstate_positions < 0] += self.boundary
            self.polaron_locs = eigstate_positions
        else:
            self.polaron_locs = (self.qd_locations.T @ (self.eigstates**2)).T

        # (4) off-diagonal J for Redfield (system-bath)
        J_off = self.hamil - np.diag(np.diag(self.hamil))
        self.J_dense = J_off.copy()

        # (5) set up Hamilonian instance, spectral density, etc. 
        self.full_ham = hamiltonian_box.Hamiltonian(
            self.eignrgs, self.eigstates,
            spec_density=self.spectrum, kT=const.kB*self.temp, J_dense=self.J_dense
            )
        
        # (6) set up Redfield instance
        self.redfield = redfield_box.Redfield(
            self.full_ham, self.polaron_locs, self.qd_locations, self.kappa_polaron, self.r_hop, self.r_ove,
            time_verbose=True
        )

    
    def make_kmatrix_box(self, center_global):

        # (1) use the global indices of polaron and site inside box
        pol_box  = self.pol_idxs_last
        site_box = self.site_idxs_last

        # (2) refine the polaron and site indices by additional constraints on r_hop and r_ove
        # NOTE : refine_by_radius function can maybe be moved into this module ? 
        pol_g, site_g = self.redfield.refine_by_radius(
                    pol_idxs_global = pol_box,
                    site_idxs_global = site_box,
                    center_global = center_global,                      # global index of the center polaron
                    periodic=True,                                      # or False to match array setup
                    grid_dims=self.lattice_dimension
                    )

        # 2) compute rates on those exact indices (no re-derivation)
        self.rates, self.final_states, tot_time = self.redfield.make_redfield_box(
            pol_idxs_global=pol_g, site_idxs_global=site_g, center_global=center_global
        )

        # 3) cache by global center index
        overall_idx_start = center_global
        self.stored_npolarons_box[overall_idx_start] = len(pol_g)
        self.stored_polaron_sites[overall_idx_start] = np.copy(self.final_states)   # global indices
        self.stored_rate_vectors[overall_idx_start]  = np.copy(self.rates)

        return tot_time


    # make box around center position where we are currently at
    # TODO : incorporate periodic boundary conditions explicty (boolean)
    def get_box(self, center, periodic=True):

        # (1) box size (unchanged)
        self.box_size = self.box_length * self.qd_spacing

        # (2) helpers (unchanged logic)
        def find_indices_within_box(points, center, grid_dims, box_size, periodic=True):
            half_box = box_size / 2
            dim = len(center)
            assert len(points[0]) == len(center) == len(grid_dims)

            if dim == 1:
                dx = np.abs(points - center[0])
                if periodic:
                    dx = np.minimum(dx, grid_dims[0] - dx)
                return np.where(dx <= half_box)[0]

            elif dim == 2:
                dx = np.abs(points[:, 0] - center[0])
                dy = np.abs(points[:, 1] - center[1])
                if periodic:
                    dx = np.minimum(dx, grid_dims[0] - dx)
                    dy = np.minimum(dy, grid_dims[1] - dy)
                return np.where((dx <= half_box) & (dy <= half_box))[0]

            else:
                raise NotImplementedError("find_indices_within_box: only 1D/2D supported.")

        # (3) global index sets inside the axis-aligned periodic box
        pol_idxs = find_indices_within_box(self.polaron_locs, center, self.lattice_dimension, self.box_size, periodic)
        site_idxs = find_indices_within_box(self.qd_locations,  center, self.lattice_dimension, self.box_size, periodic)

        # keep order, store contiguously
        self.pol_idxs_last  = np.ascontiguousarray(pol_idxs.astype(np.intp))
        self.site_idxs_last = np.ascontiguousarray(site_idxs.astype(np.intp))

        # (4) define the GLOBAL center index once
        self.center_global = int(self.get_closest_idx(center, self.polaron_locs))

        # (5) optional: local position of the center inside the box (rarely needed now)
        where = np.nonzero(self.pol_idxs_last == self.center_global)[0]
        # If the box is tight or discrete, it should be present; if not, refine_by_radius will handle it.
        self.center_local = int(where[0]) if where.size == 1 else None



    def make_kmc_step(self, polaron_start_site):

        # (1) build box (just indices + center_global)
        self.get_box(polaron_start_site)

        center_global = self.center_global
        start_pol = self.polaron_locs[center_global]

        # (2) compute (or reuse) rates
        if self.stored_npolarons_box[center_global] == 0:
            tot_time = self.make_kmatrix_box(center_global)
        else:
            tot_time = 0.0
            self.final_states = self.stored_polaron_sites[center_global]  # global indices
            self.rates        = self.stored_rate_vectors[center_global]

        # (3) rejection-free KMC step
        cum_rates = np.cumsum(self.rates)
        S = cum_rates[-1]
        u = np.random.uniform()
        self.j = int(np.searchsorted(cum_rates, u * S))
        self.time += -np.log(np.random.uniform()) / S

        # (4) final polaron coordinate in GLOBAL frame
        end_pol = self.polaron_locs[self.final_states[self.j]]

        return start_pol, end_pol, tot_time

    
    
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

    
    
    def simulate_kmc(self, t_final, qd_array_refresh = 100):

        times_msds = np.linspace(0, t_final, int(t_final * 100))            # time ranges to use for computation of msds
                                                                            # NOTE : can adjust the coarseness of time grid (here: 1000)
        msds = np.zeros((self.nrealizations, len(times_msds)))              # mean squared displacements

        self.simulated_time = 0
        
        # loop over realization
        for r in range(self.nrealizations):

            self.make_qd_array()
            self.set_temp(self.temp)

            # loop over 
            for n in range(self.ntrajs):
                    
                self.time = 0                                       # reset clock for each trajectory
                self.step_counter = 0                               # keeping track of the # of KMC steps
                self.trajectory_start = np.zeros(self.dims)         # initialize trajectory start point
                self.trajectory_current = np.zeros(self.dims)       # initialize current trajectopry state
                self.sds = np.zeros(len(times_msds))                # store sq displacements on times_msd time grid
                
                comp_time = time.time()
                time_idx = 0
                sq_displacement = 0
                
                while self.time < t_final:

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
                    start_pol, end_pol, tot_time = self.make_kmc_step(start_pol)
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
                msds[r] = n/(n+1)*msds + 1/(n+1)*self.sds
                
                # return progress
                # print("{} KMC trajectories evolved, with {} KMC steps and an sds of {} before t_final is reached! Computed in {} s". format(n+1, self.step_counter, self.sds[-1], time.time()-comp_time))
                # if self.sds[-1] > 10000:
                #    print("uh oh {}".format(self.sds[-1]))
            
            print('----------------------------------')
            print('---- SIMULATED TIME SUMMARY -----')
            print(f'total simulated time {self.simulated_time:.3f}')
            print('----------------------------------')
        return times_msds, msds[0]

    
    
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
    




