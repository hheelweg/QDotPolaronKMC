import numpy as np
from scipy import integrate
from . import const
from . import utils
from . import hamiltonian_box, redfield_box
import time



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
        
        # make QD lattice
        self.make_qd_array()
        
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
        # self.box_radius = math.ceil(min(r_hop, r_ove))
        self.box_radius = r_box
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
        # TODO : add dims == 1 and dims == 3.
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
  

    def get_hamil(self):

        self.hamil = np.diag(self.qdnrgs)
        displacement_vector_matrix = self.get_disp_vector_matrix(self.qd_locations)
        for i in np.arange(self.n):
            for j in np.arange(i+1, self.n):
                self.hamil[i, j] = self.J_c * self.kappa_polaron * self.get_kappa(self.qddipoles[i, :], self.qddipoles[j, :], self.qd_locations[i, :], self.qd_locations[j, :]) \
                    * 1/np.linalg.norm(displacement_vector_matrix[:, i, j])**3
                self.hamil[j, i] = self.hamil[i, j]
        [self.eignrgs, self.eigstates] = utils.diagonalize(self.hamil)


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

        
    def make_kmc_step(self, start_site):
        
        """
        perform a single KMC step from site coordinate (start_site) by locally diagoanlizing
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

    # HH : compute the full rate matrix for the entire grid (just as a sanity check
    # otherwise takes way too long to compute)
    def make_kmatrix_tensorFull(self, center):
        # create list of system_bath Hamiltonians
        J = self.hamil - np.diag(np.diag(self.hamil))  # diagonal elements need to be 0
        dim = len(self.hamil)
        ham_sysbath = []
        for i in range(dim):
            ham_list=[]
            for j in range(dim):
                ham_coupl=np.zeros((dim,dim))
                ham_coupl[i,j]=J[i,j]
                ham_list.append(ham_coupl)
            ham_sysbath.append(ham_list)   
        
        my_fullham = hamiltonian_box.HamiltonianFull(self.eignrgs, self.eigstates,
                                                    ham_sysbath, self.spectrum, const.kB * self.temp)
        my_fullredfield = redfield_box.RedfieldFull(my_fullham, self.kappa_polaron)

        # make Redfield rates for final polaron states in box
        self.ratesFull = my_fullredfield.make_redfield_tensor(center)
    
    # HH : see above only for sanity check
    def test_full_rates(self, start_site):
        # convert coordinate to index
        eig_state_locs = np.matmul(self.eigstates ** 2, self.qd_locations)
        start_idx = self.get_closest_idx(start_site, eig_state_locs)
        # print(rates)
        #self.make_kmatrix_full(start_idx)
        start = time.time()
        self.make_kmatrix_tensorFull(start_idx)
        end = time.time()
        print('time taken to compute rates', end - start)
        print('full rates', self.ratesFull)
        print('length rate vector', len(self.ratesFull))
        print('cum rates', np.sum(self.ratesFull))
        
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
            print("{} KMC trajectories evolved, with {} KMC steps and an sds of {} before t_final is reached!". format(n+1, self.step_counter, self.sds[-1]))
            if self.sds[-1] > 10000:
                print("uh oh {}".format(self.sds[-1]))
        return times_msds, msds

    
    def get_closest_idx(self, pos, array):
        """
        auxiliary function: find index of array of coordinates that is closest to pos
        """
        pos = pos.copy()
        array= array.copy()
        
        # account for periodic boundary conditions in pos
        for j in range(self.dims):
            if pos[j] > self.boundary - 1/2*self.qd_spacing: pos[j] = pos[j] - self.boundary
            elif pos[j] < -1/2*self.qd_spacing: pos[j] = pos[j] + self.boundary
        
        # account for periodic boundary conditions in array
        for i, coords in enumerate(array):
            for j in range(self.dims):
                if coords[j] > self.boundary - 1/2*self.qd_spacing: coords[j] = coords[j] - self.boundary
                elif coords[j] <-1/2*self.qd_spacing: coords[j] = coords[j] + self.boundary
        
        # find closest index
        idx = np.argmin([np.linalg.norm(pos - coord) for coord in array])
        return idx
    
    
    def get_diffusivity_hh(self, msds, times, dims):
        # note : I here assume that the whole time arrange is approx. linear (might break down)
        fit_params, cov = np.polyfit(times, msds, 1, cov=True)
        diff = fit_params[0]/(2*dims)
        # obtain error on diffusvity as from error on slope parameter 
        diff_err = np.sqrt(np.diag(cov))[0]/(2*dims)
        return diff, diff_err





