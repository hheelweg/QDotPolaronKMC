import numpy as np
import math
from scipy import integrate
from . import hamiltonian, redfield, utils
from .config import GeometryConfig, DisorderConfig
from .hamiltonian import SpecDens

# class to set up QD Lattice 
class QDLattice():

    
    def __init__(self, geom : GeometryConfig, dis : DisorderConfig, seed_realization : int):

        # load geometric and energetic properties for set-up of lattice
        self.geom = geom
        self.dis = dis

        # set seed for 
        self.seed_realization = int(seed_realization)
        self.rng = np.random.default_rng(self.seed_realization)

        # initialize lattice
        self._make_lattice()

        # computing backend for QDLattice (GPU/CPU)
        self.backend = None

        
    # NOTE: old make_qd_array method (basically unchanged)
    def _make_lattice(self):    
        # set locations for each QD
        self.qd_locations = np.zeros((self.geom.n_sites, self.geom.dims))
        # if seed = None, then we draw a new random configuration every single time we call this method
        #np.random.seed(self.dis.seed_base)
        for i in np.arange(self.geom.n_sites):
            if self.geom.dims == 2:
                self.qd_locations[i, :] = (i%self.geom.N * self.geom.qd_spacing, np.floor(i/self.geom.N) * self.geom.qd_spacing) \
                    + self.rng.normal(0, self.geom.qd_spacing * self.dis.relative_spatial_disorder, [1, self.geom.dims])
            elif self.geom.dims == 1:
                self.qd_locations[i, :] = (i%self.geom.N * self.geom.qd_spacing) \
                    + self.rng.normal(0, self.geom.qd_spacing * self.dis.relative_spatial_disorder, [1, self.geom.dims])
            elif self.geom.dims == 3:
                raise NotImplementedError("3 dimensions currently not implemented!")
            
        self.qd_locations[self.qd_locations < 0] = self.qd_locations[self.qd_locations < 0] + self.geom.N * self.geom.qd_spacing
        self.qd_locations[self.qd_locations > self.geom.N * self.geom.qd_spacing] = \
            self.qd_locations[self.qd_locations > self.geom.N * self.geom.qd_spacing] - self.geom.N * self.geom.qd_spacing
        # set nrgs for each QD
        self.qdnrgs = self.rng.normal(self.dis.nrg_center, self.dis.inhomog_sd, self.geom.n_sites)
        
        # set dipole moment orientation for each QD
        if self.dis.dipolegen == 'random':
            self.qddipoles = self.rng.normal(0, 1, [self.geom.n_sites, 3])
            self.qddipoles = self.qddipoles/np.array([[i] for i in np.linalg.norm(self.qddipoles, axis = 1)])
        elif self.dis.dipolegen == 'alignZ':
            self.qddipoles = np.zeros([self.geom.n_sites, 3])
            self.qddipoles[:, 2] = np.ones(self.geom.n_sites)
        else:
            raise Exception("Invalid dipole generation type") 
        
        self.stored_npolarons_box = np.zeros(self.geom.n_sites)
        self.stored_polaron_sites = [np.array([]) for _ in np.arange(self.geom.n_sites)]
        self.stored_rate_vectors = [np.array([]) for _ in np.arange(self.geom.n_sites)]
    

    @staticmethod
    def _init_box_dims(r_hop, r_ove, spacing, max_length):
        # convert to actual units (scaled r_hop/r_ove)
        r_hop *= spacing
        r_ove *= spacing
        # box radius and dimensions:
        # NOTE : this uses box_radius = min(r_hop, r_ove) rounded to the next higher integer
        box_radius = math.ceil(min(r_hop / spacing, r_ove /spacing))
        # return quadratic box-length in actual units
        box_length = (2 * box_radius + 1) * spacing
        # raise wanring if lattice dimensions are exceeded
        if box_length > max_length * spacing:
            raise ValueError('The lattice dimensions are exceeded! Please choose r_hop and r_ove accordingly!')
        return r_hop, r_ove, box_length
    

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


    def _build_J_backend(self, qd_pos, qd_dip, J_c, kappa_polaron, boundary=None,
                        backend=None, tile_ij=1024):
        """
        Fast pairwise dipole coupling builder with tiling and CPU/GPU backend.

        J_ij = J_c * kappa_polaron *
            [ μ_i·μ_j - 3 (μ_i·r̂_unwrap)(μ_j·r̂_unwrap) ] / ||r_wrap||^3

        - Magnitude uses WRAPPED displacement (minimum image) like your original.
        - Direction uses UNWRAPPED displacement (matches your get_kappa path).
        - Works for 1D/2D positions embedded in 3D dipole space.
        - Computes upper triangle in tiles and mirrors (saves ~2x time/mem).
        - Avoids allocating (n,n,3) tensors; peak memory ~ O(tile_ij^2).

        Parameters
        ----------
        qd_pos : (n, d) float64, d in {1,2}
        qd_dip : (n, 3) float64   # dipoles in 3D
        J_c, kappa_polaron : float
        boundary : Optional[float]   # box length for wrapping (None → no wrap)
        backend : Backend            # your backend (has .xp, .f, .to_host)
        tile_ij : int                # tile size along i/j (tune per VRAM/CPU RAM)

        Returns
        -------
        J : (n, n) float64 on host
        """
        bk = backend
        xp = bk.xp
        f  = bk.f   # float32 (GPU if use_c64) or float64 (CPU default)

        qd_pos = np.asarray(qd_pos, dtype=np.float64, order="C")   # host input
        qd_dip = np.asarray(qd_dip, dtype=np.float64, order="C")

        n, d = qd_pos.shape
        assert d in (1, 2)

        # --- embed positions to 3D once (host) ---
        pos3 = np.zeros((n, 3), dtype=np.float64)
        pos3[:, :d] = qd_pos
        mu = qd_dip
        mu_unit = mu / np.linalg.norm(mu, axis=1, keepdims=True)  # host

        # Upload small, reused arrays (GPU) or keep as NumPy (CPU)
        pos3_g   = bk.from_host(pos3, dtype=f, order="C")
        mu_unit_g= bk.from_host(mu_unit, dtype=f, order="C")

        J = np.zeros((n, n), dtype=np.float64, order="C")  # we’ll fill on host

        L = float(boundary) if boundary is not None else None
        scale = float(J_c) * float(kappa_polaron)

        # helper: wrap along first d dimensions (minimum image), in-place
        def _apply_wrap_inplace(delta, L, d):
            # delta shape: (Bi, Bj, 3), backend array
            if L is None:
                return
            # exact rule: (> L/2) -= L, (< -L/2) += L  on the first d axes
            for ax in range(d):
                v = delta[..., ax]
                mask_hi = v >  (L * 0.5)
                mask_lo = v < -(L * 0.5)
                # note: xp.where allocates; in-place boolean assignment is fine
                v[mask_hi] = v[mask_hi] - L
                v[mask_lo] = v[mask_lo] + L
            # remaining dims (2→z) untouched

        # Tiled upper triangle
        for i0 in range(0, n, tile_ij):
            i1 = min(i0 + tile_ij, n)
            Bi = i1 - i0

            # slice once for the tile along i
            posI = pos3_g[i0:i1, :]          # (Bi,3)
            muI  = mu_unit_g[i0:i1, :]       # (Bi,3)

            for j0 in range(i0, n, tile_ij):
                j1 = min(j0 + tile_ij, n)
                Bj = j1 - j0

                posJ = pos3_g[j0:j1, :]      # (Bj,3)
                muJ  = mu_unit_g[j0:j1, :]   # (Bj,3)

                # unwrapped and wrapped deltas (Bi,Bj,3)
                d_unwrap = posJ[None, :, :] - posI[:, None, :]
                d_wrap   = d_unwrap.copy()
                _apply_wrap_inplace(d_wrap, L, d)

                # r_wrap for magnitude
                r2 = xp.einsum('ijk,ijk->ij', d_wrap, d_wrap)         # (Bi,Bj)
                # unit vector from UNWRAPPED deltas (direction)
                r2_dir = xp.einsum('ijk,ijk->ij', d_unwrap, d_unwrap) # (Bi,Bj)

                # avoid divide-by-zero on the diagonal block
                if i0 == j0:
                    # set diagonal entries r2=inf so inv_r3→0; r2_dir=1 so rhat well-defined
                    idx = xp.arange(Bi)
                    r2[idx, idx] = xp.inf
                    r2_dir[idx, idx] = 1.0

                # --- safe inverses (no errstate) ---
                eps = xp.asarray(xp.finfo(f).tiny, dtype=f)
                r   = xp.sqrt(r2)
                den = r2 * r
                inv_r3 = xp.where(den > 0, 1.0 / den, 0.0).astype(f, copy=False)

                norm_dir = xp.sqrt(r2_dir)
                norm_dir = xp.maximum(norm_dir, eps)
                rhat = d_unwrap / norm_dir[:, :, None]

                # angular factor kappa = μi·μj - 3(μi·rhat)(μj·rhat)
                mui_dot_muj = muI @ muJ.T                        # (Bi,Bj)
                mui_dot_r   = xp.einsum('ik,ijk->ij', muI, rhat) # (Bi,Bj)
                # (rhat · μj) for each j: use μj on axis=1
                muj_dot_r   = xp.einsum('ijk,jk->ij', rhat, muJ) # (Bi,Bj)
                kappa = mui_dot_muj - 3.0 * (mui_dot_r * muj_dot_r)

                Jij = scale * kappa * inv_r3                     # (Bi,Bj)

                # bring this tile back to host and write into J (upper), mirror to lower
                Jij_h = bk.to_host(Jij).astype(np.float64, copy=False)
                J[i0:i1, j0:j1] = Jij_h
                if j0 != i0:
                    J[j0:j1, i0:i1] = Jij_h.T

        # zero diagonal to be safe
        np.fill_diagonal(J, 0.0)
        return J

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


    # setup polaron-transformed Hamiltonian
    def _setup_hamil(self, kappa_polaron, periodic = True):
        import time
        # (1) set up polaron-transformed Hamiltonian 
        # (1.1) coupling terms in Hamiltonian
        start = time.time()
        # J = self._build_J(
        #                 qd_pos=self.qd_locations,
        #                 qd_dip=self.qddipoles,
        #                 J_c=self.dis.J_c,
        #                 kappa_polaron=kappa_polaron,
        #                 boundary=(self.geom.boundary if periodic else None)
        #                 )
        J = self._build_J_backend(
                        qd_pos=self.qd_locations,
                        qd_dip=self.qddipoles,
                        J_c=self.dis.J_c,
                        kappa_polaron=kappa_polaron,
                        boundary=(self.geom.boundary if periodic else None),
                        backend=self.backend
                        )
        end = time.time()
        print(f"time taken for building J: {end-start:.4f}")
        # (1.2) site energies and total Hamiltonian
        self.hamil = np.diag(self.qdnrgs).astype(np.float64, copy=False)
        self.hamil += J
        


        # (2) keep original diagonalization routine
        start = time.time()
        self.eignrgs, self.eigstates = utils.diagonalize(self.hamil, self.backend)
        end = time.time()
        print(f"time taken for diagonalization: {end-start:.4f}")

        # (3) polaron positions 
        if periodic:
            locations_unit_circle = (self.qd_locations / self.geom.boundary) * (2*np.pi)  # (n,d)
            unit_circle_ycoords = np.sin(locations_unit_circle)
            unit_circle_xcoords = np.cos(locations_unit_circle)
            psi2 = self.eigstates**2                                                 # (n,n)
            unit_circle_eig_xcoords = (unit_circle_xcoords.T @ psi2).T               # == your transpose/matmul pattern
            unit_circle_eig_ycoords = (unit_circle_ycoords.T @ psi2).T
            eigstate_positions = np.arctan2(unit_circle_eig_ycoords, unit_circle_eig_xcoords) * (self.geom.boundary / (2*np.pi))
            eigstate_positions[eigstate_positions < 0] += self.geom.boundary
            self.polaron_locs = eigstate_positions
        else:
            self.polaron_locs = (self.qd_locations.T @ (self.eigstates**2)).T

        # (4) off-diagonal J for Redfield (system-bath)
        J_off = self.hamil - np.diag(np.diag(self.hamil))
        self.J_dense = J_off.copy()

        # (5) set up Hamilonian instance etc. 
        self.full_ham = hamiltonian.Hamiltonian(
            self.eignrgs, self.eigstates,
            J_dense = self.J_dense
            )
        
        # (6) optional : get IPR statistics
        ipr_mean, ipr_std = utils.get_ipr(self.eigstates)


    # setup instance of Redfield class
    def _setup_redfield(self):

        self.redfield = redfield.Redfield(
            self.full_ham, self.polaron_locs, self.qd_locations, self.kappa_polaron,
            self.backend,
            time_verbose=True
        )


    # this calls _setup_hamil and _setup_redfield, and links the bath to the QDLattice
    def _setup(self, bath):

        assert isinstance(bath, SpecDens), 'Need to specify valid SpecDens instance \
                                            to set up QDLattic Hamiltonian.'

        # set (inverse) temperature (do we still need this?)
        self.beta = bath.beta

        # compute 𝜅 for polaron transformation          
        kappa_polaron = self.get_kappa_polaron(bath.spectrum)

        # polaron-tranformed Hamiltonian
        self._setup_hamil(kappa_polaron)

        # add bath and temperature
        self.full_ham.spec = bath
        self.full_ham.beta = bath.beta

        # initialize instance of Redfield class
        self._setup_redfield()


    # NOTE : currently only implemented for cubic-exp spectral density
    # TODO : check this function and compar to paper as well
    # have this feed in a general spectral density type moving forward
    def get_kappa_polaron(self, spectrum = None, freq_max = 1):

        lamda = spectrum[1]
        omega_c = spectrum[2]
        
        # TODO: update this to account for different spectrum funcs
        spectrum_func = lambda w: (np.pi*lamda/(2*omega_c**3))*w**3*np.exp(-w/omega_c)
        integrand = lambda freq : 1/np.pi * spectrum_func(freq)/np.power(freq, 2) * 1/np.tanh(self.beta * freq/2)
        self.kappa_polaron = np.exp(-integrate.quad(integrand, 0, freq_max)[0])
        return self.kappa_polaron


