import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.fft import dct, dst
from . import const
from . import utils
import warnings
import time


class HamiltonianSystem():

    def __init__(self, evals, eigstates):

        self.init_system(evals, eigstates)

    def copy(self):
        import copy
        return copy.deepcopy(self)
        
    def init_system(self, evals, eigstates):
        self.nsite = np.size(evals)
        self.evals, self.Umat = evals, eigstates
        
        self.omega_diff = np.zeros((self.nsite,self.nsite)) # used for transformation to interaction picture and parameter Omega
        for i in range(self.nsite):
            for j in range(self.nsite):
                self.omega_diff[i,j] = (self.evals[i] - self.evals[j])


    def site2eig(self, rho):
        def _site2eig(rho2):
            return utils.matrix_dot(self.Umat.conj().T, rho2, self.Umat)
        return utils.transform_rho(_site2eig, rho)

    def eig2site(self, rho):
        def _eig2site(rho2):
            return utils.matrix_dot(self.Umat, rho2, self.Umat.conj().T)
        return utils.transform_rho(_eig2site, rho)


class Hamiltonian(HamiltonianSystem):

    def __init__(self, evals, eigstates, qd_lattice,
                 ham_sysbath, spec_density, kT):
        # Set these shared constants
        const.kT = kT
        
        self.qd_lattice_rel = qd_lattice

        self.init_system(evals, eigstates)
        self.sysbath = ham_sysbath

        # sepctral density
        if type(spec_density) != SpecDens:
            max_energy_diff = np.max(evals) - np.min(evals)
            self.spec = SpecDens(spec_density, max_energy_diff)
        else:
            self.spec = spec_density

class _PhiTransformer:
    """
    Fast & accurate Eq. (17) via Gauss–Legendre quadrature on [0, W],
    evaluated vectorially on a uniform τ-grid for later FFT use.
      φ(τ) = ∫_0^W dω/π * J(ω)/ω^2 [ cos(ωτ) coth(βω/2) - i sin(ωτ) ].
    """
    def __init__(self, J_callable, beta, W, N_tau, Q=512):
        """
        J_callable(ω): spectral density (vectorized over ω≥0)
        beta         : 1/(k_B T)
        W            : frequency cutoff for the ω-integral
        N_tau        : number of τ-grid points (use power of two for FFT friendliness, e.g. 16384)
        Q            : number of GL nodes (accuracy knob; 256–1024 typical)
        """
        self.J = J_callable
        self.beta = float(beta)
        self.W = float(W)
        self.N_tau = int(N_tau)
        self.Q = int(Q)

        # --- Build τ-grid for FFT step
        self.dtau = np.pi / self.W
        self.tau_grid = np.arange(self.N_tau) * self.dtau

        # --- Gauss–Legendre nodes ξ_j ∈ (-1,1) and weights w_j
        xi, w = leggauss(self.Q)                   # exact for polynomials ≤ degree 2Q-1
        # Map to ω ∈ (0, W): ω = (W/2)(ξ+1), dω = (W/2) dξ
        omega = 0.5 * self.W * (xi + 1.0)
        dω = 0.5 * self.W * w                      # GL weights on [0, W]

        # --- Build kernels G_R(ω), G_I(ω)
        Jw = np.asarray(self.J(omega), dtype=float)
        eps = 1e-300
        denom = np.maximum(omega**2, eps)

        x = 0.5 * self.beta * omega
        # stable coth
        coth = np.empty_like(x)
        small = np.abs(x) < 1e-6
        coth[~small] = 1.0 / np.tanh(x[~small])
        xs = x[small]
        coth[small] = 1.0/xs + xs/3.0 - (xs**3)/45.0

        GR = (Jw / (np.pi * denom)) * coth       # real kernel multiplier
        GI = (Jw / (np.pi * denom))              # imag kernel multiplier

        # --- Vectorized evaluation on τ-grid: outer products of ω and τ
        ωτ = np.outer(omega, self.tau_grid)      # shape (Q, N_tau)

        cos_mat = np.cos(ωτ)
        sin_mat = np.sin(ωτ)

        # Integrals: sum_j dω_j * G(ω_j) * cos/sin(ω_j τ_n)
        phi_real = (dω * GR) @ cos_mat           # shape (N_tau,)
        phi_imag = (dω * GI) @ sin_mat

        # Assemble φ(τ) on grid: Re − i * Im  (matches Eq. 17)
        self.phi_grid = phi_real - 1j * phi_imag

    def phi(self, tau):
        """
        Interpolate φ(τ) from the precomputed τ-grid (piecewise linear).
        (You can switch to PCHIP/CubicSpline if you prefer.)
        """
        tau = np.atleast_1d(tau).astype(float)
        # piecewise-linear interpolation with clamp-to-zero outside grid
        t = self.tau_grid
        idx = np.clip(np.searchsorted(t, tau) - 1, 0, t.size - 2)
        t0 = t[idx]; t1 = t[idx + 1]
        w = (tau - t0) / np.maximum(t1 - t0, 1e-300)
        φ0 = self.phi_grid[idx]
        φ1 = self.phi_grid[idx + 1]
        out = (1 - w) * φ0 + w * φ1
        out[(tau < t[0]) | (tau > t[-1])] = 0.0
        return out if out.ndim else out[()]


class _BathCorrFFT:
    """Fast Eq. (15) via complex FFT on τ-grid from _PhiTransformer with caching."""
    def __init__(self, phi_tr, omega_c, default_eta=None, window=None, win_beta=8.0):
        self.tr = phi_tr
        self.omega_c = float(omega_c)
        self.dt = self.tr.dtau
        self.tau = self.tr.tau_grid
        self.phi_tau = self.tr.phi_grid  # complex φ(τ) on grid
        self.window = window
        self.win_beta = float(win_beta)
        self.default_eta = (1e-3 * self.omega_c) if default_eta is None else float(default_eta)

        # Full FFT frequency grid (positive & negative)
        omega_full = 2.0 * np.pi * np.fft.fftfreq(self.tau.size, d=self.dt)
        self._pos_mask = omega_full >= 0
        self.omega_grid = omega_full[self._pos_mask]   # keep non-negative ω only

        # Cache: (lamda, kappa, eta, window, win_beta) -> K_grid (non-negative ω)
        self._cache = {}

    def _build(self, lamda, kappa, eta):
        key = (float(lamda), float(kappa), float(eta), self.window, self.win_beta)
        if key in self._cache:
            return self._cache[key]

        if kappa == 0.0 or lamda == 0.0:
            K_pos = np.zeros_like(self.omega_grid, dtype=complex)
            self._cache[key] = K_pos
            return K_pos

        # C(τ) (Eq. 16) and damping
        C = (kappa**2) * (np.exp(lamda * self.phi_tau) - 1.0)
        f = C * np.exp(-eta * self.tau)
        if self.window == 'kaiser':
            f = f * np.kaiser(self.tau.size, self.win_beta)
        elif self.window == 'hann':
            f = f * np.hanning(self.tau.size)

        # Complex FFT; our convention needs +iωτ, numpy uses -iωτ → conjugate
        F_full = self.dt * np.fft.fft(f)
        K_full = np.conj(F_full)

        # Keep non-negative frequencies
        K_pos = K_full[self._pos_mask]
        self._cache[key] = K_pos
        return K_pos

    def eval(self, omega, lamda, kappa, eta=None, return_grid=False):
        if eta is None:
            eta = self.default_eta
        Kgrid = self._build(lamda, kappa, eta)
        if return_grid:
            return self.omega_grid, Kgrid

        omega = np.atleast_1d(omega).astype(float)
        Re = np.interp(omega, self.omega_grid, Kgrid.real, left=0.0, right=0.0)
        Im = np.interp(omega, self.omega_grid, Kgrid.imag, left=0.0, right=0.0)
        out = Re + 1j*Im
        return out if out.ndim else out[()]


class SpecDens:
    def __init__(self, spec_dens_list, max_energy_diff):
        sd_type = spec_dens_list[0]
        self.bath_method = spec_dens_list[-1]
        self.max_energy_diff = max_energy_diff  # if you need it elsewhere

        if sd_type == 'cubic-exp':
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.J = self.cubic_exp
            self.low_freq_cutoff = self.omega_c / 200.0
            self.omega_inf = 20 * self.omega_c

        # Build fast φ(τ) (Eq. 17) and FFT engine (Eq. 15) if using 'exact'
        if self.bath_method == 'exact':
            beta = 1.0 / const.kT
            # knobs: W and N; adjust if you need tighter accuracy
            W = 40.0 * self.omega_c
            #N = 16385  # ~2^14+1
            N_tau = 16384             # power of two recommended
            Q = 512                   # try 512; bump to 1024 if you need tighter φ
            self._phi_tr = _PhiTransformer(self.J, beta, W, N_tau, Q=Q)
            self._fft = _BathCorrFFT(self._phi_tr, self.omega_c, default_eta=1e-3*self.omega_c, window=None)
            # API compatibility:
            self.Phi = self._phi_tr.phi
            self.correlationFT = self._correlationFT_fft
            # verify
            self.verify()

        elif self.bath_method == 'first-order':
            raise ValueError("Not implemented in this SpecDens class")

        else:
            raise SystemExit("Unknown bath_method")

    def cubic_exp(self, omega):
        w = abs(omega)
        Jw = (self.lamda / (2 * self.omega_c**3)) * w**3 * np.exp(-w / self.omega_c)
        return Jw * (omega >= 0) - Jw * (omega < 0)


    # fast Eq. (15) via FFT using cached grids
    def _correlationFT_fft(self, omega, lamda, kappa, eta=None, return_grid=False):
        # support scalar or array ω
        return self._fft.eval(omega, lamda=float(lamda), kappa=float(kappa), eta=eta, return_grid=return_grid)
    

    def verify(self, tol=1e-8, kappa_test=1.0, eta_factor=1e-3, verbose=True):
        """
        Run internal consistency checks for Eqs. (17)->(16)->(15).
        Returns a dict with numerical errors and booleans.
        """
        out = {}

        if getattr(self, 'bath_method', None) != 'exact' or not hasattr(self, '_phi_tr'):
            raise RuntimeError("verify() requires bath_method='exact' with the fast transformers initialized.")

        # --- Grab grids
        tau = self._phi_tr.tau_grid
        phi_grid = self._phi_tr.phi_grid
        dt = self._phi_tr.dtau

        # 1) Im φ(0) ≈ 0
        im0 = float(abs(phi_grid.imag[0]))
        out['imag_phi_at_0'] = im0
        out['imag_phi_at_0_ok'] = (im0 < tol)

        # 2) Symmetry φ(-τ) = φ(τ)* on the grid (discrete reversal)
        sym_err = float(np.max(np.abs(phi_grid - np.conj(phi_grid[::-1]))))
        out['phi_conj_sym_err'] = sym_err
        out['phi_conj_sym_ok'] = (sym_err < 10*tol)

        # 3) Slope at 0 for cubic-exp J: φ'(0) = -i * (lamda/π)
        #    Numeric slope from first two samples
        if hasattr(self, 'cubic_exp') and self.J == self.cubic_exp:
            # forward diff (more stable here since tau[0]=0)
            dphi_num = (phi_grid[1] - phi_grid[0]) / (tau[1] - tau[0])
            dphi_target = -1j * (self.lamda / np.pi)
            slope_err = complex(dphi_num - dphi_target)
            out['phi_prime_num'] = dphi_num
            out['phi_prime_target'] = dphi_target
            out['phi_prime_abs_err'] = float(abs(slope_err))
            out['phi_prime_rel_err'] = float(abs(slope_err) / max(1e-30, abs(dphi_target)))
            out['phi_prime_ok'] = (abs(slope_err) < 100*tol)
        else:
            out['phi_prime_ok'] = None  # not applicable for non-cubic-exp J

        # 4) λ=0 ⇒ C(τ)=0 ⇒ K(ω)=0 (Eq. 16->15)
        omegas = np.linspace(0.0, min(10*self.omega_c, 0.5/ dt)*2*np.pi, 256)
        K0 = self._fft.eval(omegas, lamda=0.0, kappa=kappa_test)
        null_err = float(np.max(np.abs(K0)))
        out['lambda0_K_maxabs'] = null_err
        out['lambda0_ok'] = (null_err < 100*tol)

        # 5) η-extrapolation consistency: ||K_η - K_{2η}||
        eta = eta_factor * self.omega_c
        K1 = self._fft._build(lamda=1.0, kappa=kappa_test, eta=eta)
        K2 = self._fft._build(lamda=1.0, kappa=kappa_test, eta=2*eta)
        # Evaluate on native grid to avoid interp noise:
        diff_norm = float(np.linalg.norm(K1 - K2, ord=np.inf))
        ref_norm = float(np.linalg.norm(K1, ord=np.inf) + 1e-30)
        out['eta_diff_inf'] = diff_norm
        out['eta_diff_rel'] = diff_norm / ref_norm
        out['eta_consistent'] = (out['eta_diff_rel'] < 1e-2)  # heuristic; smaller is better

        if verbose:
            print("[SpecDens.verify] Eq.17: Im φ(0) =", im0, " OK?" , out['imag_phi_at_0_ok'])
            print("[SpecDens.verify] Eq.17 symmetry ||φ - φ*rev||_∞ =", sym_err, " OK?" , out['phi_conj_sym_ok'])
            if out['phi_prime_ok'] is not None:
                print("[SpecDens.verify] φ'(0) num =", out['phi_prime_num'],
                    " target =", out['phi_prime_target'],
                    " |err| =", out['phi_prime_abs_err'],
                    " rel =", out['phi_prime_rel_err'],
                    " OK?", out['phi_prime_ok'])
            print("[SpecDens.verify] Eq.16→15 λ=0 ⇒ K=0: max|K| =", null_err, " OK?" , out['lambda0_ok'])
            print("[SpecDens.verify] η-consistency: rel ||K_η - K_{2η}||_∞ =", out['eta_diff_rel'],
                " OK?" , out['eta_consistent'])
        return out


# previous version 
class SpecDensOld():

    def __init__(self, spec_dens_list, max_energy_diff):
        sd_type = spec_dens_list[0]
        self.bath_method = spec_dens_list[-1]
        print('num called')
        
        if sd_type == 'cubic-exp':
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.J = self.cubic_exp
            self.low_freq_cutoff = self.omega_c/200
            self.omega_inf = 20*self.omega_c
            #self.cheby_tau_cutoff = 40/self.omega_c
            #omega_grid = np.linspace(1e-6, 20 * self.omega_c, 5000)
            omega_grid = np.linspace(1e-6, 20 * self.omega_c, 5000)

            # real part of bath correlation function 
            def re_bath_corr(omega):
                def coth(x):
                    return 1.0/np.tanh(x)

                beta = 1.0/const.kT
                omega += 1e-14
                n_omega = 0.5*(coth(beta*omega/2) - 1.0)
                return (self.J(omega)*(n_omega+1))/omega**2
            
            re_vals = re_bath_corr(omega_grid)
            print('vectorized real bath correlation function', re_vals.shape, flush = True)

            hilb = np.imag(hilbert(re_vals))
            # Interpolators
            self.re_interp = interp1d(omega_grid, re_vals, kind='cubic', bounds_error=False, fill_value=0.0)
            self.hilb_interp = interp1d(omega_grid, hilb, kind='cubic', bounds_error=False, fill_value=0.0)

        
        if self.bath_method == 'exact':
            self.Phi = self.phi     # use exact phi
            self.correlationFT = self.bathCorrFT
        
        elif self.bath_method == 'first-order':
            #self.correlationFT = self.firstOrderFT
            self.correlationFT = self.fastfirstOrderFT
        else:
            raise SystemExit

            
    def cubic_exp(self, omega):
        """Evaluate a cubic (super-Ohmic) spectral density with an exponential
        cutoff.
        """
        w = abs(omega)
        Jw = (self.lamda/(2*self.omega_c**3))*w**3*np.exp(-w/self.omega_c)
        return Jw*(omega >= 0) - Jw*(omega < 0)
    
            
    # compute phi based on Eq. (17)
    def phi(self, tau):
        
        beta = 1.0/const.kT
        uppLim=np.inf
        uppLim= 100 * self.omega_c
        
        if type(tau) == float or type(tau) == int or type(tau) == np.float64:
            tau = np.array([tau])
        phi_real = np.zeros(np.shape(tau))
        phi_imag = np.zeros(np.shape(tau))
        
        for i in np.arange(np.size(tau)):
            if tau[i] > 0 and tau[i] < self.low_freq_cutoff:
                # fix low frequencies
                # real part of integrand
                def integrand_real(omega, tau):
                    return 1/(np.pi*omega**2)*self.J(omega)/np.tanh(beta*omega/2) * np.cos(tau[i] * omega)
                
                # imaginary part of integral
                def integrand_imag(omega, tau):
                    return 1/(np.pi*omega**2)*self.J(omega) * np.sin(tau[i] * omega)
                
                phi_real[i]=integrate.quad(integrand_real, 1E-14, uppLim, args=(tau,))[0]
                phi_imag[i]=integrate.quad(integrand_imag, 1E-14, uppLim, args=(tau,))[0]
            else:
                # real part of integrand
                def integrand_real(omega, tau):
                    return 1/(np.pi*omega**2)*self.J(omega)/np.tanh(beta*omega/2)
                
                # imaginary part of integral
                def integrand_imag(omega, tau):
                    return 1/(np.pi*omega**2)*self.J(omega)
                
                phi_real[i]=integrate.quad(integrand_real, 1E-14, uppLim, args=(tau,), weight='cos', wvar=tau[i], limit=200)[0]
                phi_imag[i]=integrate.quad(integrand_imag, 1E-14, uppLim, args=(tau,), weight='sin', wvar=tau[i], limit=200)[0]
        return phi_real-1j*phi_imag
    
       
    # perform half-sided Fourier trasnform of bath correlation function based on Eq. (15)
    def bathCorrFT(self, omega, lamda, kappa):
        if type(omega) == float or type(omega) == int or type(omega) == np.float64:
            omega = np.array([omega])
        if lamda==0:
            return np.zeros(np.shape(omega))
        else:
            bathCorrFT_real1 = np.zeros(np.shape(omega))
            bathCorrFT_real2 = np.zeros(np.shape(omega))
            bathCorrFT_imag1 = np.zeros(np.shape(omega))
            bathCorrFT_imag2 = np.zeros(np.shape(omega))

            def integrandFT_real(tau):
                return kappa**2*np.real(np.exp(-lamda*self.Phi(tau))-1)
            def integrandFT_imag(tau):
                return kappa**2*np.imag(np.exp(-lamda*self.Phi(tau))-1)
            
            # perform half-sided Fourier transform (real/imaginary parts separately)
            uppLim=np.inf
            uppLim = 100 * self.omega_c
            lowLim=1E-14
            for i in np.arange(np.size(omega)):
                bathCorrFT_real1[i]=integrate.quad(integrandFT_real, lowLim, uppLim, limit=200, weight='cos', wvar=omega[i], limlst = 200)[0]
                bathCorrFT_real2[i]=integrate.quad(integrandFT_imag, lowLim, uppLim, limit=200, weight='sin', wvar=omega[i], limlst = 200)[0]
                bathCorrFT_imag1[i]=integrate.quad(integrandFT_imag, lowLim, uppLim, limit=200, weight='cos', wvar=omega[i], limlst = 200)[0]
                bathCorrFT_imag2[i]=integrate.quad(integrandFT_real, lowLim, uppLim, limit=200, weight='sin', wvar=omega[i], limlst = 200)[0]

            return bathCorrFT_real1+bathCorrFT_real2+1j*bathCorrFT_imag1-1j*bathCorrFT_imag2



    # first order approximation to the bath correlation function in Eq. (16)
    def firstOrderFT(self, omega, lamda, kappa):
        
        def re_bath_corr(omega):
            def coth(x):
                return 1.0/np.tanh(x)

            beta = 1.0/const.kT
            omega += 1e-14
            n_omega = 0.5*(coth(beta*omega/2) - 1.0)
            return (self.J(omega)*(n_omega+1))/omega**2
        
        # quad integration (expensive)
        omega = -omega
        ppv = integrate.quad(re_bath_corr, 
                             -self.omega_inf, self.omega_inf,
                             limit=1000, weight='cauchy', wvar=omega)
        ppv = -ppv[0]
        return -kappa**2*lamda*(re_bath_corr(omega) + (1j/np.pi)*ppv)
    

    def fastfirstOrderFT(self, omega, lamda, kappa):
        
        omega = np.abs(omega)
        real_part = self.re_interp(omega)
        imag_part = self.hilb_interp(omega)
        return -kappa**2 * lamda * (real_part + 1j * imag_part / np.pi)




