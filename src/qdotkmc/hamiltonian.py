import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.signal import hilbert
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.fft import dct, dst
from . import const
from . import utils


class Hamiltonian():

    def __init__(self, evals, eigstates, J_dense=None):


        self.evals = np.asarray(evals, dtype=np.float64)                            # eigenvalues
        self.Umat  = np.asarray(eigstates, dtype=np.complex128, order='C')          # eigenvectors

        self.spec = None                                                            # SpecDense instance initialization
        self.beta = None

        self._init_system(self.evals, self.Umat)
        self.J_dense = np.asarray(J_dense, dtype=np.float64, order='C')


    # TODO : do we need this? maybe just put this into __init__
    def _init_system(self, evals, eigstates):
        self.nsite = int(np.size(evals))
        self.omega_diff = np.subtract.outer(self.evals, self.evals)


class _PhiTransformer:
    """Accurate Eq. (17) on a fixed (τ) grid via direct quad integration."""

    def __init__(self, J_callable, beta, omega_c, omega_inf, low_freq_cutoff, N_tau=2000, tau_max_factor=70.0):
        
        self.J = J_callable
        self.beta = float(beta)
        self.omega_c = float(omega_c)

        # set up τ grid
        self.N_tau = int(N_tau)
        self.tau_max = tau_max_factor / self.omega_c
        self.tau_grid = np.linspace(0.0, self.tau_max, self.N_tau)
        phi_real = np.zeros_like(self.tau_grid)
        phi_imag = np.zeros_like(self.tau_grid)

        # integration limits for integral over ω 
        uppLim = omega_inf
        lowLim = 1e-12

        for i, tau in enumerate(self.tau_grid):
            if tau > 0 and tau < (low_freq_cutoff):
                # no quad weights
                def integrand_real(omega):
                    return 1/(np.pi*omega**2)*self.J(omega)/np.tanh(beta*omega/2) * np.cos(tau * omega)
                def integrand_imag(omega):
                    return 1/(np.pi*omega**2)*self.J(omega) * np.sin(tau * omega)
                phi_real[i] = integrate.quad(integrand_real, lowLim, uppLim)[0]
                phi_imag[i] = integrate.quad(integrand_imag, lowLim, uppLim)[0]
            else:
                def integrand_real(omega):
                    return 1/(np.pi*omega**2)*self.J(omega)/np.tanh(beta*omega/2)
                def integrand_imag(omega):
                    return 1/(np.pi*omega**2)*self.J(omega)
                phi_real[i] = integrate.quad(integrand_real, lowLim, uppLim, weight='cos', wvar=tau, limit=200)[0]
                phi_imag[i] = integrate.quad(integrand_imag, lowLim, uppLim, weight='sin', wvar=tau, limit=200)[0]

        self.phi_grid = phi_real - 1j * phi_imag

    def phi(self, tau):
        tau = np.atleast_1d(tau).astype(float)
        re = np.interp(tau, self.tau_grid, self.phi_grid.real, left=0.0, right=0.0)
        im = np.interp(tau, self.tau_grid, self.phi_grid.imag, left=0.0, right=0.0)
        out = re - 1j*im
        return out if out.ndim else out[()]


class _BathCorrFFT:
    """Half-sided Eq. (15) via FFT on τ ∈ [0,T] """
    def __init__(self, phi_tr, omega_c):
        self.tr = phi_tr
        self.omega_c = float(omega_c)

        # τ-grid supplied by the accurate φ-transformer (uniform linspace)
        self.tau = self.tr.tau_grid
        self.dt  = self.tau[1] - self.tau[0]
        self.phi_tau = self.tr.phi_grid  # complex φ(τ) on this grid

        # frequency grid in angular freq (rad/time). Keep ω ≥ 0.
        f_full = np.fft.fftfreq(self.tau.size, d=self.dt)     # cycles / time
        omega_full = 2.0 * np.pi * f_full                     # rad / time
        self._pos_mask = (omega_full >= 0.0)
        self.omega_grid = omega_full[self._pos_mask]

        # cache: (lamda,kappa) -> K(ω≥0)
        self._cache = {}

    def _build(self, lamda, kappa):
        key = (float(lamda), float(kappa))
        if key in self._cache:
            return self._cache[key]

        if kappa == 0.0 or lamda == 0.0:
            K_pos = np.zeros_like(self.omega_grid, dtype=complex)
            self._cache[key] = K_pos
            return K_pos

        # C(τ) = κ^2 (e^{-λ φ(τ)} − 1)
        C = (kappa**2) * (np.exp(-lamda * self.phi_tau) - 1.0)

        # FFT is ∑ f(τ) e^{-iωτ}; conjugate → e^{+iωτ}. Riemann factor is dt.
        F_full = self.dt * np.fft.fft(C)
        K_full = np.conj(F_full)

        K_pos = K_full[self._pos_mask]
        self._cache[key] = K_pos
        return K_pos

    def eval(self, omega, lamda, kappa, return_grid=False, omega_is_energy=True):
        Kpos = self._build(lamda, kappa)
        if return_grid:
            return self.omega_grid, Kpos

        w = np.atleast_1d(omega).astype(float)
        if omega_is_energy:
            w = w / const.hbar  # ΔE → ω

        out = np.empty(w.shape, dtype=complex)
        mask_pos = (w >= 0.0)
        wp = w[mask_pos]
        wn = -w[~mask_pos]  # reflect negatives

        # linear interp with edge clamping (avoid accidental zeros just outside grid)
        def interp_edge(x, xp, fp):
            y = np.interp(x, xp, fp, left=fp[0], right=fp[-1])
            return y

        Re_p = interp_edge(wp, self.omega_grid, Kpos.real)
        Im_p = interp_edge(wp, self.omega_grid, Kpos.imag)
        out[mask_pos] = Re_p + 1j*Im_p

        # K(-ω) = K(ω)^* for e^{+iωτ} with C(-τ)=C(τ)^*
        Re_n = interp_edge(wn, self.omega_grid, Kpos.real)
        Im_n = interp_edge(wn, self.omega_grid, Kpos.imag)
        out[~mask_pos] = np.conj(Re_n + 1j*Im_n)

        return out if out.ndim else out[()]


class SpecDens:

    def __init__(self, spec_dens_list, kT):
        sd_type = spec_dens_list[0]
        self.beta = 1.0 / kT
        self.spectrum = spec_dens_list

        if sd_type == 'cubic-exp':
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.J = self.cubic_exp
            self.low_freq_cutoff = self.omega_c / 200.0
            self.omega_inf = 40 * self.omega_c

        # Build fast φ(τ) (Eq. 17) and FFT engine (Eq. 15)
        self._phi_tr = _PhiTransformer(self.J, self.beta, self.omega_c, self.omega_inf, self.low_freq_cutoff)
        self._fft = _BathCorrFFT(self._phi_tr, self.omega_c)
        self.correlationFT = self._correlationFT_fft


    # cubic-exponential bath spectral density
    def cubic_exp(self, omega):
        w = abs(omega)
        Jw = (self.lamda / (2 * self.omega_c**3)) * w**3 * np.exp(-w / self.omega_c)
        return Jw * (omega >= 0) - Jw * (omega < 0)


    # fast Eq. (15) via FFT using cached grids
    def _correlationFT_fft(self, omega, lamda, kappa, eta=None, return_grid=False):
        return self._fft.eval(
                            omega, lamda=float(lamda), kappa=float(kappa),
                            return_grid=return_grid,
                            omega_is_energy=False   
                            )
    





