import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.signal import hilbert
from numpy.polynomial.legendre import leggauss
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
        if type(spec_density) != SpecDensOld:
            max_energy_diff = np.max(evals) - np.min(evals)
            self.spec = SpecDensOld(spec_density, max_energy_diff)
        else:
            self.spec = spec_density

class _PhiTransformer:
    """Fast Eq. (17) on a fixed (ω,τ) grid using DCT/DST + cubic splines."""
    def __init__(self, J_callable, beta, W, N, omega_min=0.0):
        self.J = J_callable
        self.beta = float(beta)
        self.W = float(W)
        self.N = int(N)
        self.dw = self.W / (self.N - 1)
        w = np.arange(self.N) * self.dw
        if omega_min > 0.0:
            w = np.maximum(w, omega_min)
        self.omega = w

        def _coth_half_beta_w(w_):
            x = 0.5 * self.beta * w_
            out = np.empty_like(x)
            small = np.abs(x) < 1e-6
            out[~small] = 1.0 / np.tanh(x[~small])
            xs = x[small]
            out[small] = 1.0/xs + xs/3.0 - (xs**3)/45.0
            return out

        Jw = np.asarray(self.J(w), dtype=float)
        eps = 1e-300
        denom = np.maximum(w**2, eps)
        GR = (Jw / (np.pi * denom)) * _coth_half_beta_w(w)
        GI = (Jw / (np.pi * denom))
        GI[0] = 0.0  # sin(0·τ)=0

        cos_sum = dct(GR, type=1, norm=None)
        sin_sum = dst(GI, type=1, norm=None)

        phi_real_grid = self.dw * cos_sum
        phi_imag_grid = self.dw * sin_sum
        self.phi_grid = phi_real_grid - 1j * phi_imag_grid

        self.dtau = np.pi / self.W
        self.tau_grid = np.arange(self.N) * self.dtau

        # light-weight cubic splines (manual) to avoid extra imports
        # (SciPy CubicSpline OK too; here we use numpy polyfit chunked)
        # For robustness/speed, we’ll just do linear interp; cubic gives tiny gains here.
        self._phi_re = self.phi_grid.real
        self._phi_im = self.phi_grid.imag

    def _interp1(self, xq, x, y):
        # vectorized piecewise-linear; out-of-range -> 0
        xq = np.atleast_1d(xq)
        idx = np.clip(np.searchsorted(x, xq) - 1, 0, x.size - 2)
        x0 = x[idx]; x1 = x[idx + 1]
        y0 = y[idx]; y1 = y[idx + 1]
        w = (xq - x0) / np.maximum(x1 - x0, 1e-300)
        out = y0 * (1 - w) + y1 * w
        out[(xq < x[0]) | (xq > x[-1])] = 0.0
        return out

    def phi(self, tau):
        tau = np.atleast_1d(tau).astype(float)
        re = self._interp1(tau, self.tau_grid, self._phi_re)
        im = self._interp1(tau, self.tau_grid, self._phi_im)
        out = re - 1j * im
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
            W = 15.0 * self.omega_c
            N = 16385  # ~2^14+1
            self._phi_tr = _PhiTransformer(self.J, beta, W, N, omega_min=1e-12)
            self._fft = _BathCorrFFT(self._phi_tr, self.omega_c, default_eta=1e-3*self.omega_c, window=None)
            # API compatibility:
            self.Phi = self._phi_tr.phi
            self.correlationFT = self._correlationFT_fft

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
            #self.correlationFT = self.bathCorrFT
            self.correlationFT = self.correlationFT_ref
        
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
        
    def correlationFT_ref(self, omega, lamda, kappa, eta=None,
                            epsabs=1e-9, epsrel=1e-7, limlst=500, limit=2000):
        """
        Reference implementation of Eq. (15) by direct quadrature on τ.
        Correct signs, half-transform, small damping and η-extrapolation.
        """
        omega = np.atleast_1d(omega).astype(float)
        out = np.empty_like(omega, dtype=complex)

        if lamda == 0 or kappa == 0:
            out[:] = 0.0
            return out if out.shape != () else out[()]

        if eta is None:
            eta = 1e-3 * self.omega_c

        # build C(τ) pieces with damping
        def A(t):  # a(τ) = Re C(τ) · e^{-ητ}
            z = kappa**2 * (np.exp(lamda * self.Phi(t)) - 1.0)
            return np.real(z) * np.exp(-eta * t)

        def B(t):  # b(τ) = Im C(τ) · e^{-ητ}
            z = kappa**2 * (np.exp(lamda * self.Phi(t)) - 1.0)
            return np.imag(z) * np.exp(-eta * t)

        def K_eta(w):
            A_cos = integrate.quad(A, 0.0, np.inf, weight='cos', wvar=w,
                                limlst=limlst, limit=limit, epsabs=epsabs, epsrel=epsrel)[0]
            B_sin = integrate.quad(B, 0.0, np.inf, weight='sin', wvar=w,
                                limlst=limlst, limit=limit, epsabs=epsabs, epsrel=epsrel)[0]
            A_sin = integrate.quad(A, 0.0, np.inf, weight='sin', wvar=w,
                                limlst=limlst, limit=limit, epsabs=epsabs, epsrel=epsrel)[0]
            B_cos = integrate.quad(B, 0.0, np.inf, weight='cos', wvar=w,
                                limlst=limlst, limit=limit, epsabs=epsabs, epsrel=epsrel)[0]
            Re = A_cos - B_sin         # <-- MINUS here
            Im = A_sin + B_cos         # <-- PLUS here
            return Re + 1j*Im

        # Richardson extrapolation to η→0+
        K1 = np.array([K_eta(w) for w in omega])
        old_eta = eta
        eta *= 2.0
        def A2(t): return A(t) * np.exp(-(eta-old_eta)*t)
        def B2(t): return B(t) * np.exp(-(eta-old_eta)*t)
        A, B = A2, B2  # reuse in closures
        K2 = np.array([K_eta(w) for w in omega])
        K = 2*K1 - K2

        return K if out.shape != () else K[()]



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




