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
        
        # NOTE : do we need to divide by const.HBAR ? (08/09/2025)
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
    """Accurate Eq. (17) on a fixed (τ) grid via direct quad integration."""

    def __init__(self, J_callable, beta, omega_c, omega_inf, N_tau=2000, tau_max_factor=70.0):
        
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
            if tau > 0 and tau < (self.omega_c / 200.0):
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

    def __init__(self, spec_dens_list, max_energy_diff):
        sd_type = spec_dens_list[0]
        self.bath_method = spec_dens_list[-1]
        self.max_energy_diff = max_energy_diff  # if you need it elsewhere
        beta = 1.0 / const.kT

        if sd_type == 'cubic-exp':
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.J = self.cubic_exp
            self.low_freq_cutoff = self.omega_c / 200.0
            self.omega_inf = 40 * self.omega_c

        # Build fast φ(τ) (Eq. 17) and FFT engine (Eq. 15) if using 'exact'
        if self.bath_method == 'exact':
            self._phi_tr = _PhiTransformer(self.J, beta, self.omega_c, self.omega_inf)
            self._fft = _BathCorrFFT(self._phi_tr, self.omega_c)
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
        return self._fft.eval(
                            omega, lamda=float(lamda), kappa=float(kappa),
                            return_grid=return_grid,
                            omega_is_energy=False   
                            )
    


# previous version 
class SpecDensOld():

    def __init__(self, spec_dens_list, max_energy_diff):
        sd_type = spec_dens_list[0]
        self.bath_method = spec_dens_list[-1]
        print('num called old')
        
        if sd_type == 'cubic-exp':
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.J = self.cubic_exp
            self.low_freq_cutoff = self.omega_c/200
            self.omega_inf = 40*self.omega_c
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
            self.correlationFT = self.bathCorrFT_new
        
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
        
    # perform half-sided Fourier transform of bath correlation function based on Eq. (15)
    # K(ω) = ∫_0^∞ e^{iωτ} C(τ) dτ with C(τ) = κ^2 (exp(λ φ(τ)) - 1)
    def bathCorrFT_new(self, omega, lamda, kappa, tmax_factor=70.0, sign='plus'):
        """
        Half-sided FT of C(τ) = κ^2 (exp(λ Φ(τ)) - 1).
        sign='minus' implements K(ω)=∫_0^∞ e^{-iωτ} C(τ)dτ  (common in Redfield)
        sign='plus'  implements K(ω)=∫_0^∞ e^{+iωτ} C(τ)dτ
        """
        # normalize omega to array (keep your original behavior)
        if isinstance(omega, (float, int, np.floating)):
            omega = np.array([float(omega)], dtype=float)
        else:
            omega = np.asarray(omega, dtype=float)

        if lamda == 0 or kappa == 0:
            return np.zeros_like(omega, dtype=complex)

        # a(τ), b(τ) pieces of C(τ)
        def a_tau(tau):
            return kappa**2 * np.real(np.exp(- lamda * self.Phi(tau)) - 1.0)  # SIGN in exponent fixed ( +λ )
        def b_tau(tau):
            return kappa**2 * np.imag(np.exp(- lamda * self.Phi(tau)) - 1.0)

        lowLim = 1e-14
        uppLim = tmax_factor / self.omega_c   # finite time cutoff (≈∞ numerically)

        # integrals
        Acos = np.zeros_like(omega, dtype=float)  # ∫ a cos
        Asin = np.zeros_like(omega, dtype=float)  # ∫ a sin
        Bcos = np.zeros_like(omega, dtype=float)  # ∫ b cos
        Bsin = np.zeros_like(omega, dtype=float)  # ∫ b sin

        for i, w in enumerate(omega):
            Acos[i] = integrate.quad(a_tau, lowLim, uppLim, weight='cos', wvar=w, limit=200)[0]
            Asin[i] = integrate.quad(a_tau, lowLim, uppLim, weight='sin', wvar=w, limit=200)[0]
            Bcos[i] = integrate.quad(b_tau, lowLim, uppLim, weight='cos', wvar=w, limit=200)[0]
            Bsin[i] = integrate.quad(b_tau, lowLim, uppLim, weight='sin', wvar=w, limit=200)[0]

        if sign == 'minus':          # K(ω)=∫ e^{-iωτ}C(τ)dτ
            # Re K = ∫ (a cos + b sin),  Im K = ∫ (-a sin + b cos)
            Re = Acos + Bsin
            Im = -Asin + Bcos
        else:                        # sign == 'plus', K(ω)=∫ e^{+iωτ}C(τ)dτ
            # Re K = ∫ (a cos − b sin),  Im K = ∫ ( a sin + b cos)
            Re = Acos - Bsin
            Im =  Asin + Bcos

        return Re + 1j*Im



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




