import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.signal import hilbert
from scipy.interpolate import interp1d
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
        

class SpecDens():

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




