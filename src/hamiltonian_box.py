import numpy as np
from scipy import integrate
from scipy import optimize
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
                 ham_sysbath, spec_density, kT, max_energy_diff = np.nan):
        # Set these shared constants
        const.kT = kT
        
        self.qd_lattice_rel = qd_lattice

        self.init_system(evals, eigstates)
        self.sysbath = ham_sysbath

        # spectral density
        self.spec=SpecDens(spec_density, max_energy_diff)
        

class SpecDens():

    def __init__(self, spec_dens_list, max_energy_diff):
        sd_type = spec_dens_list[0]
        self.bath_method = spec_dens_list[-1]
        
        if sd_type == 'cubic-exp':
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.J = self.cubic_exp
            self.low_freq_cutoff = self.omega_c/200
            self.omega_inf = 10*self.omega_c
            self.cheby_tau_cutoff = 40/self.omega_c
        elif sd_type == 'gauss-sum':
            self.heights = spec_dens_list[1]
            self.peak_nrgs = spec_dens_list[2]
            self.sds = spec_dens_list[3]
            self.J = self.gauss_sum
            self.low_freq_cutoff = max(0, np.min(self.peak_nrgs - 7 * self.sds))
            self.omega_inf = max(self.peak_nrgs + 7 * self.sds)
            self.cheby_tau_cutoff = self.omega_inf
        else:
            raise SystemExit
        
        if self.bath_method == 'exact':
            self.Phi = self.phi     # use exact phi
            self.correlationFT = self.bathCorrFT
        elif self.bath_method == 'fit':
            self.getPhiFit()        # perform phi fit
            self.qualityFit()       # assess quality of getPhiFit
            self.Phi = self.PhiFit  # use fit
            self.correlationFT = self.bathCorrFT
        elif self.bath_method == 'cheby-fit':
            self.getChebyFit()
            self.Phi = self.chebyPhiFit
            if np.isnan(max_energy_diff):
                self.correlationFT = self.bathCorrFT
            else:
                kappaSetupTime = time.time()
                self.getBathCorrFTFit(max_energy_diff)
                self.correlationFT = self.bathCorrFTFitReal
                kappaSetupTime = time.time() - kappaSetupTime
                print("kappa setup time: %f" % kappaSetupTime)
        elif self.bath_method == 'first-order':
            self.correlationFT = self.firstOrderFT
        else:
            raise SystemExit

            
    def cubic_exp(self, omega):
        """Evaluate a cubic (super-Ohmic) spectral density with an exponential
        cutoff.
        """
        w = abs(omega)
        Jw = (self.lamda/(2*self.omega_c**3))*w**3*np.exp(-w/self.omega_c)
        return Jw*(omega >= 0) - Jw*(omega < 0)
    def gauss_sum(self, omega):
        """Evaluage a spectral density that approximates the sum of several gaussian peaks"""
        if type(omega) == float or type(omega) == int or type(omega) == np.float64:
            w = np.array([np.abs(omega)])
        else:
            w = np.abs(omega)
        spec = np.zeros(np.shape(w))
        for i in np.arange(len(self.heights)):
            spec += self.heights[i] * np.exp(-0.5 * (w - self.peak_nrgs[i])**2/(self.sds[i]**2))
        # modify behavior near 0 to avoid divergent behavior
        # cutoff_nrg = min(self.peak_nrgs/10)
        cutoff_nrg = 0.001
        zero_taper = np.zeros(np.shape(w))
        taper_inds = np.nonzero(np.logical_and(w > cutoff_nrg,w < 2*cutoff_nrg))
        zero_taper[taper_inds] = (w[taper_inds] - 0.001)/0.001
        zero_taper[np.nonzero(w >= 2 * cutoff_nrg)] = 1
        spec *= zero_taper
        return spec             
        
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
    """
    # perform half-sided Fourier trasnform of bath correlation function based on Eq. (15)
    # old version, doesn't work with vectorized inputs
    def bathCorrFT(self, omega, lamda, kappa):
        
        if lamda==0:
            return 0
        else:
    
            def integrandFT_real(tau):
                return kappa**2*np.real(np.exp(-lamda*self.Phi(tau))-1)
            def integrandFT_imag(tau):
                return kappa**2*np.imag(np.exp(-lamda*self.Phi(tau))-1)
            
            # perform half-sided Fourier transform (real/imaginary parts separately)
            uppLim=np.inf
            lowLim=1E-14
            bathCorrFT_real1=integrate.quad(integrandFT_real, lowLim, uppLim, limit=200, weight='cos', wvar=omega)
            bathCorrFT_real2=integrate.quad(integrandFT_imag, lowLim, uppLim, limit=200, weight='sin', wvar=omega)
            bathCorrFT_imag1=integrate.quad(integrandFT_imag, lowLim, uppLim, limit=200, weight='cos', wvar=omega)
            bathCorrFT_imag2=integrate.quad(integrandFT_real, lowLim, uppLim, limit=200, weight='sin', wvar=omega)
    
            return bathCorrFT_real1[0]+bathCorrFT_real2[0]+1j*bathCorrFT_imag1[0]-1j*bathCorrFT_imag2[0]
    """
    
    def getPhiFit(self):
        # determine tau_max for which asolute value of phi exceeds cutoff
        cutoff = 1E-5 
        taus_range = np.arange(0.1, 400, 2)
        integrals_range = [self.phi(tau).real[0] for tau in taus_range]
        cutoff_idx = np.min(np.where(np.flip(np.asarray(integrals_range)) >= cutoff))
        taus_max = taus_range[len(taus_range)-1-cutoff_idx]
        
        # parameters for fitting (might have to adjust these from case to case)
        self.taus = np.linspace(0.1, taus_max, 2000)
        polyDegree = 20  # degree polynomial for fit
        
        # evaluate exact integral for all taus in tau
        self.integral_real=np.array([self.phi(tau).real[0] for tau in self.taus])
        self.integral_imag=np.array([self.phi(tau).imag[0] for tau in self.taus])
        
        # get inflection points for real and imaginary part of phi:
        idx_real, self.cutoff_real = get_inflection(self.taus, self.integral_real)
        idx_imag, self.cutoff_imag = get_inflection(self.taus, self.integral_imag)
        
        # until cutoff: perfrom polynomial fit
        def polynomialFit(taus, phis):
            params = np.polyfit(taus, phis, polyDegree)
            return params
        
        self.polyParams_real = polynomialFit(self.taus[:idx_real], self.integral_real[:idx_real])
        self.polyParams_imag = polynomialFit(self.taus[:idx_imag], self.integral_imag[:idx_imag])
        
        # after cutoff: perform biexponential decay fit
        def biexpFit(taus, phis):
            bounds = (-np.inf, np.inf)
            initial = [-1, 1,1, 1, 1,1]
            params, pcov = optimize.curve_fit(biexpDecay, xdata=taus, ydata=phis,
                                            p0= initial, bounds=bounds, maxfev=10000, method="trf") 
            return params
        
        self.expParam_real = biexpFit(self.taus[idx_real:]-self.cutoff_real, self.integral_real[idx_real:])
        self.expParam_imag = biexpFit(self.taus[idx_imag:]-self.cutoff_imag, self.integral_imag[idx_imag:])
        
        
    def getChebyFit(self):
        degree = 100
        self.chebyphi_real = np.polynomial.chebyshev.Chebyshev.interpolate(lambda tau: np.real(self.phi(tau)), degree, [0, self.cheby_tau_cutoff])
        self.chebyphi_im = np.polynomial.chebyshev.Chebyshev.interpolate(lambda tau: np.imag(self.phi(tau)), degree, [0, self.cheby_tau_cutoff])
        
        # after cutoff: perform biexponential decay fit
        def biexpFit(taus, phis):
            bounds = (-np.inf, np.inf)
            initial = [phis[0], 0,0, -phis[0], 0,0]
            params, pcov = optimize.curve_fit(biexpDecay, xdata=taus, ydata=phis,
                                            p0= initial, bounds=bounds, maxfev=10000, method="trf") 
            return params
        self.biexpFitTaus = np.linspace(self.cheby_tau_cutoff, self.cheby_tau_cutoff * 2, 100)
        self.expParam_real = biexpFit(self.biexpFitTaus - self.cheby_tau_cutoff, np.real(self.phi(self.biexpFitTaus)))
        self.expParam_imag = biexpFit(self.biexpFitTaus - self.cheby_tau_cutoff, np.imag(self.phi(self.biexpFitTaus)))
    
    def getBathCorrFTFit(self, max_energy_diff):
        degree = 25
        self.chebyk_real = []
        # note: calculates bath corr FT divided by kappa^2
        # kappa^2 is added later for calculating final term
        for lamda in [-2, -1, 0, 1, 2]:
            self.chebyk_real.append(np.polynomial.chebyshev.Chebyshev.interpolate(lambda omega: np.real(self.bathCorrFT(omega, lamda, 1)), degree, [-max_energy_diff, max_energy_diff]))
    
    def qualityFit(self):
        # compute the quality of the fitted function for phi by computing the mean least squared deviations
        integral_fit = np.array([self.PhiFit(tau) for tau in self.taus])
        # real part and imaginary part
        mean_sq_error = [np.mean(np.square(self.integral_real-integral_fit.real)), 
                         np.mean(np.square(self.integral_imag-integral_fit.imag))]
        # raise warning if prespecified fit_tolerance is exceeded
        fit_tolerance = 1E-4
        if any(i >= fit_tolerance for i in mean_sq_error):
            warnings.warn("fit tolerance {} exceeded! Please check fit parameters. Results might be wrong!".format(fit_tolerance),
                          stacklevel = 2)
        
    def PhiFit(self, tau):
        # real part of phi
        if tau <= self.cutoff_real:
            phi_real = np.poly1d(self.polyParams_real)(tau)
        else:
            phi_real = biexpDecay(tau-self.cutoff_real, *self.expParam_real)
        # imaginary part of phi
        if tau <= self.cutoff_imag:
            phi_imag = np.poly1d(self.polyParams_imag)(tau)
        else:
            phi_imag = biexpDecay(tau-self.cutoff_imag, *self.expParam_imag)
        
        return phi_real + 1j*phi_imag
    
    def chebyPhiFit(self, tau):
        if tau <= self.cheby_tau_cutoff:
            return self.chebyphi_real(tau) + 1j * self.chebyphi_im(tau)
        else:
            return biexpDecay(tau-self.cheby_tau_cutoff, *self.expParam_real) + 1j * biexpDecay(tau-self.cheby_tau_cutoff, *self.expParam_imag)
    
    # only returns real portion
    def bathCorrFTFitReal(self, omega, lamda, kappa):
        return kappa**2 * self.chebyk_real[int(lamda) + 2](omega)
    
    def plotFit(self):
        import matplotlib.pyplot as plt
        taus = np.linspace(0, self.cheby_tau_cutoff * 2, 400)
        exactphi = self.phi(taus)
        realchebyphi = np.zeros(400)
        imagchebyphi = np.zeros(400)
        for i in np.arange(400):
            realchebyphi[i] = np.real(self.chebyPhiFit(taus[i]))
            imagchebyphi[i] = np.imag(self.chebyPhiFit(taus[i]))
        plt.plot(taus, np.real(exactphi))
        plt.plot(taus, realchebyphi)
        plt.show()
        plt.plot(taus, np.real(exactphi) - realchebyphi)
        plt.show()
        plt.plot(taus, np.imag(exactphi))
        plt.plot(taus, imagchebyphi)
        plt.show()
        plt.plot(taus, np.imag(exactphi) - imagchebyphi)
        return
    
    # first order approximation to the bath correlation function in Eq. (16)
    def firstOrderFT(self, omega, lamda, kappa):
        
        def re_bath_corr(omega):
            def coth(x):
                return 1.0/np.tanh(x)

            beta = 1.0/const.kT
            omega += 1e-14
            n_omega = 0.5*(coth(beta*omega/2) - 1.0)
            return (self.J(omega)*(n_omega+1))/omega**2
        
        omega = -omega
        ppv = integrate.quad(re_bath_corr, 
                             -self.omega_inf, self.omega_inf,
                             limit=1000, weight='cauchy', wvar=omega)
        ppv = -ppv[0]
        return -kappa**2*lamda*(re_bath_corr(omega) + (1j/np.pi)*ppv)
        




# auxiliary functions for method = 'fit'
def get_inflection(x, y):
    """
    compute rightmost inflection points for y
    ----------
    x : 1D-array 
    y : 1D-array (same length as x)

    Returns
    -------
    rightmost inflection point as IDX and as x value
    """
    second_deriv = np.gradient(np.gradient(y))
    inflectionsIDX = np.where(np.diff(np.sign(second_deriv)))[0] # returns all inflection points as indices in y
    inflection = x[inflectionsIDX[len(inflectionsIDX)-1]]
    return inflectionsIDX[len(inflectionsIDX)-1], inflection


def biexpDecay(x, a,b,c,d,e,f):
    return a * np.exp(-b*(x-c))-d * np.exp(-e*(x-f))



