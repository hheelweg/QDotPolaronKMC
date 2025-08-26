import numpy as np
import scipy.linalg as la
import pandas as pd
from threadpoolctl import threadpool_limits
from . import lattice


# diagonalize Hamiltonian
# NOTE : might want to make this more efficient with GPU/torch etc.
def diagonalize(H, S=None, 
                cpu_threads: int = 16,
                cpu_driver: str = "evr",
                uplo: str = "L"):
    """
    Diagonalize a real, symmetrix matrix and return sorted results.
    
    Return the eigenvalues and eigenvectors (column matrix) 
    sorted from lowest to highest eigenvalue.
    """
    # CPU path (MKL/OpenBLAS). Pin to a reasonable #threads just for this call.
    with threadpool_limits(limits=cpu_threads):
        # SciPy ≥1.10 lets you pick driver; 'evr' (MRRR) often fastest & stable.
        E, C = la.eigh(H, driver=cpu_driver, lower=(uplo == "L"))
    
    idx = np.argsort(E)
    E = E[idx]
    C = C[:, idx]
    return E,C


def export_msds(times, msds, file_name = "msds.csv"):

    # obtain mean MSD averaged over all realizations of noise
    msds_mean = np.mean(msds, axis = 0)

    # create df and save it
    data = np.column_stack([times, msds_mean.T, msds.T])
    columns = ["time"] + ["ave. msd"] + [f"msd lattice_{i}" for i in range(msds.shape[0])]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_name, index=False)


# get diffusivity from single (or pooled) MSD trajectory and times array via linear regression
# NOTE : this is conceptually identical to former get_diffusivity_hh function
def get_diffusivity(msd, times, dim, tail_frac=1.0):
    """
    Estimate the diffusion coefficient from a single MSD curve using a linear fit. The
    estimated diffusion coefficient, computed via the Einstein relation:
    MSD(t) ≈ 2 * dim * D * t   (in the diffusive regime). Here D is extracted from
    the slope of the MSD vs. t fit. 

    Inputs
    ----------
    msd : array_like, shape (T,)
        Mean-squared displacement values sampled at the times given in `times`.
        This could be for a single trajectory or a pooled average. 
    times : array_like, shape (T,)
        Time values corresponding to the MSD samples.
    dim : int
        Dimensionality of the QDLattice (e.g., 2 for 2D, 3 for 3D).
    tail_frac : float, optional
        Fraction of the time points at the tail (largest times) to use for fitting.
        Must be in (0, 1]. For example, 0.5 uses the last half of the data points.

    Returns
    -------
    D : float
        Estimated diffusion coefficient.
    D_stderr : float
        Standard error of the diffusion coefficient from the linear regression.

    Notes
    -----
    - The fit is performed on MSD(t) = a + b t, with b converted to D = b / (2 * dim).
    - If the number of points in the selected tail is < 3, the last three points are used.
    - Only finite time and MSD values are used in the fit.
    """

    # (0) setup
    t = np.asarray(times, float)
    y = np.asarray(msd,   float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if t.size < 3:
        raise ValueError("Not enough points to fit.")

    # (1) pick tail (data points on which to perform linear regression on)
    q = max(0.0, min(1.0, 1.0 - tail_frac))
    tmin = np.quantile(t, q)
    sel = t >= tmin
    T, Y = t[sel], y[sel]
    if T.size < 3:
        # last resort: last 3 points
        T, Y = t[-3:], y[-3:]

    # (2) linear least squares: Y ≈ a + b T
    A = np.vstack([T, np.ones_like(T)]).T
    coef, residuals, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    b, a = coef[0], coef[1]

    # (3) slope stderr
    yfit = A @ coef
    dof = max(len(T) - 2, 1)
    ss_res = float(np.sum((Y - yfit)**2))
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    b_stderr = float(np.sqrt(cov[0, 0]))

    D = b / (2.0 * dim)
    D_stderr = b_stderr / (2.0 * dim)
    return D, D_stderr


# summary multiple diffusivities based on 
def summarize_diffusivity(msds, times, dim, tail_frac=1.0):
    """
    Inverse-variance weighted mean of D_i with standard error 1/sqrt(sum w_i)

    Notes
    --------------------------------------
    Inverse variance weighting makes sense if the different realizations (trajectories, MSD fits, etc.) 
    are not equally reliable - for example, if one trajectory's MSD fit has a small standard error because 
    it has low noise or long sampling, it should contribute more to the overall diffusivity estimate than 
    one with a large standard error.

    Inputs
    ------
    msds : 2D-array-like (R, T)
        For each of the R noise realizations (QDLattices), there is a trajectory's MSD of length T.
    times : 1D array-like (T,)
        Time values corresponding to the MSD samples.
    dim : int
        Dimensionality of the QDLattice (e.g., 2 for 2D, 3 for 3D).

    Returns
    -------
    D_mean : float
        Average diffusivity over all realizations, obtained from a linear fit to MSD(t) in the diffusive 
        regime for each realization (QDLattice).
    D_stderr : float
        Standard error of the diffusivity estimate across realizations.
    """
    # (0) setup
    msds = np.asarray(msds, float)
    R, T = msds.shape
    Ds, sDs = [], []

    # (1) get diffusivities and standard errors from get_diffusivity function
    for i in range(R):
        try:
            D_i, sD_i = get_diffusivity(msds[i], times, dim, tail_frac=tail_frac)
            if np.isfinite(D_i) and np.isfinite(sD_i) and sD_i > 0:
                Ds.append(D_i)
                sDs.append(sD_i)
        except Exception:
            continue

    Ds = np.asarray(Ds)
    sDs = np.asarray(sDs)

    # (2) inverse-variance weighting
    w = 1.0 / (sDs ** 2)
    D_weighted = float(np.sum(w * Ds) / np.sum(w))
    D_weighted_stderr = float(1.0 / np.sqrt(np.sum(w)))

    return D_weighted, D_weighted_stderr


# smallest index set capturing (1 - θ) of mass / cumulative sum
# NOTE : needed for selection of sites/polarons
def mass_core_by_theta(w_col, theta: float):
    w = np.asarray(w_col, float).ravel()
    if w.size == 0:
        return np.empty((0,), dtype=np.intp)
    order = np.argsort(w)[::-1]
    csum  = np.cumsum(w[order])
    target = (1.0 - float(theta)) * csum[-1]
    k = int(np.searchsorted(csum, target, side="left")) + 1
    return np.sort(order[:k]).astype(np.intp)


# get closest index (formerly in montecarlo.py)
# NOTE : need to implement non-periodic boundary conditions? 
def get_closest_idx(qd_lattice, pos, array, periodic=True):
    """
    Find the index in `array` closest to `pos` under periodic boundary conditions.
    """
    assert isinstance(qd_lattice, lattice.QDLattice), 'need to specify valid QDLattice instance.'

    # Vectorized periodic displacement
    delta = array - pos  # shape (N, dims)

    # Apply periodic boundary condition (minimum image convention)
    delta -= np.round(delta / qd_lattice.geom.boundary) * qd_lattice.geom.boundary

    # Compute squared distances
    dists_squared = np.sum(delta**2, axis=1)

    return np.argmin(dists_squared)



def get_ipr(Umat):
    # returns ipr of one column vector, or mean ipr of multiple column vectors
    IPRs = 1/np.sum(Umat ** 4, axis = 0)
    return np.mean(IPRs), np.std(IPRs)