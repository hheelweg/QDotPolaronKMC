import numpy as np
import scipy.linalg as la
import pandas as pd


"""
This file contains some helper functions that will be called
"""


# diagonalize Hamiltonian
# NOTE : might want to make this more efficient with GPU/torch etc.
def diagonalize(H, S=None):
    """
    Diagonalize a real, symmetrix matrix and return sorted results.
    
    Return the eigenvalues and eigenvectors (column matrix) 
    sorted from lowest to highest eigenvalue.
    """
    E,C = la.eigh(H,S)
    E = np.real(E)
    #C = np.real(C)

    idx = E.argsort()
    #idx = (-E).argsort()
    E = E[idx]
    C = C[:,idx]

    return E,C


def export_msds(times, msds, file_name = "msds.csv"):

    # obtain mean MSD averaged over all realizations of noise
    msds_mean = np.mean(msds, axis = 0)

    # create df and save it
    data = np.column_stack([times, msds_mean.T, msds.T])
    columns = ["time"] + ["ave. msd"] + [f"msd lattice_{i}" for i in range(msds.shape[0])]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_name, index=False)


def get_diffusivity(msd, times, dim):
    t = np.asarray(times); y = np.asarray(msd)
    if tmin is None: tmin = np.quantile(t, 0.25)
    if tmax is None: tmax = np.quantile(t, 0.75)
    mask = (t >= tmin) & (t <= tmax)
    T, Y = t[mask], y[mask]
    A = np.vstack([T, np.ones_like(T)]).T
    # least squares
    coef, residuals, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    b, a = coef[0], coef[1]
    # goodness & stderr
    yfit = A @ coef
    ss_res = np.sum((Y - yfit)**2)
    ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    dof = max(len(T) - 2, 1)
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    b_stderr = np.sqrt(cov[0,0])
    D = b / (2.0 * dim)
    return D, b, a, b_stderr, r2

