import numpy as np
import scipy.linalg as la
import pandas as pd
from typing import Callable, Dict, Any, Tuple


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


# get diffusivity from single msd trajectory and times array
def get_diffusivity(msd, times, dim, tail_frac=1.0):
    """
    Minimal single-curve diffusivity:
      - Fit MSD(t) ~ a + b t on the last `tail_frac` of points.
      - Return D = b/(2*dim) and its standard error.
    """
    t = np.asarray(times, float)
    y = np.asarray(msd,   float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if t.size < 3:
        raise ValueError("Not enough points to fit.")

    # pick tail
    q = max(0.0, min(1.0, 1.0 - tail_frac))
    tmin = np.quantile(t, q)
    sel = t >= tmin
    T, Y = t[sel], y[sel]
    if T.size < 3:
        # last resort: last 3 points
        T, Y = t[-3:], y[-3:]

    # linear least squares: Y ≈ a + b T
    A = np.vstack([T, np.ones_like(T)]).T
    coef, residuals, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    b, a = coef[0], coef[1]

    # slope stderr
    yfit = A @ coef
    dof = max(len(T) - 2, 1)
    ss_res = float(np.sum((Y - yfit)**2))
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    b_stderr = float(np.sqrt(cov[0, 0]))

    D = b / (2.0 * dim)
    D_stderr = b_stderr / (2.0 * dim)
    return D, D_stderr



# summary multiple diffusivities
def summarize_diffusivity(msds, times, dim, tail_frac=1.0):
    """
    Minimal many-curve summary:
      - Fit each realization with fit_diffusivity_simple.
      - Inverse-variance weighted mean of D_i with standard error 1/sqrt(sum w_i).
    Returns: (D_weighted, D_weighted_stderr, n_used)
    """
    msds = np.asarray(msds, float)
    R, T = msds.shape
    Ds, sDs = [], []

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

    # inverse-variance weighting
    w = 1.0 / (sDs ** 2)
    D_weighted = float(np.sum(w * Ds) / np.sum(w))
    D_weighted_stderr = float(1.0 / np.sqrt(np.sum(w)))

    return D_weighted, D_weighted_stderr, len(Ds)
