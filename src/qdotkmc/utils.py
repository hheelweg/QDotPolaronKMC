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


def get_diffusivity(msd, times, dim, slope_tol=0.05, min_points=10):
    msd = np.asarray(msd, dtype=float)
    times = np.asarray(times, dtype=float)

    # Compute local log-log slopes
    log_t = np.log(times[1:])
    log_msd = np.log(msd[1:])
    slope_local = np.gradient(log_msd, log_t)

    # Identify indices where slope ~ 1
    mask = np.abs(slope_local - 1.0) < slope_tol
    if not np.any(mask):
        raise ValueError("No diffusive regime found with given slope tolerance.")

    # Find largest consecutive block in mask
    max_len = 0
    start_idx = 0
    curr_len = 0
    curr_start = 0
    for i, val in enumerate(mask):
        if val:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > max_len:
                max_len = curr_len
                start_idx = curr_start
        else:
            curr_len = 0

    if max_len < min_points:
        raise ValueError("Diffusive regime too short.")

    fit_slice = slice(start_idx, start_idx + max_len)
    t_fit = times[fit_slice]
    msd_fit = msd[fit_slice]

    # Fit MSD ~ a + b t in linear space
    A = np.vstack([t_fit, np.ones_like(t_fit)]).T
    coef, _, _, _ = np.linalg.lstsq(A, msd_fit, rcond=None)
    b, a = coef[0], coef[1]

    D = b / (2 * dim)
    slope_mean = np.mean(slope_local[fit_slice])

    return D, slope_mean, a, (t_fit[0], t_fit[-1])

